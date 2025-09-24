import os
import io
import tempfile
import base64
import re  
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
import networkx as nx
from PIL import Image

from src.config import settings
from src.video.preprocess import extract_frames
from src.features.clip_extractor import CLIPFeatureExtractor
from src.features.captioner import generate_captions
from src.tree.builder import Leaf, build_adaptive_tree, save_tree

from src.utils.tree_export import to_nested_tree
from src.utils.d3_viewer import build_d3_html
from streamlit.components.v1 import html

from src.llm.query import rank_leaves_by_query, gather_context_from_topk, openai_answer


st.set_page_config(page_title="Adaptive Video Tree for LLM", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 3.6rem; }
h1, h2, h3 { letter-spacing: .2px; }
.kbd { background:#f5f5f7;border-radius:6px;padding:2px 6px;border:1px solid #e5e7eb;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco;}
.small-note { color:#64748b;font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎄 Adaptive Tree-based Video Representation for LLM")


def _ensure_defaults():
    ss = st.session_state
    ss.setdefault("pending_video_path", "")  
    ss.setdefault("video_path", "")         
    ss.setdefault("video_fp", "")
    ss.setdefault("frames", None)      
    ss.setdefault("embs", None)         
    ss.setdefault("captions", None)     
    ss.setdefault("G", None)            
    ss.setdefault("messages", [])       
    ss.setdefault("last_rank", None)    
    ss.setdefault("processing", False)  

def _file_fp(p: str) -> str:
    try:
        stat = Path(p).stat()
        return f"{stat.st_size}-{int(stat.st_mtime)}"
    except Exception:
        return Path(p).name

def _reset_video_state(keep_chat: bool = True):
    keys = ["frames", "embs", "captions", "G", "last_rank"]
    for k in keys:
        st.session_state.pop(k, None)
    if not keep_chat:
        st.session_state["messages"] = []

_ensure_defaults()


with st.expander("⚙️ Settings (technical)", expanded=False):
    target_fps = st.number_input("Sampling FPS", value=settings.TARGET_FPS, step=0.1)
    sim_thr = st.slider("Merge similarity threshold", 0.70, 0.99,
                        value=settings.SIMILARITY_THRESHOLD, step=0.01)
    max_frames = st.number_input("Max frames (0 = unlimited)", value=0)
    captioning = st.checkbox("Generate offline captions (BLIP)",
                             value=settings.CAPTIONING_ENABLED)
    topk_ctx = st.slider("Top-K frames for context", 1, 20, 5)


@st.cache_resource(show_spinner=False)
def load_clip() -> CLIPFeatureExtractor:
    return CLIPFeatureExtractor()

@st.cache_data(show_spinner=True)
def compute_embeddings(image_paths: list[str]) -> np.ndarray:
    clip = load_clip()
    return clip.encode_images(image_paths)

@st.cache_data(show_spinner=False)
def _thumb_data_url(src_path: str, size=(224, 126), quality=90) -> str:
    """Tạo thumbnail sắc nét và trả về data URL (base64)."""
    img = Image.open(src_path).convert("RGB")
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS
    img.thumbnail(size, resample)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/webp;base64,{b64}"

def _vi_translate_captions(caps: list[str]) -> list[str]:
    """Dịch caption sang tiếng Việt (nếu có API key). Luôn chạy để cây hiển thị VI."""
    if not settings.OPENAI_API_KEY or not any(caps):
        return caps
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(caps))
    prompt = (
        "Bạn là một trợ lý dịch thuật. Hãy dịch các caption sau sang TIẾNG VIỆT, "
        "giữ nguyên thứ tự, ngắn gọn, tự nhiên. Chỉ trả về danh sách theo số thứ tự.\n\n"
        + numbered
    )
    out = openai_answer(prompt, "") or ""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        stripped = l
        for sep in [". ", ") ", ".", ")", "- "]:
            head = stripped[:4]
            if any(head.startswith(f"{d}{sep}") for d in "123456789"):
                idx = stripped.find(sep)
                if idx != -1 and idx <= 3:
                    stripped = stripped[idx+len(sep):].strip()
                break
        cleaned.append(stripped)
    return cleaned if len(cleaned) == len(caps) else caps

def _compress_context(ctx: str, max_chars: int = 3500) -> str:
    """Cắt ngắn ngữ cảnh để LLM phản hồi nhanh hơn (không đổi logic)."""
    if len(ctx) <= max_chars:
        return ctx
    head = ctx[: max_chars - 200]
    tail = ctx[-200:]
    return head + "\n...\n" + tail


def _leaf_index_from_node(node_id, data) -> Optional[int]:
    """Lấy chỉ số leaf từ thuộc tính 'idx' hoặc tên node 'L{n}'."""
    if "idx" in data and isinstance(data["idx"], (int, np.integer)):
        return int(data["idx"])
    m = re.match(r"^L(\d+)$", str(node_id))
    return int(m.group(1)) if m else None

def _force_vi_captions_on_graph(G: nx.DiGraph, captions_vi: list[str]) -> int:
    """
    Gán lại caption/label/title (và các alias thường gặp) cho toàn bộ leaf = tiếng Việt.
    Trả về số node leaf đã cập nhật.
    """
    changed = 0
    for n, d in G.nodes(data=True):
        if d.get("kind") == "leaf":
            li = _leaf_index_from_node(n, d)
            if li is not None and 0 <= li < len(captions_vi):
                cap_vi = captions_vi[li] or d.get("caption", "")
                for key in ("caption", "label", "title", "text", "name", "desc", "description"):
                    G.nodes[n][key] = cap_vi
                changed += 1
    return changed

tab_video, tab_tree, tab_qa, tab_sum = st.tabs(
    ["🎥 Video", "🌳 Cây video", "💬 Hỏi – đáp", "📝 Tóm tắt"]
)

with tab_video:
    st.markdown("**Tải video** (chấp nhận: mp4, mov, mkv, avi)")
    uploaded = st.file_uploader("Chọn video", type=["mp4", "mov", "mkv", "avi"], label_visibility="collapsed")

    preview_path = ""
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            st.session_state["pending_video_path"] = tmp.name
        preview_path = st.session_state["pending_video_path"]
    elif st.session_state.get("video_path"):
        preview_path = st.session_state["video_path"]

    if preview_path:
        st.video(preview_path)

    col1, col2 = st.columns([1, 4])
    with col1:
        start = st.button("🚀 Nạp video")
    with col2:
        if not st.session_state.get("pending_video_path") and not st.session_state.get("video_path"):
            st.caption("Hãy chọn một video rồi bấm **Nạp video** để bắt đầu.")
        else:
            st.caption("Bấm **Nạp video** để xử lý.")

    if start:
        if st.session_state.get("pending_video_path"):
            new_path = st.session_state["pending_video_path"]
            fp = _file_fp(new_path)
            if fp != st.session_state.get("video_fp"):
                _reset_video_state(keep_chat=True)
                st.session_state["video_path"] = new_path
                st.session_state["video_fp"] = fp
            else:
                st.session_state["video_path"] = new_path
        elif st.session_state.get("video_path"):
            pass
        else:
            st.warning("Vui lòng chọn video trước khi nạp.")
            st.stop()

        st.session_state["processing"] = True

    if st.session_state.get("processing"):
        video_path = st.session_state["video_path"]

        out_dir = os.path.join(
            settings.DATA_DIR,
            settings.FRAMES_DIRNAME,
            f"{Path(video_path).stem}_{st.session_state['video_fp'].replace(':','-')}"
        )
        os.makedirs(out_dir, exist_ok=True)

        st.info("🔧 Đang trích xuất khung hình…")
        frames = extract_frames(
            video_path,
            out_dir=out_dir,
            target_fps=target_fps,
            max_frames=None if max_frames == 0 else int(max_frames),
            frame_size=settings.FRAME_SIZE,
        )
        st.success(f"✅ Đã trích xuất {len(frames)} khung hình.")

        image_paths = [f.path for f in frames]

        st.info("🧠 Đang tính embedding CLIP…")
        embs = compute_embeddings(image_paths)
        st.success("✅ Hoàn tất embedding.")

        if captioning:
            st.info("📝 Đang sinh mô tả khung (captions)…")
            caps_en = generate_captions(image_paths)
            st.success("✅ Đã tạo captions (EN).")
            st.info("🌐 Đang dịch captions sang Tiếng Việt…")
            captions = _vi_translate_captions(caps_en)
            st.success("✅ Đã dịch captions (VI).")
        else:
            captions = ["" for _ in image_paths]
            st.info("⏭️ Bỏ qua tạo captions (đang tắt).")

        st.session_state["frames"] = frames
        st.session_state["embs"] = embs
        st.session_state["captions"] = captions

        st.info("🌲 Đang xây dựng cây…")
        leaves = [
            Leaf(idx=i, time_s=float(frames[i].time_s),
                 image_path=image_paths[i], emb=embs[i], caption=captions[i])
            for i in range(len(frames))
        ]
        G: nx.DiGraph = build_adaptive_tree(leaves, threshold=sim_thr)

        _force_vi_captions_on_graph(G, captions)

        st.session_state["G"] = G
        save_tree(G, os.path.join(settings.DATA_DIR, settings.TREE_JSON))
        st.success("✅ Hoàn tất xây dựng cây.")

        st.session_state["processing"] = False
        st.success("🎉 Xử lý xong! Chuyển sang tab **Cây video** hoặc **Hỏi – đáp** để xem.")

with tab_tree:
    if st.session_state.get("G") is not None:
        tree_json = to_nested_tree(st.session_state["G"]) 
        viewer_html = build_d3_html(
            tree_json,
            width=1200,
            height=700,
            thumb_w=224,
            thumb_h=126
        )
        html(viewer_html, height=720, scrolling=True)

with tab_qa:
    if st.session_state.get("frames") is not None:
        st.markdown("**Đặt câu hỏi về video** *(LLM trả lời bằng Tiếng Việt)*")

        for role, content in st.session_state["messages"]:
            with st.chat_message(role):
                st.markdown(content)

        q = st.chat_input("Nhập câu hỏi của bạn…")
        if q:
            st.session_state["messages"].append(("user", q))

            ranking = rank_leaves_by_query(
                st.session_state["G"], q, load_clip(), st.session_state["embs"]
            )
            st.session_state["last_rank"] = ranking
            ctx_raw = gather_context_from_topk(st.session_state["G"], ranking, k=topk_ctx)

            ctx = _compress_context(ctx_raw, max_chars=3500)
            prompt_vi = f"Hãy trả lời ngắn gọn, rõ ràng bằng **TIẾNG VIỆT**.\n\nCâu hỏi: {q}"

            ans = openai_answer(prompt_vi, ctx) or "Xin lỗi, mình chưa có câu trả lời phù hợp."
            st.session_state["messages"].append(("assistant", ans))

            with st.chat_message("assistant"):
                st.markdown(ans)

with tab_sum:
    if st.session_state.get("frames") is not None:
        st.markdown("**Tóm tắt video (đoạn văn, tiếng Việt)**")
        if st.button("Tóm tắt video"):
            n = len(st.session_state["frames"])
            ones = list(enumerate(np.ones(n, dtype=float)))
            ctx_raw = gather_context_from_topk(st.session_state["G"], ones, k=topk_ctx)
            ctx = _compress_context(ctx_raw, max_chars=3500)

            prompt_sum = (
                "Hãy viết một đoạn văn tóm tắt ngắn gọn bằng **TIẾNG VIỆT** (khoảng 4–6 câu). "
                "Giữ mạch kể tự nhiên, có kết nối giữa các câu, **không** dùng gạch đầu dòng, "
                "tránh lặp ý. "
                "Tập trung vào các cảnh/đoạn nổi bật và diễn tiến chính."
            )

            summary = openai_answer(prompt_sum, ctx) or "Không tạo được tóm tắt."
            st.markdown(summary)
