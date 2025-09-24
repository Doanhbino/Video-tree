from __future__ import annotations
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx

from ..features.clip_extractor import CLIPFeatureExtractor
from ..config import settings

def _normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize theo từng hàng (an toàn nếu đã normalize trước đó)."""
    if x.ndim != 2:
        raise ValueError("leaf_embs phải là ma trận 2D")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

def _normalize_vec(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize vector 1D."""
    if v.ndim != 1:
        v = v.reshape(-1)
    n = np.linalg.norm(v) + eps
    return v / n

def _leaf_index_from_node(node_id: str, data: dict) -> Optional[int]:
    """Ưu tiên lấy idx từ thuộc tính 'idx', nếu không có thì parse từ tên node 'L{num}'."""
    if "idx" in data and isinstance(data["idx"], (int, np.integer)):
        return int(data["idx"])
    m = re.match(r"^L(\d+)$", str(node_id))
    if m:
        return int(m.group(1))
    return None

def rank_leaves_by_query(
    G: nx.DiGraph,
    query: str,
    clip: CLIPFeatureExtractor,
    leaf_embs: np.ndarray,
    topk: Optional[int] = None,
) -> list[tuple[int, float]]:
    """
    Xếp hạng leaf theo độ tương đồng CLIP với câu hỏi.
    - Chuẩn hoá cả query embedding và leaf embeddings → cos sim chuẩn.
    - Vector hoá hoàn toàn (rất nhanh).
    - Nếu truyền topk, dùng argpartition cho nhanh; mặc định trả về full ranking.

    Trả về: danh sách (leaf_idx, score) giảm dần theo score.
    """
    if leaf_embs.size == 0:
        return []

    q = clip.encode_text([query])[0].astype(np.float32, copy=False)
    q = _normalize_vec(q)

    E = leaf_embs.astype(np.float32, copy=False)
    E = _normalize_rows(E)

    scores = E @ q  

    n = scores.shape[0]
    if topk is not None and 0 < topk < n:
        idx_part = np.argpartition(-scores, topk - 1)[:topk]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]
    else:
        idx_sorted = np.argsort(-scores)

    ranking = [(int(i), float(scores[i])) for i in idx_sorted]
    return ranking


def gather_context_from_topk(
    G: nx.DiGraph,
    ranking: list[tuple[int, float]],
    k: int = 5
) -> str:
    """
    Lấy ngữ cảnh từ Top-K leaf dựa trên ranking (idx, score).
    Ghép theo dạng:
      [t=12.3s] mô tả
    """
    leaf_map: dict[int, tuple[float, str]] = {}
    for n, d in G.nodes(data=True):
        if d.get("kind") == "leaf":
            li = _leaf_index_from_node(n, d)
            if li is not None:
                t = float(d.get("time_s", 0.0))
                cap = str(d.get("caption", "") or "").strip()
                leaf_map[li] = (t, cap)

    lines: List[str] = []
    for idx, _score in ranking[:k]:
        if idx in leaf_map:
            t, cap = leaf_map[idx]
            cap = cap if cap else "không có mô tả"
            lines.append(f"[t={t:.1f}s] {cap}")

    return "\n".join(lines)


def _pick_model() -> str:
    """
    Nếu có cấu hình model nhanh (OPENAI_FAST_MODEL) thì ưu tiên, ngược lại dùng OPENAI_MODEL.
    Cho phép tối ưu độ trễ mà không đổi API.
    """
    fast = getattr(settings, "OPENAI_FAST_MODEL", None)
    if isinstance(fast, str) and fast.strip():
        return fast.strip()
    return settings.OPENAI_MODEL

def _squash(text: str, max_chars: int = 4000) -> str:
    """Cắt ngắn ngữ cảnh để giảm độ trễ khi gọi LLM (giữ đầu/cuối)."""
    if not text or len(text) <= max_chars:
        return text
    head = text[: max_chars - 200]
    tail = text[-200:]
    return head + "\n...\n" + tail


def openai_answer(
    query: str,
    context: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """
    Trả lời bằng OpenAI Chat Completions (tiếng Việt).
    - Nén ngữ cảnh để phản hồi nhanh hơn.
    - Giới hạn max_tokens & temperature hợp lý.
    - Tự chọn model nhanh nếu có (OPENAI_FAST_MODEL), fallback OPENAI_MODEL.
    """
    try:
        from openai import OpenAI
    except Exception:
        return ("[LLM tắt] Thư viện OpenAI chưa sẵn sàng. "
                "Hãy cài đặt gói 'openai' và thiết lập API key.")

    if not settings.OPENAI_API_KEY:
        return ("[LLM tắt] Thiếu OPENAI_API_KEY trong .env.\n"
                f"Mẫu ngữ cảnh (rút gọn):\n{_squash(context, 1000)}")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = _pick_model()

    ctx = _squash(context, max_chars=4000)
    system = "Bạn là trợ lý phân tích video. Trả lời ngắn gọn, rõ ràng, trích mốc thời gian khi phù hợp."
    user = f"Ngữ cảnh (captions + timestamps):\n{ctx}\n\n---\n\nCâu hỏi: {query}\nTrả lời bằng TIẾNG VIỆT:"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def summarize_video(
    context: str,
    *,
    max_tokens: int = 300,
    temperature: float = 0.3,
) -> str:
    """
    Tóm tắt video (tiếng Việt) dựa trên ngữ cảnh captions + timestamps.
    """
    try:
        from openai import OpenAI
    except Exception:
        return ("[LLM tắt] Thư viện OpenAI chưa sẵn sàng. "
                "Hãy cài đặt gói 'openai' và thiết lập API key.")

    if not settings.OPENAI_API_KEY:
        return ("[LLM tắt] Thiếu OPENAI_API_KEY trong .env.\n"
                f"Mẫu ngữ cảnh:\n{_squash(context, 1000)}")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = _pick_model()

    ctx = _squash(context, max_chars=4000)
    system = "Bạn là trợ lý phân tích video. Viết tóm tắt mạch lạc, súc tích."
    user = (
        "Hãy tóm tắt nội dung video bằng **TIẾNG VIỆT**, 4–6 gạch đầu dòng, "
        "nêu rõ các cảnh/đoạn quan trọng và mốc thời gian khi phù hợp.\n\n"
        f"Ngữ cảnh (captions + timestamps):\n{ctx}\n\nTóm tắt:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
