from __future__ import annotations
from typing import Dict, List, Any, Iterable
import io, os, base64
import networkx as nx
from PIL import Image

# ==== thông số ảnh ====
THUMB_SIZE = (224, 126)   
MONTAGE_GRID = (2, 2)     
CAPTION_MAX_CHARS = 70
SHOW_TIME = True

def _safe_resample():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return Image.LANCZOS

def _encode_data_url(img: Image.Image, fmt="WEBP", quality=90) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "WEBP":
        img.save(buf, format="WEBP", quality=quality, method=6)
        mime = "image/webp"
    else:
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _thumb_data_url(path: str, size=THUMB_SIZE) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return ""
    im.thumbnail(size, _safe_resample())
    return _encode_data_url(im, fmt="WEBP", quality=90)

def _montage_data_url(paths: Iterable[str], grid=MONTAGE_GRID, cell_size=THUMB_SIZE) -> str:
    paths = [p for p in paths if p and os.path.exists(p)]
    cols, rows = grid
    cw, ch = cell_size
    canvas = Image.new("RGB", (cols*cw, rows*ch), (245,245,245))
    for i, p in enumerate(paths[:cols*rows]):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im.thumbnail((cw, ch), _safe_resample())
        x = (i % cols) * cw
        y = (i // cols) * ch
        canvas.paste(im, (x, y))
    return _encode_data_url(canvas, fmt="WEBP", quality=90)

def _short(s: str, n=CAPTION_MAX_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n-1] + "…"

def _leaf_desc(G: nx.DiGraph, node: str) -> List[str]:
    "Danh sách id leaf con/cháu theo thời gian"
    desc = [d for d in nx.descendants(G, node) if G.nodes[d].get("kind") == "leaf"]
    desc = sorted(desc, key=lambda x: G.nodes[x].get("time_s", 0.0))
    return desc

def _title_for_node(nid: str, data: Dict[str, Any]) -> str:
    kind = data.get("kind", "node")
    if kind == "leaf":
        t = float(data.get("time_s", 0.0))
        return f"{nid} [t={t:.1f}s]" if SHOW_TIME else str(nid)
    else:
        t0 = float(data.get("start_s", 0.0))
        t1 = float(data.get("end_s", 0.0))
        return f"{nid} [{t0:.1f}s–{t1:.1f}s]" if SHOW_TIME else str(nid)

def _image_url_for_node(G: nx.DiGraph, nid: str, data: Dict[str, Any]) -> str:
    kind = data.get("kind", "node")
    if kind == "leaf":
        return _thumb_data_url(data.get("image_path", ""))
    children = list(G.successors(nid))
    leaf_imgs: List[str] = []
    for c in children:
        cdata = G.nodes[c]
        if cdata.get("kind") == "leaf":
            leaf_imgs.append(cdata.get("image_path", ""))
        else:
            dleaf = _leaf_desc(G, c)
            if dleaf:
                leaf_imgs.append(G.nodes[dleaf[0]].get("image_path", ""))
    leaf_imgs = [p for p in leaf_imgs if p]
    return _montage_data_url(leaf_imgs) if leaf_imgs else ""

def _node_payload(G: nx.DiGraph, nid: str) -> Dict[str, Any]:
    data = G.nodes[nid]
    return {
        "id": nid,
        "title": _title_for_node(nid, data),
        "caption": _short(data.get("caption", "") if data.get("kind")=="leaf" else data.get("summary","")),
        "img": _image_url_for_node(G, nid, data),
        "kind": data.get("kind", "node"),
    }

def _to_nested(G: nx.DiGraph, nid: str) -> Dict[str, Any]:
    payload = _node_payload(G, nid)
    kids = list(G.successors(nid))
    if kids:
        payload["children"] = [_to_nested(G, c) for c in kids]
    return payload

def to_nested_tree(G: nx.DiGraph) -> Dict[str, Any]:
    "Chọn root (node in-degree=0). Nếu nhiều, tạo root giả."
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        try:
            roots = [min(G.nodes, key=lambda x: G.nodes[x].get("time_s", 0.0))]
        except Exception:
            roots = [next(iter(G.nodes))]
    if len(roots) == 1:
        return _to_nested(G, roots[0])
    fake = {"id": "root", "title": "root", "caption": "", "img": "", "kind": "node",
            "children": [_to_nested(G, r) for r in roots]}
    return fake
