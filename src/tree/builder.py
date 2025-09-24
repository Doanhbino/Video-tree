from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os, json
import numpy as np
import networkx as nx
from ..config import settings

@dataclass
class Leaf:
    idx: int
    time_s: float
    image_path: str
    emb: np.ndarray
    caption: str = ""

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def build_adaptive_tree(leaves: List[Leaf], threshold: float | None = None, max_children: int | None = None) -> nx.DiGraph:
    """Bottom-up: merge **adjacent** segments whose average embedding cosine similarity >= threshold.
    Repeat until a single root remains or no more merges possible. Keep adjacency to preserve time order.
    """
    thr = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
    maxc = max_children if max_children is not None else settings.MAX_CHILDREN_PER_NODE

    segments = [{"leaves": [i], "emb": leaves[i].emb.copy()} for i in range(len(leaves))]

    levels = []
    levels.append(segments)

    while True:
        segs = levels[-1]
        if len(segs) <= 1:
            break

        merged = []
        i = 0
        while i < len(segs):
            j = i + 1
            current = segs[i]
            while j < len(segs):
                sim = cosine_sim(current["emb"], segs[j]["emb"])
                if sim >= thr and len(current["leaves"]) + len(segs[j]["leaves"]) <= maxc * 4:
                    new_leaves = current["leaves"] + segs[j]["leaves"]
                    new_emb = (current["emb"] * len(current["leaves"]) + segs[j]["emb"] * len(segs[j]["leaves"])) / len(new_leaves)
                    new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                    current = {"leaves": new_leaves, "emb": new_emb}
                    j += 1
                else:
                    break
            merged.append(current)
            i = j

        if len(merged) == len(segs):
            break
        levels.append(merged)

    G = nx.DiGraph()
    for i, leaf in enumerate(leaves):
        G.add_node(f"L{i}", label=f"Leaf {i}\n{leaf.time_s:.1f}s", kind="leaf",
                   time_s=float(leaf.time_s), image_path=leaf.image_path, caption=leaf.caption)

    node_id = 0
    parent_nodes_for_level: list[list[str]] = []

    for level_idx, segs in enumerate(levels[::-1]): 
        current_level_nodes = []
        for seg in segs:
            nid = f"N{node_id}"
            node_id += 1
            first_leaf = seg["leaves"][0]
            last_leaf = seg["leaves"][-1]
            t0, t1 = leaves[first_leaf].time_s, leaves[last_leaf].time_s
            G.add_node(nid, label=f"Node {nid}\n[{t0:.1f}s–{t1:.1f}s]", kind="node",
                       start_s=float(t0), end_s=float(t1))
            current_level_nodes.append(nid)
        parent_nodes_for_level.append(current_level_nodes)

    for l in range(len(parent_nodes_for_level)-1):
        parents = parent_nodes_for_level[l]
        children = parent_nodes_for_level[l+1]
        ci = 0
        for p in parents:
            p_data = G.nodes[p]
            p_start, p_end = p_data.get("start_s", 0), p_data.get("end_s", 0)
            while ci < len(children):
                c = children[ci]
                c_data = G.nodes[c]
                c_start, c_end = c_data.get("start_s", 0), c_data.get("end_s", 0)
                if c_start >= p_start and c_end <= p_end:
                    G.add_edge(p, c)
                    ci += 1
                else:
                    break

    bottom_nodes = parent_nodes_for_level[-1]
    bi = 0
    for bn in bottom_nodes:
        b_data = G.nodes[bn]
        b_start, b_end = b_data.get("start_s", 0), b_data.get("end_s", 0)
        while bi < len(leaves):
            lf = leaves[bi]
            if lf.time_s >= b_start and lf.time_s <= b_end:
                G.add_edge(bn, f"L{bi}")
                bi += 1
            else:
                break

    return G

def save_tree(G: nx.DiGraph, out_path: str):
    data = nx.readwrite.json_graph.node_link_data(G)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def load_tree(path: str) -> nx.DiGraph:
    with open(path, "r") as f:
        data = json.load(f)
    return nx.readwrite.json_graph.node_link_graph(data, directed=True)
