from __future__ import annotations
from typing import Dict
import networkx as nx

def to_graphviz_dot(G: nx.DiGraph) -> str:
    """Xuất Graphviz DOT string từ đồ thị."""
    lines = ["digraph VideoTree {", '  node [shape=box, style="rounded"];']
    for n, data in G.nodes(data=True):
        label = data.get("label", str(n)).replace("\n", "\\n")
        lines.append(f'  "{n}" [label="{label}"];')
    for u, v, data in G.edges(data=True):
        elabel = data.get("label", "")
        if elabel:
            lines.append(f'  "{u}" -> "{v}" [label="{elabel}"];')
        else:
            lines.append(f'  "{u}" -> "{v}";')
    lines.append("}")
    return "\n".join(lines)
