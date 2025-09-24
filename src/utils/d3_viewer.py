from __future__ import annotations
import json
from typing import Any, Dict

def build_d3_html(tree: Dict[str, Any],
                  width: int = 1200,
                  height: int = 700,
                  thumb_w: int = 224,
                  thumb_h: int = 126) -> str:
    """
    Trả về HTML hiển thị cây bằng D3.js với layout NẰM NGANG (trái → phải),
    có zoom/pan và click để collapse/expand. Mỗi node hiển thị ảnh + caption.
    """
    data_json = json.dumps(tree, ensure_ascii=False)
    return f"""
<!DOCTYPE html>
<meta charset="utf-8"/>
<style>
  html, body {{ margin:0; padding:0; background:#fff; }}
  .wrap {{
    width: 100%;
    height: {height}px;
    border: 1px solid #e5e7eb;
    position: relative;
    overflow: hidden;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  }}
  .node rect {{
    fill: #ffffff;
    stroke: #d1d5db;
    rx: 10; ry: 10;
    filter: drop-shadow(0 1px 0 rgba(0,0,0,0.03));
  }}
  .node-title {{
    font-size: 12px;
    font-weight: 600;
    fill: #111827;
    text-anchor: middle;
  }}
  .node-caption {{
    font-size: 11px;
    fill: #374151;
    text-anchor: middle;
  }}
  .link {{
    fill: none;
    stroke: #9ca3af;
    stroke-width: 1.2px;
  }}
</style>
<div class="wrap" id="d3wrap"></div>

<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
  const data = {data_json};

  const W = {width}, H = {height};
  const IMG_W = {thumb_w}, IMG_H = {thumb_h};
  const NODE_W = IMG_W + 16, NODE_H = IMG_H + 52; // padding + caption space
  // Lề trái lớn hơn để có chỗ cho root
  const MARGIN = {{top: 30, right: 40, bottom: 30, left: 120}};

  const wrap = d3.select("#d3wrap");
  const svg = wrap.append("svg")
      .attr("width", "100%")
      .attr("height", H);

  const g = svg.append("g")
      .attr("transform", `translate(${{MARGIN.left}},${{MARGIN.top}})`);

  // Zoom/pan
  const zoom = d3.zoom()
      .scaleExtent([0.3, 2.5])
      .on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  // Build hierarchy
  const root = d3.hierarchy(data);
  root.x0 = 0; root.y0 = 0;

  // ---- HORIZONTAL LAYOUT ----
  // Trục X là NGANG, trục Y là DỌC
  const stepX = NODE_W + 100; // khoảng cách NGANG giữa cha-con
  const stepY = NODE_H + 30;  // khoảng cách DỌC giữa anh-em
  const tree = d3.tree().nodeSize([stepY, stepX]); // ⟵ hoán vị: [dY, dX]

  // Collapse mặc định các nhánh sâu
  function collapse(d) {{
    if (d.children) {{
      d._children = d.children;
      d.children = null;
    }}
    if (d._children) d._children.forEach(collapse);
  }}
  if (root.children) root.children.forEach(collapse);

  function toggle(d) {{
    if (d.children) {{ d._children = d.children; d.children = null; }}
    else {{ d.children = d._children; d._children = null; }}
  }}

  // Đường cong cho layout ngang
  function diagonal(s, d) {{
    // s.x/s.y, d.x/d.y đã là (hoành/tung) vì nodeSize đã hoán vị
    const path = `M ${{s.x}},${{s.y}}
                  C ${{(s.x + d.x) / 2}},${{s.y}}
                    ${{(s.x + d.x) / 2}},${{d.y}}
                    ${{d.x}},${{d.y}}`;
    return path;
  }}

  function update(source) {{
    tree(root);

    // LINKS
    const link = g.selectAll("path.link")
      .data(root.links(), d => d.target.data.id);

    link.enter().append("path")
      .attr("class", "link")
      .attr("d", d => diagonal(d.source, d.target));

    link.transition().duration(300)
      .attr("d", d => diagonal(d.source, d.target));

    link.exit().transition().duration(200).remove();

    // NODES
    const node = g.selectAll("g.node")
      .data(root.descendants(), d => d.data.id);

    const nodeEnter = node.enter().append("g")
      .attr("class", "node")
      // Vị trí khởi đầu theo toạ độ cũ (x0: ngang, y0: dọc) — layout ngang
      .attr("transform", d => `translate(${{source.x0 || 0}},${{source.y0 || 0}})`)
      .on("click", (event, d) => {{ toggle(d); update(d); }});

    nodeEnter.append("rect")
      .attr("x", -NODE_W/2).attr("y", -NODE_H/2)
      .attr("width", NODE_W).attr("height", NODE_H)
      .attr("fill", "#fff")
      .attr("stroke", "#d1d5db");

    // Tiêu đề (phía trên)
    nodeEnter.append("text")
      .attr("class", "node-title")
      .attr("y", -NODE_H/2 + 14)
      .text(d => d.data.title || d.data.id);

    // Ảnh
    nodeEnter.append("image")
      .attr("href", d => d.data.img || "")
      .attr("x", -IMG_W/2).attr("y", -IMG_H/2 + 12)
      .attr("width", IMG_W).attr("height", IMG_H)
      .attr("preserveAspectRatio", "xMidYMid slice");

    // Caption
    nodeEnter.append("text")
      .attr("class", "node-caption")
      .attr("y", IMG_H/2 + 22)
      .text(d => d.data.caption || "");

    // Cập nhật vị trí theo layout ngang (x: NGANG, y: DỌC)
    const nodeUpdate = nodeEnter.merge(node);
    nodeUpdate.transition().duration(300)
      .attr("transform", d => `translate(${{d.x}},${{d.y}})`);

    node.exit().transition().duration(200)
      .attr("transform", d => `translate(${{source.x}},${{source.y}})`)
      .remove();

    // Lưu vị trí để transition mượt
    root.each(d => {{ d.x0 = d.x; d.y0 = d.y; }});
  }}

  update(root);
</script>
"""
