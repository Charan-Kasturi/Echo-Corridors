"""
make_diagrams.py
=================
Small hand-laid-out SVG diagrams of the two environments.

Each diagram function takes a `bw` flag: False (default) produces the
colour version used in the README and the GitHub-rendered report; True
produces a strictly black-and-white version (weight and dash pattern carry
the distinctions colour would otherwise carry) for the print report, which
also omits the image's own baked-in title since the print layout supplies
a numbered "Figure N." caption outside the image.
"""
import random
from src import environment


def node_circle(x, y, label, r=16, fill="#eef3f7", stroke="#2C6E9B", text_fill="#000"):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x}" y="{y+4}" font-size="11" text-anchor="middle" fill="{text_fill}" '
            f'font-family="Georgia, serif">{label}</text>')


def edge_line(x1, y1, x2, y2, color="#888", width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>'


def bottleneck_diagram(path, bw=False):
    W, H = 760, 340
    pos = {
        "baseA": (80, 90), "a1": (190, 90), "a2": (300, 90),
        "baseB": (80, 250), "b1": (190, 250), "b2": (300, 250),
        "P": (400, 170),
        "Q": (520, 170),
        "v1": (630, 90), "v2": (630, 170), "v3": (700, 250),
    }
    if bw:
        # differentiated by weight/dash only: thin solid = known corridor,
        # thick solid = the shared bottleneck, dashed = still unexplored.
        edges = [
            ("baseA", "a1", "#000", 1.6, None), ("a1", "a2", "#000", 1.6, None),
            ("a2", "P", "#000", 1.6, None),
            ("baseB", "b1", "#000", 1.6, None), ("b1", "b2", "#000", 1.6, None),
            ("b2", "P", "#000", 1.6, None),
            ("P", "Q", "#000", 5.5, None),
            ("Q", "v1", "#000", 1.6, "6,4"), ("Q", "v2", "#000", 1.6, "6,4"),
            ("v2", "v3", "#000", 1.6, "6,4"),
        ]
    else:
        edges = [
            ("baseA", "a1", "#888", 2, None), ("a1", "a2", "#888", 2, None),
            ("a2", "P", "#888", 2, None),
            ("baseB", "b1", "#888", 2, None), ("b1", "b2", "#888", 2, None),
            ("b2", "P", "#888", 2, None),
            ("P", "Q", "#C1553B", 5, None),
            ("Q", "v1", "#3F9E6B", 2, "6,4"), ("Q", "v2", "#3F9E6B", 2, "6,4"),
            ("v2", "v3", "#3F9E6B", 2, "6,4"),
        ]
    node_fill = "#fff" if bw else "#eef3f7"
    node_stroke = "#000" if bw else "#2C6E9B"

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Georgia, serif">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for u, v, color, width, dash in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        parts.append(edge_line(x1, y1, x2, y2, color, width, dash))
    for n, (x, y) in pos.items():
        parts.append(node_circle(x, y, n, fill=node_fill, stroke=node_stroke))
    if not bw:
        parts.append('<text x="20" y="24" font-size="16" font-weight="bold">'
                      'Bottleneck facility: two wings sharing one bridge to the valley</text>')
    bridge_c, valley_c = ("#000", "#000") if bw else ("#C1553B", "#3F9E6B")
    parts.append(f'<line x1="20" y1="300" x2="50" y2="300" stroke="{bridge_c}" stroke-width="5"/>'
                  f'<text x="58" y="304" font-size="12">shared bridge edge (both wings\' only route to the valley)</text>')
    parts.append(f'<line x1="20" y1="320" x2="50" y2="320" stroke="{valley_c}" stroke-width="2" stroke-dasharray="6,4"/>'
                  f'<text x="58" y="324" font-size="12">unexplored valley corridors</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def sample_facility_diagram(path, rows=4, cols=4, seed=20260906, p_haz=0.15, bw=False):
    rng = random.Random(seed)
    g = environment.generate_random_facility(rng, rows=rows, cols=cols, prune_prob=0.35, min_dead_ends=2)
    truth = environment.assign_hazards(g, rng, p_haz)

    cell = 110
    margin = 50
    W = cols * cell + 2 * margin
    H = rows * cell + 2 * margin

    def coord(n):
        r, c = divmod(n, cols)
        return margin + c * cell + cell // 2, margin + r * cell + cell // 2

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Georgia, serif">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    dead_ends = set(g.dead_ends())
    for eid, (u, v) in g.edges.items():
        x1, y1 = coord(u); x2, y2 = coord(v)
        if bw:
            color = "#000"
            width = 4.5 if truth[eid] == 1 else 1.3
        else:
            color = "#C1553B" if truth[eid] == 1 else "#bbbbbb"
            width = 4 if truth[eid] == 1 else 2.5
        parts.append(edge_line(x1, y1, x2, y2, color, width))
    for n in g.nodes:
        x, y = coord(n)
        if n == g.base:
            fill, stroke, tf = ("#000", "#000", "#fff") if bw else ("#2C6E9B", "#123", "#000")
            parts.append(node_circle(x, y, "B", fill=fill, stroke=stroke, r=15, text_fill=tf))
        elif n in dead_ends:
            fill, stroke = ("#fff", "#000") if bw else ("#fff3e0", "#B08B2D")
            parts.append(node_circle(x, y, "", fill=fill, stroke=stroke, r=8))
        else:
            fill, stroke = ("#fff", "#888") if bw else ("#eef3f7", "#999")
            parts.append(node_circle(x, y, "", fill=fill, stroke=stroke, r=6))
    if not bw:
        parts.append('<text x="20" y="24" font-size="16" font-weight="bold">Sample random facility '
                      f'(rows={rows}, cols={cols}) -- dead ends force round-trip revisits</text>')
    base_fill, base_stroke = ("#000", "#000") if bw else ("#2C6E9B", "#123")
    de_fill, de_stroke = ("#fff", "#000") if bw else ("#fff3e0", "#B08B2D")
    hz_color = "#000" if bw else "#C1553B"
    parts.append(f'<circle cx="30" cy="{H-40}" r="7" fill="{base_fill}" stroke="{base_stroke}"/>'
                  f'<text x="45" y="{H-36}" font-size="12">base station</text>')
    parts.append(f'<circle cx="30" cy="{H-20}" r="7" fill="{de_fill}" stroke="{de_stroke}"/>'
                  f'<text x="45" y="{H-16}" font-size="12">dead-end room</text>')
    parts.append(f'<line x1="180" y1="{H-40}" x2="215" y2="{H-40}" stroke="{hz_color}" '
                  f'stroke-width="{4.5 if bw else 4}"/>'
                  f'<text x="222" y="{H-36}" font-size="12">true hazard on corridor'
                  f'{" (thick line)" if bw else ""}</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    bottleneck_diagram("results/figures/bottleneck_topology.svg")
    sample_facility_diagram("results/figures/sample_facility.svg")
    import os
    os.makedirs("results/figures/print", exist_ok=True)
    bottleneck_diagram("results/figures/print/bottleneck_topology_bw.svg", bw=True)
    sample_facility_diagram("results/figures/print/sample_facility_bw.svg", bw=True)
    print("diagrams written")
