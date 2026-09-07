"""
make_diagrams.py
=================
Small hand-laid-out SVG diagrams of the two environments, for the report.
"""
import random
from src import environment


def node_circle(x, y, label, r=16, fill="#eef3f7", stroke="#2C6E9B"):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x}" y="{y+4}" font-size="11" text-anchor="middle" font-family="Georgia, serif">{label}</text>')


def edge_line(x1, y1, x2, y2, color="#888", width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>'


def bottleneck_diagram(path):
    W, H = 760, 340
    pos = {
        "baseA": (80, 90), "a1": (190, 90), "a2": (300, 90),
        "baseB": (80, 250), "b1": (190, 250), "b2": (300, 250),
        "P": (400, 170),
        "Q": (520, 170),
        "v1": (630, 90), "v2": (630, 170), "v3": (700, 250),
    }
    edges = [
        ("baseA", "a1", "#888", 2, None, "known corridor"),
        ("a1", "a2", "#888", 2, None, "known corridor"),
        ("a2", "P", "#888", 2, None, "known corridor"),
        ("baseB", "b1", "#888", 2, None, "known corridor"),
        ("b1", "b2", "#888", 2, None, "known corridor"),
        ("b2", "P", "#888", 2, None, "known corridor"),
        ("P", "Q", "#C1553B", 5, None, "SHARED BRIDGE (cut edge)"),
        ("Q", "v1", "#3F9E6B", 2, "6,4", "unexplored (valley)"),
        ("Q", "v2", "#3F9E6B", 2, "6,4", "unexplored (valley)"),
        ("v2", "v3", "#3F9E6B", 2, "6,4", "unexplored (valley)"),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Georgia, serif">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for u, v, color, width, dash, _ in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        parts.append(edge_line(x1, y1, x2, y2, color, width, dash))
    for n, (x, y) in pos.items():
        parts.append(node_circle(x, y, n))
    parts.append('<text x="20" y="24" font-size="16" font-weight="bold">Bottleneck facility: two wings sharing one bridge to the valley</text>')
    parts.append(f'<line x1="20" y1="300" x2="50" y2="300" stroke="#C1553B" stroke-width="5"/>'
                  f'<text x="58" y="304" font-size="12">shared bridge edge (both wings\' only route to the valley)</text>')
    parts.append(f'<line x1="20" y1="320" x2="50" y2="320" stroke="#3F9E6B" stroke-width="2" stroke-dasharray="6,4"/>'
                  f'<text x="58" y="324" font-size="12">unexplored valley corridors</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def sample_facility_diagram(path, rows=4, cols=4, seed=20260906, p_haz=0.15):
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
        color = "#C1553B" if truth[eid] == 1 else "#bbbbbb"
        width = 4 if truth[eid] == 1 else 2.5
        parts.append(edge_line(x1, y1, x2, y2, color, width))
    for n in g.nodes:
        x, y = coord(n)
        if n == g.base:
            parts.append(node_circle(x, y, "B", fill="#2C6E9B", stroke="#123", r=15))
        elif n in dead_ends:
            parts.append(node_circle(x, y, "", fill="#fff3e0", stroke="#B08B2D", r=8))
        else:
            parts.append(node_circle(x, y, "", fill="#eef3f7", stroke="#999", r=6))
    parts.append(f'<text x="20" y="24" font-size="16" font-weight="bold">Sample random facility '
                  f'(rows={rows}, cols={cols}) -- dead ends force round-trip revisits</text>')
    parts.append(f'<circle cx="30" cy="{H-40}" r="7" fill="#2C6E9B" stroke="#123"/>'
                  f'<text x="45" y="{H-36}" font-size="12">base station</text>')
    parts.append(f'<circle cx="30" cy="{H-20}" r="7" fill="#fff3e0" stroke="#B08B2D"/>'
                  f'<text x="45" y="{H-16}" font-size="12">dead-end room</text>')
    parts.append(f'<line x1="180" y1="{H-40}" x2="215" y2="{H-40}" stroke="#C1553B" stroke-width="4"/>'
                  f'<text x="222" y="{H-36}" font-size="12">true hazard on corridor</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    bottleneck_diagram("results/figures/bottleneck_topology.svg")
    sample_facility_diagram("results/figures/sample_facility.svg")
    print("diagrams written")
