"""
make_animation.py
==================
Animated SVG "replays" of a single simulated round, for readers who want to
watch an agent actually move through the environment rather than only read
convergence curves.

Why SVG animation instead of a .gif: this project intentionally has zero
third-party dependencies (no numpy/matplotlib/Pillow -- see README), and
this Python distribution has no working `pip` to install one. A GIF
encoder would have to be hand-written from scratch (palette quantization +
LZW) for no real benefit: SMIL-animated SVG plays natively in any browser
and in GitHub's own rendered markdown (an `![]()` reference to an .svg file
is served as a normal <img>, and animated SVG plays inside an <img> tag in
every major engine), so it gives the same "watch it move" result without
adding a binary asset format and a hand-rolled codec. The static topology
diagrams (make_diagrams.py) remain as a non-animated fallback.
"""
import random

from src import environment, planner

AGENT_BLUE = "#2C6E9B"
AGENT_ORANGE = "#C1553B"


def node_circle(x, y, label, r=16, fill="#eef3f7", stroke="#2C6E9B"):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x}" y="{y+4}" font-size="11" text-anchor="middle" font-family="Georgia, serif">{label}</text>')


def edge_line(x1, y1, x2, y2, color="#ccc", width=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def path_edges_to_node_seq(graph, start, edge_path):
    """Turn a list of edge ids into the sequence of nodes actually visited."""
    seq = [start]
    cur = start
    for eid in edge_path:
        u, v = graph.edges[eid]
        nxt = v if u == cur else u
        seq.append(nxt)
        cur = nxt
    return seq


def path_string(seq, pos):
    pts = [f"{pos[n][0]},{pos[n][1]}" for n in seq]
    return "M " + " L ".join(pts)


def moving_dot(seq, pos, color, dur, label, r=9):
    d = path_string(seq, pos)
    return (
        f'<circle r="{r}" fill="{color}" stroke="#fff" stroke-width="2">'
        f'<animateMotion dur="{dur}s" repeatCount="indefinite" path="{d}"/>'
        f'</circle>'
        f'<circle r="{r+4}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.5">'
        f'<animateMotion dur="{dur}s" repeatCount="indefinite" path="{d}"/>'
        f'</circle>'
    )


def traced_route(seq, pos, color, dur, width=4):
    """A route line that visibly 'draws itself' in sync with the moving dot,
    using the pathLength trick so it works regardless of true geometric
    length (needed because a there-and-back route retraces itself)."""
    d = path_string(seq, pos)
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" opacity="0.55" pathLength="100" '
        f'stroke-dasharray="100" stroke-dashoffset="100">'
        f'<animate attributeName="stroke-dashoffset" from="100" to="0" '
        f'dur="{dur}s" repeatCount="indefinite"/>'
        f'</path>'
    )


def single_agent_animation(path, seed=20260906, p_haz=0.15, dur=7):
    rng = random.Random(seed)
    g = environment.generate_random_facility(rng, rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)
    truth = environment.assign_hazards(g, rng, p_haz)

    cell, margin = 110, 50
    cols = 4
    W = cols * cell + 2 * margin
    H = 4 * cell + 2 * margin

    def coord(n):
        r, c = divmod(n, cols)
        return margin + c * cell + cell // 2, margin + r * cell + cell // 2

    pos = {n: coord(n) for n in g.nodes}
    belief = {e: p_haz for e in g.edges}
    edge_path = planner.plan_best_path(g, belief, g.base, 0.85, 0.04, max_out_len=4, max_candidates=250)
    node_seq = path_edges_to_node_seq(g, g.base, edge_path)
    revisited = len(edge_path) != len(set(edge_path))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H+40}" '
             f'viewBox="0 0 {W} {H+40}" font-family="Georgia, serif">',
             f'<rect width="{W}" height="{H+40}" fill="#ffffff"/>']
    for eid, (u, v) in g.edges.items():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        color = "#C1553B" if truth[eid] == 1 else "#ddd"
        parts.append(edge_line(x1, y1, x2, y2, color, 4 if truth[eid] == 1 else 2.5))

    parts.append(traced_route(node_seq, pos, AGENT_BLUE, dur))

    for n in g.nodes:
        x, y = pos[n]
        if n == g.base:
            parts.append(node_circle(x, y, "B", fill=AGENT_BLUE, stroke="#123"))
        elif n in g.dead_ends():
            parts.append(node_circle(x, y, "", fill="#fff3e0", stroke="#B08B2D", r=8))
        else:
            parts.append(node_circle(x, y, "", fill="#eef3f7", stroke="#999", r=6))

    parts.append(moving_dot(node_seq, pos, AGENT_BLUE, dur, "agent"))

    caption = ("planned round trip revisits a corridor (dead-end detour)" if revisited
               else "planned round trip -- no revisit needed this round")
    parts.append(f'<text x="20" y="24" font-size="16" font-weight="bold">'
                  f'One simulated round trip on a random facility ({caption})</text>')
    parts.append(f'<text x="20" y="{H+30}" font-size="12" fill="#555">'
                  f'Blue dot = agent following its planned round trip from base station B and back. '
                  f'Red corridors carry a planted hazard (not visible to the agent in advance).</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return revisited


def bottleneck_animation(path, dur=8):
    g = environment.generate_bottleneck_facility()
    pos = {
        "baseA": (80, 90), "a1": (190, 90), "a2": (300, 90),
        "baseB": (80, 250), "b1": (190, 250), "b2": (300, 250),
        "P": (400, 170), "Q": (520, 170),
        "v1": (630, 90), "v2": (630, 170), "v3": (700, 250),
    }
    W, H = 760, 380
    # Illustrates a LATER round, after each wing has already searched its
    # own three corridors down to near-zero uncertainty -- exactly the
    # point (confirmed in the round-by-round Experiment 2 data) at which
    # the still-unexplored valley beyond the shared bridge becomes the
    # clearly better target for both wings at once.
    wing_edges = {0, 1, 2, 3, 4, 5}
    belief = {e: (0.02 if e in wing_edges else 0.20) for e in g.edges}
    edge_path_a = planner.plan_best_path(g, belief, g.base_a, 0.85, 0.04, max_out_len=5, max_candidates=250)
    edge_path_b = planner.plan_best_path(g, belief, g.base_b, 0.85, 0.04, max_out_len=5, max_candidates=250)
    seq_a = path_edges_to_node_seq(g, g.base_a, edge_path_a)
    seq_b = path_edges_to_node_seq(g, g.base_b, edge_path_b)
    both_used_bridge = (g.bridge_edge in edge_path_a) and (g.bridge_edge in edge_path_b)

    static_edges = [
        ("baseA", "a1"), ("a1", "a2"), ("a2", "P"),
        ("baseB", "b1"), ("b1", "b2"), ("b2", "P"),
        ("Q", "v1"), ("Q", "v2"), ("v2", "v3"),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="Georgia, serif">', f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for u, v in static_edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        parts.append(edge_line(x1, y1, x2, y2, "#ddd", 2.5))
    bx1, by1 = pos["P"]; bx2, by2 = pos["Q"]
    parts.append(edge_line(bx1, by1, bx2, by2, "#e7b8ac", 6))

    parts.append(traced_route(seq_a, pos, AGENT_BLUE, dur))
    parts.append(traced_route(seq_b, pos, AGENT_ORANGE, dur))

    for n, (x, y) in pos.items():
        if n == "baseA":
            parts.append(node_circle(x, y, "A", fill=AGENT_BLUE, stroke="#123"))
        elif n == "baseB":
            parts.append(node_circle(x, y, "B", fill=AGENT_ORANGE, stroke="#321"))
        elif n in ("P", "Q"):
            parts.append(node_circle(x, y, n, fill="#fff", stroke="#C1553B", r=14))
        else:
            parts.append(node_circle(x, y, "", fill="#eef3f7", stroke="#999", r=6))

    parts.append(moving_dot(seq_a, pos, AGENT_BLUE, dur, "agent A"))
    parts.append(moving_dot(seq_b, pos, AGENT_ORANGE, dur, "agent B"))

    parts.append('<text x="20" y="24" font-size="16" font-weight="bold">'
                 'Later round, both wings already searched: both agents converge on the shared bridge</text>')
    note = ("both agents' planned routes use the shared bridge edge this round"
            if both_used_bridge else
            "in this particular round only one agent used the shared bridge")
    parts.append(f'<text x="20" y="{H-16}" font-size="12" fill="#555">'
                  f'Blue = agent A (wing A), orange = agent B (wing B); {note}. '
                  f'The thick pink edge (P-Q) is the sole route to the valley for both wings.</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return both_used_bridge


if __name__ == "__main__":
    revisited = single_agent_animation("results/figures/sim_single_agent.svg")
    overlap = bottleneck_animation("results/figures/sim_bottleneck.svg")
    print("single-agent example revisits a corridor:", revisited)
    print("bottleneck example: both agents share the bridge this round:", overlap)
