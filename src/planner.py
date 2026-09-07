"""
planner.py
==========
Greedy expected-information-gain path planner over a graph environment.

The reference paper's own problem formulation (Problem 1) requires agents to
"start and end their search rounds at their designated base stations" --
i.e. every mission is a round trip. On a graph with dead-end branches, a
round trip into a dead end has no choice but to re-traverse the connecting
corridor on the way back. We enumerate two kinds of round-trip candidates:

  * "there and back"  -- go out along a simple path, return by the same
    edges (forces a revisit of every edge on the path).
  * "loop" -- go out along one simple path and back along a different,
    edge-disjoint simple path to the same turnaround node (no revisit),
    when the graph's connectivity happens to offer one.

Candidates are scored by exact expected information gain (Eq. (1) of the
reference paper), always computed with the *correct* multi-visit Bayesian
model (inference.bn_exact_update) regardless of which update rule is later
used to actually record the observation -- this keeps planning/physical
behaviour identical across compared inference methods, isolating the
comparison to the belief-update step alone.
"""
from __future__ import annotations

from itertools import combinations

from . import inference
from . import metrics


def enumerate_outbound_paths(graph, start, max_len):
    """All simple paths (no repeated nodes) from `start`, length 1..max_len,
    returned as a dict end_node -> list of edge-id-sequences."""
    by_end = {}

    def dfs(node, visited_nodes, path_edges):
        if path_edges:
            by_end.setdefault(node, []).append(list(path_edges))
        if len(path_edges) >= max_len:
            return
        for nbr, eid in graph.adj[node]:
            if nbr not in visited_nodes:
                visited_nodes.add(nbr)
                path_edges.append(eid)
                dfs(nbr, visited_nodes, path_edges)
                path_edges.pop()
                visited_nodes.remove(nbr)

    dfs(start, {start}, [])
    return by_end


def build_roundtrip_candidates(graph, start, max_out_len=4, max_candidates=250):
    by_end = enumerate_outbound_paths(graph, start, max_out_len)
    candidates = []

    for end_node, paths in by_end.items():
        for p in paths:
            candidates.append(p + list(reversed(p)))  # there-and-back

        if end_node != start and len(paths) > 1:
            for p1, p2 in combinations(paths, 2):
                if set(p1).isdisjoint(p2):
                    candidates.append(p1 + list(reversed(p2)))
                    candidates.append(p2 + list(reversed(p1)))

        if len(candidates) >= max_candidates:
            break

    # de-duplicate identical edge sequences
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:max_candidates]


def expected_info_gain(belief: dict, path: list, lethality: float, false_positive: float) -> float:
    prior_H = sum(metrics.binary_entropy(belief[e]) for e in set(path))
    S = inference.path_survival_probability(belief, path, lethality, false_positive)

    survive_belief = inference.bn_exact_update(belief, path, destroyed=False,
                                                lethality=lethality, false_positive=false_positive)
    destroy_belief = inference.bn_exact_update(belief, path, destroyed=True,
                                                lethality=lethality, false_positive=false_positive)
    H_survive = sum(metrics.binary_entropy(survive_belief[e]) for e in set(path))
    H_destroy = sum(metrics.binary_entropy(destroy_belief[e]) for e in set(path))

    expected_post_H = S * H_survive + (1 - S) * H_destroy
    return prior_H - expected_post_H


def plan_best_path(graph, belief: dict, start, lethality: float, false_positive: float,
                    max_out_len=4, max_candidates=250):
    candidates = build_roundtrip_candidates(graph, start, max_out_len, max_candidates)
    if not candidates:
        return []
    best_path, best_gain = None, -1.0
    for c in candidates:
        gain = expected_info_gain(belief, c, lethality, false_positive)
        if gain > best_gain:
            best_gain = gain
            best_path = c
    return best_path
