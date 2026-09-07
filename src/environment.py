"""
environment.py
================
Graph-shaped deployment environments for path-based sensor simulation.

Unlike the open 15x15 continuous grid used in the reference paper (Srivastava,
Kontoudis, Sofge & Otte, "Bayesian Networks for Path-Based Sensors"), the
environments here are built as *graphs*: rooms/junctions as nodes, corridors
as edges. This is meant to model realistic infrastructure -- collapsed
buildings, tunnels, mine drifts, road networks -- where:

  1. Dead-end branches are common, so a round-trip mission (start and end at
     the same base station, as the paper's own Problem 1 formulation
     requires) is frequently forced to re-traverse the same edge twice.
  2. Some corridors are cut-edges / bridges: the *only* connection between
     two regions. No partition of the graph can put a cut-edge on only one
     side without also cutting one side off from it entirely -- unlike in
     continuous 2D space, where an entropy-weighted Voronoi partition can
     always be drawn without forcing two agents to share a cell.

Both properties are used later to probe gaps in the reference method that a
purely open, cycle-rich grid world would never expose.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Graph:
    """A small undirected multigraph-free environment graph."""
    nodes: list
    edges: dict            # edge_id -> (u, v)
    adj: dict = field(default_factory=dict)   # node -> list[(neighbor, edge_id)]
    base: object = None    # default base station node
    name: str = "graph"

    def __post_init__(self):
        if not self.adj:
            self.adj = {n: [] for n in self.nodes}
            for eid, (u, v) in self.edges.items():
                self.adj[u].append((v, eid))
                self.adj[v].append((u, eid))

    def is_connected(self) -> bool:
        if not self.nodes:
            return True
        start = self.nodes[0]
        seen = {start}
        q = deque([start])
        while q:
            n = q.popleft()
            for nbr, _ in self.adj[n]:
                if nbr not in seen:
                    seen.add(nbr)
                    q.append(nbr)
        return len(seen) == len(self.nodes)

    def degree(self, node) -> int:
        return len(self.adj[node])

    def dead_ends(self) -> list:
        return [n for n in self.nodes if self.degree(n) == 1]

    def hop_distance(self, src) -> dict:
        """BFS hop distances (in edges) from src to every reachable node."""
        dist = {src: 0}
        q = deque([src])
        while q:
            n = q.popleft()
            for nbr, _ in self.adj[n]:
                if nbr not in dist:
                    dist[nbr] = dist[n] + 1
                    q.append(nbr)
        return dist


def generate_random_facility(rng: random.Random, rows: int = 4, cols: int = 4,
                              prune_prob: float = 0.35, min_dead_ends: int = 2,
                              max_tries: int = 300) -> Graph:
    """Random pruned-grid 'building': start from a full grid of rooms, then
    randomly delete corridors while preserving connectivity, until at least
    `min_dead_ends` single-corridor rooms exist (a realistic irregular floor
    plan almost always has a few of these; an open 15x15 grid, by contrast,
    never does).
    """
    base_nodes = [r * cols + c for r in range(rows) for c in range(cols)]

    for _try in range(max_tries):
        edges = {}
        eid = 0
        for r in range(rows):
            for c in range(cols):
                n = r * cols + c
                if c + 1 < cols:
                    edges[eid] = (n, n + 1); eid += 1
                if r + 1 < rows:
                    edges[eid] = (n, n + cols); eid += 1

        # candidate removal order, shuffled
        removable = list(edges.keys())
        rng.shuffle(removable)

        kept = dict(edges)
        for e in removable:
            if rng.random() < prune_prob:
                trial = dict(kept)
                del trial[e]
                g = Graph(nodes=list(base_nodes), edges=trial)
                if g.is_connected():
                    kept = trial

        g = Graph(nodes=list(base_nodes), edges=kept, name="random_facility")
        if g.is_connected() and len(g.dead_ends()) >= min_dead_ends:
            g.base = 0
            return g

    # fall back: return whatever the last attempt produced (still connected)
    g = Graph(nodes=list(base_nodes), edges=kept, name="random_facility")
    g.base = 0
    return g


def generate_bottleneck_facility() -> Graph:
    """A fixed, hand-designed environment for the multi-agent overlap study.

    Two wings (A and B), each with its own base station, both open onto a
    shared junction node P. The only way from P to the unexplored, hazard-
    bearing 'valley' region is a single bridge edge (P, Q) -- a true graph
    cut-edge. Any agent from either wing that wants to search the valley
    (by far the highest-entropy, highest-value region every round) MUST
    cross that same physical edge. No partitioning scheme can avoid this:
    it is a topological bottleneck, not a planning inefficiency.

    Nodes:
        baseA, a1, a2         -- wing A corridor into the shared junction
        baseB, b1, b2         -- wing B corridor into the shared junction
        P                     -- shared junction (reachable from both wings)
        Q                     -- far side of the bridge
        v1, v2, v3            -- valley rooms (the actual search target)
    """
    nodes = ["baseA", "a1", "a2", "baseB", "b1", "b2", "P", "Q", "v1", "v2", "v3"]
    edges = {
        0: ("baseA", "a1"),
        1: ("a1", "a2"),
        2: ("a2", "P"),
        3: ("baseB", "b1"),
        4: ("b1", "b2"),
        5: ("b2", "P"),
        6: ("P", "Q"),        # <-- the bottleneck bridge
        7: ("Q", "v1"),
        8: ("Q", "v2"),
        9: ("v2", "v3"),
    }
    g = Graph(nodes=nodes, edges=edges, name="bottleneck_facility")
    g.bridge_edge = 6
    g.base_a = "baseA"
    g.base_b = "baseB"
    return g


def assign_hazards(graph: Graph, rng: random.Random, p_haz: float,
                    force: dict | None = None) -> dict:
    """Independent Bernoulli(p_haz) ground-truth hazard per edge (matches the
    reference paper's cell-independence assumption -- see Section 3). `force`
    can pin specific edge ids to a known ground-truth value (used to build
    controlled bottleneck-edge scenarios).
    """
    truth = {}
    for eid in graph.edges:
        truth[eid] = 1 if rng.random() < p_haz else 0
    if force:
        truth.update(force)
    return truth
