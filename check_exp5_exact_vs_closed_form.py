"""
check_exp5_exact_vs_closed_form.py
==================================
Sanity check for Experiment 5's exact-joint inference.

Starting from an independent (product) prior, the exact-joint posterior after
ONE update must have exactly the same per-corridor marginals as the
closed-form factored rule (inference.bn_exact_update), because the closed form
is exact for a product prior.  Later updates differ only because the factored
rule throws away the correlations created by that first update.

This holds for both visit models:
  * independent    -- every pass is a separate survival trial
                      (ExactJoint.update(..., "independent") vs bn_exact_update on the path)
  * once-per-round -- called "persistent" in experiment5_exact_joint.py: each
                      hazardous corridor gets one destruction draw per round,
                      at its first pass; later passes in the same round add no
                      risk (ExactJoint.update(..., "persistent") vs
                      bn_exact_update on the de-duplicated path)

Cases: random small facility graphs (2x2 to 3x4 grids, so 3-17 corridors),
random heterogeneous priors, random walks from the base that revisit
corridors, and both outcomes.  Exits non-zero if any marginal differs by more
than TOL.

Run from the repository root:
    python check_exp5_exact_vs_closed_form.py
"""
import random
import sys

import numpy as np

import experiment5_exact_joint as e5
from src import environment

N_CASES = 4000          # split evenly between the two visit models
TOL = 1e-12
SEED = 5112026


def random_walk(graph, rng, length):
    node, path = graph.base, []
    for _ in range(length):
        nbr, eid = rng.choice(graph.adj[node])
        path.append(eid)
        node = nbr
    return path


def product_prior_joint(edges, prior):
    """ExactJoint whose table is the product of per-edge (heterogeneous) priors."""
    ej = e5.ExactJoint(edges, 0.5)
    p = np.ones((2,) * ej.E)
    for e, i in ej.idx.items():
        shape = [1] * ej.E
        shape[i] = 2
        p = p * np.array([1 - prior[e], prior[e]]).reshape(shape)
    ej.p = p / p.sum()
    return ej


def run(n_cases=N_CASES, seed=SEED):
    rng = random.Random(seed)
    worst = {"independent": 0.0, "persistent": 0.0}
    n = {"independent": 0, "persistent": 0}
    max_visits_seen = 0
    for case in range(n_cases):
        model = "independent" if case % 2 == 0 else "persistent"
        rows, cols = rng.choice([(2, 2), (2, 3), (3, 3), (3, 4)])
        graph = environment.generate_random_facility(rng, rows=rows, cols=cols,
                                                     prune_prob=0.3, min_dead_ends=1)
        edges = list(graph.edges)
        prior = {e: rng.uniform(0.01, 0.99) for e in edges}
        path = random_walk(graph, rng, rng.randint(1, 12))
        destroyed = rng.random() < 0.5

        ej = product_prior_joint(edges, prior)
        ej.update(path, destroyed, model)
        exact = ej.marginals()
        closed = e5.factored_update(prior, path, destroyed, model)

        gap = max(abs(exact[e] - closed[e]) for e in edges)
        worst[model] = max(worst[model], gap)
        n[model] += 1
        max_visits_seen = max(max_visits_seen, max(path.count(e) for e in set(path)))
    return worst, n, max_visits_seen


def main():
    worst, n, max_visits = run()
    ok = all(v <= TOL for v in worst.values())
    for model in ("independent", "persistent"):
        label = "once-per-round" if model == "persistent" else model
        print(f"{label:15s} cases={n[model]:5d}  max |exact - closed form| = {worst[model]:.3e}")
    print(f"max passes over one corridor in a test path: {max_visits}")
    print("PASS" if ok else f"FAIL (tolerance {TOL:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
