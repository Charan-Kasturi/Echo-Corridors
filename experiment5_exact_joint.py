"""
experiment5_exact_joint.py
==========================
Two questions left open by Experiments 1-2 of Echo Corridors.

Q1 (factored vs exact).  Both the reference paper and Echo Corridors keep one
    independent probability per corridor.  After a "destroyed" observation the
    corridors on that path become correlated, and storing marginals discards
    that correlation.  How much does this cost in calibration, in hazard
    ranking, and in the planner's choice of path?

Q2 (visit model, 2x2).  The corrected closed form assumes every pass over a
    corridor is an independent survival trial.  If instead a hazard acts once
    per round (a "persistent" outcome: surviving the first pass means surviving
    later passes in the same round), repeated passes carry no new evidence and
    the independent-visit rule should over-count.  We cross two simulators
    {independent, persistent} with update rules assuming each model, in both
    factored and exact-joint form.

Design follows Experiment 1: the same random facility generator, seeds,
hazard prior, lethality, false-positive rate, rounds and trials.  Every
update rule sees the identical sequence of (path, outcome) pairs; paths are
always chosen by one canonical belief (the factored independent-visit rule),
so differences reflect inference only.

The exact joint posterior is held as a full table over all 2^E hazard
configurations (E <= 19 here, so at most 524,288 entries) and updated with
the exact path likelihood.  Marginals are read off by summation.

Run from the Echo-Corridors repository root:
    python /path/to/experiment5_exact_joint.py
"""
import json
import os
import random
import statistics as stats
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src import environment, inference, planner, metrics  # noqa: E402

P_HAZ = 0.15
LETHALITY = 0.85
FALSE_POSITIVE = 0.04
ROUNDS = 45
TRIALS = 40
MAX_OUT_LEN = 4
SEED0 = 20260906          # same seed family as experiment1_revisits.py

A = 1.0 - LETHALITY       # P(survive one pass | hazard)
B = 1.0 - FALSE_POSITIVE  # P(survive one pass | no hazard)

SIMULATORS = ["independent", "persistent"]
RULES = ["factored_indep", "factored_persist", "joint_indep", "joint_persist"]


def visit_counts(path, model):
    counts = {}
    for e in path:
        counts[e] = counts.get(e, 0) + 1
    if model == "persistent":
        counts = {e: 1 for e in counts}
    return counts


def simulate_outcome(path, truth, rng, model):
    """True if the robot is destroyed somewhere on the path."""
    seen = set()
    for e in path:
        if model == "persistent" and e in seen:
            continue          # outcome for this corridor already settled this round
        seen.add(e)
        p_destroy = LETHALITY if truth[e] == 1 else FALSE_POSITIVE
        if rng.random() < p_destroy:
            return True
    return False


class ExactJoint:
    """Full posterior over all 2^E hazard configurations."""

    def __init__(self, edges, prior):
        self.edges = list(edges)
        self.idx = {e: i for i, e in enumerate(self.edges)}
        self.E = len(self.edges)
        p = np.zeros((2,) * self.E)
        p[...] = 1.0
        for i in range(self.E):
            shape = [1] * self.E
            shape[i] = 2
            p = p * np.array([1 - prior, prior]).reshape(shape)
        self.p = p / p.sum()

    def update(self, path, destroyed, model):
        counts = visit_counts(path, model)
        surv = np.ones((1,) * self.E)
        for e, n in counts.items():
            shape = [1] * self.E
            shape[self.idx[e]] = 2
            surv = surv * np.array([B ** n, A ** n]).reshape(shape)
        lik = (1.0 - surv) if destroyed else surv
        p = self.p * lik
        z = p.sum()
        if z > 0:
            self.p = p / z

    def marginals(self):
        out = {}
        for e, i in self.idx.items():
            axes = tuple(j for j in range(self.E) if j != i)
            m = self.p.sum(axis=axes)
            out[e] = float(m[1])
        return out


def factored_update(belief, path, destroyed, model):
    if model == "persistent":
        path = list(dict.fromkeys(path))   # distinct corridors, order kept
    return inference.bn_exact_update(belief, path, destroyed, LETHALITY, FALSE_POSITIVE)


def score(belief, truth, k):
    _, rec = metrics.precision_recall_at_k(belief, truth, k)
    return {"brier": metrics.brier_score(belief, truth),
            "logloss": metrics.log_loss(belief, truth),
            "recall_k": rec,
            "entropy": metrics.map_entropy(belief)}


def run_trial(seed, sim_model):
    rng = random.Random(seed)
    graph = environment.generate_random_facility(rng, rows=4, cols=4,
                                                 prune_prob=0.35, min_dead_ends=2)
    truth = environment.assign_hazards(graph, rng, P_HAZ)
    k = sum(truth.values())

    fac = {"factored_indep": {e: P_HAZ for e in graph.edges},
           "factored_persist": {e: P_HAZ for e in graph.edges}}
    joint = {"joint_indep": ExactJoint(graph.edges, P_HAZ),
             "joint_persist": ExactJoint(graph.edges, P_HAZ)}

    final = {}
    decision_rounds = 0
    decision_diff = 0
    max_marg_gap = []   # per round: max |factored - exact| under the matching model

    match_fac = "factored_indep" if sim_model == "independent" else "factored_persist"
    match_joint = "joint_indep" if sim_model == "independent" else "joint_persist"

    for _ in range(ROUNDS):
        plan_belief = fac["factored_indep"]
        path = planner.plan_best_path(graph, plan_belief, graph.base, LETHALITY,
                                      FALSE_POSITIVE, max_out_len=MAX_OUT_LEN,
                                      max_candidates=250)
        if not path:
            break

        # Decision check: would the planner pick a different route if handed the
        # exact-joint marginals (matching model) instead of the factored ones?
        jm = joint[match_joint].marginals()
        path_exact = planner.plan_best_path(graph, jm, graph.base, LETHALITY,
                                            FALSE_POSITIVE, max_out_len=MAX_OUT_LEN,
                                            max_candidates=250)
        path_fac = planner.plan_best_path(graph, fac[match_fac], graph.base, LETHALITY,
                                          FALSE_POSITIVE, max_out_len=MAX_OUT_LEN,
                                          max_candidates=250)
        decision_rounds += 1
        if path_exact != path_fac:
            decision_diff += 1
        max_marg_gap.append(max(abs(jm[e] - fac[match_fac][e]) for e in graph.edges))

        destroyed = simulate_outcome(path, truth, rng, sim_model)

        fac["factored_indep"] = factored_update(fac["factored_indep"], path, destroyed, "independent")
        fac["factored_persist"] = factored_update(fac["factored_persist"], path, destroyed, "persistent")
        joint["joint_indep"].update(path, destroyed, "independent")
        joint["joint_persist"].update(path, destroyed, "persistent")

    for r in ("factored_indep", "factored_persist"):
        final[r] = score(fac[r], truth, k)
    for r in ("joint_indep", "joint_persist"):
        final[r] = score(joint[r].marginals(), truth, k)

    return {"final": final,
            "decision_diff_frac": decision_diff / max(decision_rounds, 1),
            "mean_max_marginal_gap": stats.mean(max_marg_gap) if max_marg_gap else 0.0,
            "n_edges": len(graph.edges), "n_hazards": k}


def summarise(trials):
    out = {}
    for r in RULES:
        out[r] = {}
        for m in ("brier", "logloss", "recall_k", "entropy"):
            xs = [t["final"][r][m] for t in trials]
            out[r][m] = {"mean": stats.mean(xs), "sd": stats.stdev(xs)}
    dd = [t["decision_diff_frac"] for t in trials]
    gap = [t["mean_max_marginal_gap"] for t in trials]
    out["decision_diff_frac"] = {"mean": stats.mean(dd), "sd": stats.stdev(dd)}
    out["mean_max_marginal_gap"] = {"mean": stats.mean(gap), "sd": stats.stdev(gap)}
    return out


def main():
    results = {}
    for sim in SIMULATORS:
        trials = [run_trial(SEED0 + t, sim) for t in range(TRIALS)]
        results[sim] = summarise(trials)
        print(f"\n=== simulator: {sim} ({TRIALS} trials x {ROUNDS} rounds) ===")
        print(f"{'rule':18s} {'Brier':>14s} {'log loss':>14s} {'recall@k':>14s} {'entropy':>14s}")
        for r in RULES:
            s = results[sim][r]
            print(f"{r:18s} " + " ".join(
                f"{s[m]['mean']:7.4f}±{s[m]['sd']:.3f}" for m in ("brier", "logloss", "recall_k", "entropy")))
        d = results[sim]["decision_diff_frac"]
        g = results[sim]["mean_max_marginal_gap"]
        print(f"planner chose a different path (factored vs exact, matching model): "
              f"{100*d['mean']:.1f}% of rounds (sd {100*d['sd']:.1f})")
        print(f"mean over rounds of max |factored - exact| marginal gap: {g['mean']:.3f} (sd {g['sd']:.3f})")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "experiment5_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
