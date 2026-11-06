"""
experiment1_revisits.py
========================
Does the reference paper's Bayesian-network update rule (as literally
specified, including Remark 1's under-specified treatment of repeat visits)
converge to correct hazard estimates when a realistic graph environment
forces an agent to revisit corridors?

Three update rules see the *exact same* sequence of planned paths and
observed outcomes each round (only the update rule differs, isolating the
comparison to estimation quality, not exploration behaviour):

  * weighted_average  -- Otte & Sofge [18] / Srivastava et al. [26]
  * bn_paper_literal  -- this paper's Bayesian network, with repeat visits
                         handled by literally following Remark 1 ("summing
                         the probability of each instance")
  * bn_exact          -- our corrected closed form for repeat visits
                         (see inference.py docstring for the derivation)

Ground truth for scoring is the actual planted hazard map, so we can report
Brier score and top-k precision/recall in addition to entropy.
"""
import csv
import json
import random
import statistics as stats

from src import environment, inference, planner, metrics, svgplot

P_HAZ = 0.15
LETHALITY = 0.85
FALSE_POSITIVE = 0.04
ROUNDS = 45
TRIALS = 40
MAX_OUT_LEN = 4
SEED0 = 20260906

METHODS = ["weighted_average", "bn_paper_literal", "bn_exact"]
LABELS = {
    "weighted_average": "Weighted-average [18]",
    "bn_paper_literal": "BN, paper's literal revisit rule",
    "bn_exact": "BN, corrected revisit rule (ours)",
}
UPDATE_FN = {
    "weighted_average": inference.weighted_average_update,
    "bn_paper_literal": inference.bn_literal_revisit_update,
    "bn_exact": inference.bn_exact_update,
}


def run_trial(seed):
    rng = random.Random(seed)
    graph = environment.generate_random_facility(rng, rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)
    truth = environment.assign_hazards(graph, rng, P_HAZ)
    n_true_hazards = sum(truth.values())

    belief = {m: {e: P_HAZ for e in graph.edges} for m in METHODS}
    plan_belief = belief["bn_exact"]  # the agent plans with its best (exact) model

    per_round = {m: {"entropy": [], "brier": [], "logloss": [], "pr_k": []} for m in METHODS}
    total_revisit_edge_visits = 0
    example = None

    for r in range(ROUNDS):
        plan_belief = belief["bn_exact"]  # replan from the updated belief each round
        path = planner.plan_best_path(graph, plan_belief, graph.base, LETHALITY, FALSE_POSITIVE,
                                       max_out_len=MAX_OUT_LEN, max_candidates=250)
        if not path:
            break

        counts = {}
        for e in path:
            counts[e] = counts.get(e, 0) + 1
        revisited_edges = {e: n for e, n in counts.items() if n > 1}
        for n in revisited_edges.values():
            total_revisit_edge_visits += (n - 1)

        destroyed = False
        for e in path:
            p_destroy = LETHALITY if truth[e] == 1 else FALSE_POSITIVE
            if rng.random() < p_destroy:
                destroyed = True
                break

        # capture a clean, still-uncertain "before/after" snapshot the FIRST
        # time a meaningfully revisited edge (>=2 visits) is updated, before
        # many later rounds drive beliefs to numerical extremes.
        if example is None and revisited_edges:
            target = max(revisited_edges, key=revisited_edges.get)
            example = {
                "edge": target,
                "n_visits_this_round": revisited_edges[target],
                "true_hazard": truth[target],
                "destroyed_this_round": destroyed,
                "belief_before": {m: belief[m][target] for m in METHODS},
            }

        for m in METHODS:
            belief[m] = UPDATE_FN[m](belief[m], path, destroyed, LETHALITY, FALSE_POSITIVE)
            per_round[m]["entropy"].append(metrics.map_entropy(belief[m]))
            per_round[m]["brier"].append(metrics.brier_score(belief[m], truth))
            per_round[m]["logloss"].append(metrics.log_loss(belief[m], truth))
            _, rec = metrics.precision_recall_at_k(belief[m], truth, n_true_hazards)
            per_round[m]["pr_k"].append(rec)

        if example is not None and "belief_after" not in example:
            example["belief_after"] = {m: belief[m][example["edge"]] for m in METHODS}

    return per_round, total_revisit_edge_visits, example, len(graph.edges)


def main():
    all_rounds = {m: {"entropy": [[] for _ in range(ROUNDS)],
                       "brier": [[] for _ in range(ROUNDS)],
                       "logloss": [[] for _ in range(ROUNDS)],
                       "pr_k": [[] for _ in range(ROUNDS)]} for m in METHODS}
    total_revisits = 0
    examples = []
    completed_trials = 0

    for t in range(TRIALS):
        per_round, revisits, example, n_edges = run_trial(SEED0 + t)
        total_revisits += revisits
        if example:
            examples.append(example)
        n_rounds_done = len(per_round["bn_exact"]["entropy"])
        completed_trials += 1
        for m in METHODS:
            for r in range(n_rounds_done):
                for k in ("entropy", "brier", "logloss", "pr_k"):
                    all_rounds[m][k][r].append(per_round[m][k][r])

    summary_rows = []
    for r in range(ROUNDS):
        row = {"round": r + 1}
        for m in METHODS:
            ent = all_rounds[m]["entropy"][r]
            bri = all_rounds[m]["brier"][r]
            ll = all_rounds[m]["logloss"][r]
            rec = all_rounds[m]["pr_k"][r]
            if not ent:
                continue
            row[f"{m}_entropy_mean"] = stats.mean(ent)
            row[f"{m}_entropy_std"] = stats.pstdev(ent) if len(ent) > 1 else 0.0
            row[f"{m}_brier_mean"] = stats.mean(bri)
            row[f"{m}_brier_std"] = stats.pstdev(bri) if len(bri) > 1 else 0.0
            row[f"{m}_logloss_mean"] = stats.mean(ll)
            row[f"{m}_recall_at_k_mean"] = stats.mean(rec)
        summary_rows.append(row)

    with open("results/experiment1_summary.csv", "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    final = {
        "trials": completed_trials,
        "rounds": ROUNDS,
        "p_haz": P_HAZ, "lethality": LETHALITY, "false_positive": FALSE_POSITIVE,
        "total_revisit_edge_visits_across_all_trials": total_revisits,
        "avg_revisit_edge_visits_per_trial": total_revisits / completed_trials,
        "final_round_metrics": {
            m: {
                "entropy_mean": stats.mean(all_rounds[m]["entropy"][ROUNDS - 1]),
                "brier_mean": stats.mean(all_rounds[m]["brier"][ROUNDS - 1]),
                "logloss_mean": stats.mean(all_rounds[m]["logloss"][ROUNDS - 1]),
                "recall_at_k_mean": stats.mean(all_rounds[m]["pr_k"][ROUNDS - 1]),
            } for m in METHODS
        },
        "worked_examples": examples[:5],
    }
    with open("results/experiment1_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    rounds_x = list(range(1, ROUNDS + 1))
    svgplot.line_chart(
        "results/figures/experiment1_entropy.svg",
        "Experiment 1: belief-map entropy vs. search round (revisit-forcing graph)",
        "Search round", "Total map entropy (nats)",
        [{"label": LABELS[m], "x": rounds_x,
          "y": [stats.mean(all_rounds[m]["entropy"][r]) for r in range(ROUNDS)]}
         for m in METHODS],
    )
    svgplot.line_chart(
        "results/figures/experiment1_brier.svg",
        "Experiment 1: Brier score vs. ground truth (revisit-forcing graph)",
        "Search round", "Brier score (lower = more correct)",
        [{"label": LABELS[m], "x": rounds_x,
          "y": [stats.mean(all_rounds[m]["brier"][r]) for r in range(ROUNDS)]}
         for m in METHODS],
    )
    svgplot.line_chart(
        "results/figures/experiment1_recall.svg",
        "Experiment 1: recall@k of true hazard edges (k = true hazard count)",
        "Search round", "Recall@k",
        [{"label": LABELS[m], "x": rounds_x,
          "y": [stats.mean(all_rounds[m]["pr_k"][r]) for r in range(ROUNDS)]}
         for m in METHODS],
        y_min=0.0, y_max=1.0,
    )

    print(json.dumps(final["final_round_metrics"], indent=2))
    print(f"trials={completed_trials}  avg revisit-edge-visits/trial="
          f"{final['avg_revisit_edge_visits_per_trial']:.2f}")
    if examples:
        print("worked example (edge with most revisits in a single round):")
        print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    main()
