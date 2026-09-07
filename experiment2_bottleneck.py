"""
experiment2_bottleneck.py
==========================
BN-DEVPP (Section 4.3 / Algorithm 3 of the reference paper) partitions the
search space into non-overlapping regions specifically so that each base
station's *local, parallel* belief update (Algorithm 3, line 6) never has to
be reconciled with another agent's observation of the same cell -- the
round's results are simply unioned at the server (Algorithm 3, line 8).

That assumption -- that a disjoint partition always exists -- holds in the
paper's own open, continuous 2D grid, where an entropy-weighted Voronoi
diagram can always bisect space. It does NOT hold on a graph with a cut
edge (a bridge, a single corridor, a mountain pass): if the only valuable
unexplored region lies on the far side of one bridge reachable from two
different wings, both agents will rationally choose to cross that same
physical edge in the same round, and there is no partition that gives each
agent exclusive access to it while leaving both wings connected to it.

We simulate exactly that: two agents, two wings, one shared bridge edge, and
compare two ways the server can fold both agents' round results back into
one shared map:

  * naive_union      -- each base station computes its local posterior from
                         the SAME round-start prior, in parallel, as
                         Algorithm 3 literally specifies; the server then
                         unions the two local results, and for the one edge
                         both agents touched, one agent's observation is
                         silently discarded (a coin toss in our
                         implementation, to avoid stacking the deck).
  * corrected_fusion -- the server instead composes the two agents'
                        observations sequentially (Bayes' rule applied
                        twice, prior -> posterior -> posterior), which is
                        exact for conditionally-independent evidence about
                        the same shared variable (see inference.py).
"""
import csv
import json
import random
import statistics as stats

from src import environment, inference, planner, metrics, svgplot

P_HAZ = 0.20
LETHALITY = 0.85
FALSE_POSITIVE = 0.04
ROUNDS = 6
TRIALS = 400
MAX_OUT_LEN = 5
SEED0 = 8675309


def simulate_agent(graph, belief, base, rng, truth):
    path = planner.plan_best_path(graph, belief, base, LETHALITY, FALSE_POSITIVE,
                                   max_out_len=MAX_OUT_LEN, max_candidates=250)
    destroyed = False
    for e in path:
        p_destroy = LETHALITY if truth[e] == 1 else FALSE_POSITIVE
        if rng.random() < p_destroy:
            destroyed = True
            break
    return path, destroyed


def run_trial(seed):
    rng = random.Random(seed)
    graph = environment.generate_bottleneck_facility()
    truth = environment.assign_hazards(graph, rng, P_HAZ)
    bridge = graph.bridge_edge

    # ONE shared "ground truth" planning belief drives what both agents
    # actually do each round, so that the two merge strategies below are
    # compared on identical exploration behaviour and identical outcomes --
    # isolating the comparison to the server-side fusion step alone.
    plan_belief = {e: P_HAZ for e in graph.edges}
    belief = {"naive_union": {e: P_HAZ for e in graph.edges},
              "corrected_fusion": {e: P_HAZ for e in graph.edges}}

    per_round = {m: {"bridge_entropy": [], "bridge_brier": [],
                      "map_entropy": [], "map_brier": []} for m in belief}
    overlap_flags = []

    for r in range(ROUNDS):
        path_a, destroyed_a = simulate_agent(graph, plan_belief, graph.base_a, rng, truth)
        path_b, destroyed_b = simulate_agent(graph, plan_belief, graph.base_b, rng, truth)
        shared = set(path_a) & set(path_b)
        overlap_flags.append(bridge in shared)

        for m in belief:
            round_start = dict(belief[m])
            if m == "corrected_fusion":
                merged = inference.bn_exact_update(round_start, path_a, destroyed_a,
                                                    LETHALITY, FALSE_POSITIVE)
                merged = inference.bn_exact_update(merged, path_b, destroyed_b,
                                                    LETHALITY, FALSE_POSITIVE)
            else:  # naive_union
                cand_a = inference.bn_exact_update(round_start, path_a, destroyed_a,
                                                    LETHALITY, FALSE_POSITIVE)
                cand_b = inference.bn_exact_update(round_start, path_b, destroyed_b,
                                                    LETHALITY, FALSE_POSITIVE)
                merged = dict(round_start)
                only_a = set(path_a) - shared
                only_b = set(path_b) - shared
                for e in only_a:
                    merged[e] = cand_a[e]
                for e in only_b:
                    merged[e] = cand_b[e]
                for e in shared:
                    # a coin toss decides whose local result "wins" the union
                    # -- the point is that ONE agent's evidence about this
                    # edge is discarded, not which one.
                    merged[e] = cand_a[e] if rng.random() < 0.5 else cand_b[e]

            belief[m] = merged
            per_round[m]["bridge_entropy"].append(metrics.binary_entropy(belief[m][bridge]))
            per_round[m]["bridge_brier"].append((belief[m][bridge] - truth[bridge]) ** 2)
            per_round[m]["map_entropy"].append(metrics.map_entropy(belief[m]))
            per_round[m]["map_brier"].append(metrics.brier_score(belief[m], truth))

        # planning belief itself follows the corrected (best-available) map
        plan_belief = belief["corrected_fusion"]

    return per_round, overlap_flags, truth[bridge]


def main():
    methods = ["naive_union", "corrected_fusion"]
    labels = {"naive_union": "Naive union (Algorithm 3, as specified)",
              "corrected_fusion": "Corrected sequential fusion (ours)"}

    all_rounds = {m: {k: [[] for _ in range(ROUNDS)]
                       for k in ("bridge_entropy", "bridge_brier", "map_entropy", "map_brier")}
                  for m in methods}
    total_overlap_rounds = 0
    total_rounds = 0
    n_bridge_hazard_true = 0
    # bridge Brier score in the round immediately AFTER an overlap round,
    # conditioned on overlap having actually happened that round
    post_overlap_bridge_brier = {m: [] for m in methods}
    post_no_overlap_bridge_brier = {m: [] for m in methods}

    for t in range(TRIALS):
        per_round, overlap_flags, bridge_truth = run_trial(SEED0 + t)
        total_overlap_rounds += sum(overlap_flags)
        total_rounds += ROUNDS
        n_bridge_hazard_true += bridge_truth
        for m in methods:
            for r in range(ROUNDS):
                for k in per_round[m]:
                    all_rounds[m][k][r].append(per_round[m][k][r])
                if overlap_flags[r]:
                    post_overlap_bridge_brier[m].append(per_round[m]["bridge_brier"][r])
                else:
                    post_no_overlap_bridge_brier[m].append(per_round[m]["bridge_brier"][r])

    summary_rows = []
    for r in range(ROUNDS):
        row = {"round": r + 1}
        for m in methods:
            for k in ("bridge_entropy", "bridge_brier", "map_entropy", "map_brier"):
                vals = all_rounds[m][k][r]
                row[f"{m}_{k}_mean"] = stats.mean(vals)
                row[f"{m}_{k}_std"] = stats.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    with open("results/experiment2_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    final = {
        "trials": TRIALS, "rounds": ROUNDS, "p_haz": P_HAZ,
        "lethality": LETHALITY, "false_positive": FALSE_POSITIVE,
        "fraction_of_rounds_with_forced_bridge_overlap":
            total_overlap_rounds / total_rounds,
        "fraction_trials_bridge_truly_hazardous": n_bridge_hazard_true / TRIALS,
        "final_round_bridge_metrics": {
            m: {
                "entropy_mean": stats.mean(all_rounds[m]["bridge_entropy"][ROUNDS - 1]),
                "brier_mean": stats.mean(all_rounds[m]["bridge_brier"][ROUNDS - 1]),
            } for m in methods
        },
        "final_round_map_metrics": {
            m: {
                "entropy_mean": stats.mean(all_rounds[m]["map_entropy"][ROUNDS - 1]),
                "brier_mean": stats.mean(all_rounds[m]["map_brier"][ROUNDS - 1]),
            } for m in methods
        },
        "bridge_brier_conditioned_on_overlap_this_round": {
            m: {
                "overlap_rounds_mean": stats.mean(post_overlap_bridge_brier[m]),
                "n_overlap_rounds": len(post_overlap_bridge_brier[m]),
                "no_overlap_rounds_mean": stats.mean(post_no_overlap_bridge_brier[m]),
                "n_no_overlap_rounds": len(post_no_overlap_bridge_brier[m]),
            } for m in methods
        },
    }
    with open("results/experiment2_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    rounds_x = list(range(1, ROUNDS + 1))
    svgplot.line_chart(
        "results/figures/experiment2_bridge_brier.svg",
        "Experiment 2: Brier score of the shared bottleneck edge",
        "Search round", "Brier score (lower = more correct)",
        [{"label": labels[m], "x": rounds_x,
          "y": [stats.mean(all_rounds[m]["bridge_brier"][r]) for r in range(ROUNDS)]}
         for m in methods],
    )
    svgplot.line_chart(
        "results/figures/experiment2_map_brier.svg",
        "Experiment 2: whole-map Brier score, naive union vs. corrected fusion",
        "Search round", "Brier score (lower = more correct)",
        [{"label": labels[m], "x": rounds_x,
          "y": [stats.mean(all_rounds[m]["map_brier"][r]) for r in range(ROUNDS)]}
         for m in methods],
    )

    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
