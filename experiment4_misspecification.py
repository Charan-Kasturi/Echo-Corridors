"""
experiment4_misspecification.py
=================================
The reference paper tests robustness to false positives/negatives (Section
6: "we also incorporate false-positive and false-negative sensor
observations, making the model resilient to noisy data"), but in every
reported experiment the belief-update model is handed the *true* generating
rates -- lethality and the 5%/5% fault probabilities are known exactly by
the inference procedure, because the same numbers are used to both simulate
the world and to update the belief map. Real deployments do not get this
luxury: the false-positive (mechanical/component fault) and false-negative
(survived-a-real-hazard) rates used in the update are estimates, typically
biased toward overconfidence ("our robots rarely fail on their own").

This experiment holds the TRUE simulation parameters fixed and varies only
the ASSUMED parameters fed to the planner and to each belief-update rule,
to ask: does the Bayesian-network method's reported noise resilience
survive when the noise *rate itself* is misestimated, and does it degrade
faster or slower than the older weighted-average baseline under the same
misspecification?
"""
import csv
import json
import random
import statistics as stats

from src import environment, inference, planner, metrics, svgplot

TRUE_LETHALITY = 0.85
TRUE_FALSE_POSITIVE = 0.04
P_HAZ = 0.15
ROUNDS = 40
TRIALS = 30
MAX_OUT_LEN = 4
SEED0 = 42424242

METHODS = ["weighted_average", "bn_exact"]
LABELS = {"weighted_average": "Weighted-average [18]", "bn_exact": "Bayesian network (ours, exact)"}
UPDATE_FN = {"weighted_average": inference.weighted_average_update, "bn_exact": inference.bn_exact_update}

FP_SHRINK_LEVELS = [1.0, 0.5, 0.25, 0.10, 0.02]        # assumed_fp = TRUE_FP * shrink
L_BIAS_LEVELS = [-0.35, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35]  # assumed_L = TRUE_L * (1+bias), clamped


def clamp01(x):
    return min(0.999, max(0.001, x))


def run_trial(seed, assumed_l, assumed_fp):
    rng = random.Random(seed)
    graph = environment.generate_random_facility(rng, rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)
    truth = environment.assign_hazards(graph, rng, P_HAZ)

    belief = {m: {e: P_HAZ for e in graph.edges} for m in METHODS}
    plan_belief = belief["bn_exact"]

    for r in range(ROUNDS):
        path = planner.plan_best_path(graph, plan_belief, graph.base, assumed_l, assumed_fp,
                                       max_out_len=MAX_OUT_LEN, max_candidates=200)
        if not path:
            break
        destroyed = False
        for e in path:
            p_destroy = TRUE_LETHALITY if truth[e] == 1 else TRUE_FALSE_POSITIVE
            if rng.random() < p_destroy:
                destroyed = True
                break
        for m in METHODS:
            belief[m] = UPDATE_FN[m](belief[m], path, destroyed, assumed_l, assumed_fp)

    return {m: {"brier": metrics.brier_score(belief[m], truth),
                "logloss": metrics.log_loss(belief[m], truth),
                "entropy": metrics.map_entropy(belief[m])} for m in METHODS}


def sweep(levels, make_rates, tag):
    rows = []
    for level in levels:
        assumed_l, assumed_fp = make_rates(level)
        agg = {m: {"brier": [], "logloss": [], "entropy": []} for m in METHODS}
        for t in range(TRIALS):
            result = run_trial(SEED0 + hash((tag, level, t)) % 1_000_000, assumed_l, assumed_fp)
            for m in METHODS:
                for k in agg[m]:
                    agg[m][k].append(result[m][k])
        row = {"level": level, "assumed_lethality": assumed_l, "assumed_false_positive": assumed_fp}
        for m in METHODS:
            row[f"{m}_brier_mean"] = stats.mean(agg[m]["brier"])
            row[f"{m}_brier_std"] = stats.pstdev(agg[m]["brier"]) if TRIALS > 1 else 0.0
            row[f"{m}_logloss_mean"] = stats.mean(agg[m]["logloss"])
            row[f"{m}_entropy_mean"] = stats.mean(agg[m]["entropy"])
        rows.append(row)
    return rows


def main():
    fp_rows = sweep(FP_SHRINK_LEVELS,
                     lambda lvl: (TRUE_LETHALITY, clamp01(TRUE_FALSE_POSITIVE * lvl)),
                     "fp")
    l_rows = sweep(L_BIAS_LEVELS,
                    lambda lvl: (clamp01(TRUE_LETHALITY * (1 + lvl)), TRUE_FALSE_POSITIVE),
                    "l")

    with open("results/experiment4_fp_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
        w.writeheader(); w.writerows(fp_rows)
    with open("results/experiment4_lethality_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(l_rows[0].keys()))
        w.writeheader(); w.writerows(l_rows)

    summary = {
        "true_lethality": TRUE_LETHALITY, "true_false_positive": TRUE_FALSE_POSITIVE,
        "p_haz": P_HAZ, "rounds": ROUNDS, "trials_per_point": TRIALS,
        "fp_shrink_sweep": fp_rows,
        "lethality_bias_sweep": l_rows,
    }
    with open("results/experiment4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    svgplot.line_chart(
        "results/figures/experiment4_fp_sweep.svg",
        "Experiment 4: effect of underestimating the false-positive rate",
        "Assumed false-positive rate as a fraction of the true rate", "Final Brier score",
        [{"label": LABELS[m], "x": [row["assumed_false_positive"] / TRUE_FALSE_POSITIVE for row in fp_rows],
          "y": [row[f"{m}_brier_mean"] for row in fp_rows]} for m in METHODS],
    )
    svgplot.line_chart(
        "results/figures/experiment4_lethality_sweep.svg",
        "Experiment 4: effect of a biased lethality assumption",
        "Relative bias in assumed lethality (0 = correctly specified)", "Final Brier score",
        [{"label": LABELS[m], "x": [row["level"] for row in l_rows],
          "y": [row[f"{m}_brier_mean"] for row in l_rows]} for m in METHODS],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
