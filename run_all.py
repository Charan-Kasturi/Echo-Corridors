"""
run_all.py
==========
Regenerates every result used by the report, from one commit, in one go:

  1. Experiments 1, 2, 4 and 5 in sequence (each script's own main(), which
     writes its summary CSV/JSON and colour figures under results/).
  2. Per-trial raw results under results/raw/ (one row per trial and method),
     produced by re-running each experiment's run_trial() with the same seeds.
     Every trial seeds its own RNG, so these are the exact trials behind the
     summaries.
  3. An audit: every mean in the summary files is recomputed from the raw
     CSVs and must agree to 1e-12, or this script exits non-zero.
  4. Paired differences with standard errors (results/paired_differences.json),
     computed from the raw CSVs.
  5. The Experiment 5 sanity check (check_exp5_exact_vs_closed_form.py).
  6. results/run_manifest.json: git commit, Python/numpy versions, date,
     seeds and settings for each experiment, and every output file written
     (with SHA-256).

Run from the repository root:
    python run_all.py
    python make_print_figures.py     # black-and-white figures for the print report
"""
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import statistics as stats
import subprocess
import sys
import time

import numpy as np

import experiment1_revisits as e1
import experiment2_bottleneck as e2
import experiment4_misspecification as e4
import experiment5_exact_joint as e5
import check_exp5_exact_vs_closed_form as check5

RAW = os.path.join("results", "raw")
AUDIT_TOL = 1e-12


# --------------------------------------------------------------------------- helpers

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def mean_se(xs):
    n = len(xs)
    sd = stats.stdev(xs) if n > 1 else 0.0
    return {"mean": stats.mean(xs), "se": sd / math.sqrt(n), "sd": sd, "n": n}


def paired(rows, key, a, b, metrics, method_field="method"):
    """Paired difference a - b over trials (rows grouped by `key`)."""
    by = {}
    for r in rows:
        by.setdefault(tuple(r[k] for k in key), {})[r[method_field]] = r
    out = {}
    for m in metrics:
        diffs = [g[a][m] - g[b][m] for g in by.values()]
        out[m] = mean_se(diffs)
    return out


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def eq(self, label, summary_value, raw_value):
        self.checks += 1
        if abs(summary_value - raw_value) > AUDIT_TOL:
            self.failures.append(f"{label}: summary={summary_value!r} raw={raw_value!r}")


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - git missing
        return f"unavailable ({exc})"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- raw trials

def raw_experiment1():
    rows = []
    for t in range(e1.TRIALS):
        seed = e1.SEED0 + t
        per_round, revisits, _example, n_edges = e1.run_trial(seed)
        for m in e1.METHODS:
            pr = per_round[m]
            rows.append({"trial": t, "seed": seed, "method": m, "n_edges": n_edges,
                         "rounds_completed": len(pr["entropy"]),
                         "revisit_edge_visits": revisits,
                         "entropy": pr["entropy"][-1], "brier": pr["brier"][-1],
                         "logloss": pr["logloss"][-1], "recall_at_k": pr["pr_k"][-1]})
    return rows


def raw_experiment2():
    rows = []
    for t in range(e2.TRIALS):
        seed = e2.SEED0 + t
        per_round, overlap_flags, bridge_truth = e2.run_trial(seed)
        for m in ("naive_union", "corrected_fusion"):
            pr = per_round[m]
            ov = [pr["bridge_brier"][r] for r in range(e2.ROUNDS) if overlap_flags[r]]
            no = [pr["bridge_brier"][r] for r in range(e2.ROUNDS) if not overlap_flags[r]]
            rows.append({"trial": t, "seed": seed, "method": m, "bridge_truly_hazardous": bridge_truth,
                         "n_overlap_rounds": len(ov), "n_no_overlap_rounds": len(no),
                         "overlap_rounds_bridge_brier_sum": sum(ov),
                         "no_overlap_rounds_bridge_brier_sum": sum(no),
                         "final_bridge_entropy": pr["bridge_entropy"][-1],
                         "final_bridge_brier": pr["bridge_brier"][-1],
                         "final_map_entropy": pr["map_entropy"][-1],
                         "final_map_brier": pr["map_brier"][-1]})
    return rows


def experiment4_levels():
    """(sweep, tag_offset, level_idx, level, assumed_L, assumed_FP), as in experiment4.main()."""
    out = []
    for i, lvl in enumerate(e4.FP_SHRINK_LEVELS):
        out.append(("fp_shrink", 0, i, lvl, e4.TRUE_LETHALITY,
                    e4.clamp01(e4.TRUE_FALSE_POSITIVE * lvl)))
    for i, lvl in enumerate(e4.L_BIAS_LEVELS):
        out.append(("lethality_bias", 500_000, i, lvl,
                    e4.clamp01(e4.TRUE_LETHALITY * (1 + lvl)), e4.TRUE_FALSE_POSITIVE))
    return out


def raw_experiment4():
    rows = []
    for sweep, offset, i, lvl, aL, aFP in experiment4_levels():
        for t in range(e4.TRIALS):
            seed = e4.SEED0 + offset + i * 1_000 + t
            res = e4.run_trial(seed, aL, aFP)
            for m in e4.METHODS:
                rows.append({"sweep": sweep, "level": lvl, "assumed_lethality": aL,
                             "assumed_false_positive": aFP, "trial": t, "seed": seed,
                             "method": m, **res[m]})
    return rows


def raw_experiment5():
    rows = []
    for sim in e5.SIMULATORS:
        for t in range(e5.TRIALS):
            seed = e5.SEED0 + t
            res = e5.run_trial(seed, sim)
            for rule in e5.RULES:
                rows.append({"simulator": sim, "trial": t, "seed": seed, "rule": rule,
                             "n_edges": res["n_edges"], "n_hazards": res["n_hazards"],
                             "decision_diff_frac": res["decision_diff_frac"],
                             "mean_max_marginal_gap": res["mean_max_marginal_gap"],
                             **res["final"][rule]})
    return rows


# --------------------------------------------------------------------------- audit

def audit_all(audit, r1, r2, r4, r5):
    with open("results/experiment1_summary.json") as f:
        s1 = json.load(f)
    assert all(r["rounds_completed"] == e1.ROUNDS for r in r1), "exp1: a trial ended early"
    for m in e1.METHODS:
        rows = [r for r in r1 if r["method"] == m]
        fm = s1["final_round_metrics"][m]
        for k_sum, k_raw in [("entropy_mean", "entropy"), ("brier_mean", "brier"),
                             ("logloss_mean", "logloss"), ("recall_at_k_mean", "recall_at_k")]:
            audit.eq(f"exp1 {m} {k_sum}", fm[k_sum], stats.mean(r[k_raw] for r in rows))
    audit.eq("exp1 avg revisits", s1["avg_revisit_edge_visits_per_trial"],
             stats.mean(r["revisit_edge_visits"] for r in r1 if r["method"] == "bn_exact"))

    with open("results/experiment2_summary.json") as f:
        s2 = json.load(f)
    one = [r for r in r2 if r["method"] == "corrected_fusion"]
    audit.eq("exp2 overlap fraction", s2["fraction_of_rounds_with_forced_bridge_overlap"],
             sum(r["n_overlap_rounds"] for r in one) / (len(one) * e2.ROUNDS))
    for m in ("naive_union", "corrected_fusion"):
        rows = [r for r in r2 if r["method"] == m]
        audit.eq(f"exp2 {m} final bridge brier", s2["final_round_bridge_metrics"][m]["brier_mean"],
                 stats.mean(r["final_bridge_brier"] for r in rows))
        audit.eq(f"exp2 {m} final map brier", s2["final_round_map_metrics"][m]["brier_mean"],
                 stats.mean(r["final_map_brier"] for r in rows))
        c = s2["bridge_brier_conditioned_on_overlap_this_round"][m]
        audit.eq(f"exp2 {m} overlap-round bridge brier", c["overlap_rounds_mean"],
                 sum(r["overlap_rounds_bridge_brier_sum"] for r in rows)
                 / sum(r["n_overlap_rounds"] for r in rows))
        audit.eq(f"exp2 {m} no-overlap-round bridge brier", c["no_overlap_rounds_mean"],
                 sum(r["no_overlap_rounds_bridge_brier_sum"] for r in rows)
                 / sum(r["n_no_overlap_rounds"] for r in rows))

    with open("results/experiment4_summary.json") as f:
        s4 = json.load(f)
    for sweep, key in [("fp_shrink", "fp_shrink_sweep"), ("lethality_bias", "lethality_bias_sweep")]:
        for row in s4[key]:
            for m in e4.METHODS:
                rows = [r for r in r4 if r["sweep"] == sweep and r["level"] == row["level"]
                        and r["method"] == m]
                assert len(rows) == e4.TRIALS
                for k in ("brier", "logloss", "entropy"):
                    audit.eq(f"exp4 {sweep} {row['level']} {m} {k}", row[f"{m}_{k}_mean"],
                             stats.mean(r[k] for r in rows))

    with open("results/experiment5_summary.json") as f:
        s5 = json.load(f)
    for sim in e5.SIMULATORS:
        for rule in e5.RULES:
            rows = [r for r in r5 if r["simulator"] == sim and r["rule"] == rule]
            for k in ("brier", "logloss", "recall_k", "entropy"):
                audit.eq(f"exp5 {sim} {rule} {k}", s5[sim][rule][k]["mean"],
                         stats.mean(r[k] for r in rows))
        rows = [r for r in r5 if r["simulator"] == sim and r["rule"] == e5.RULES[0]]
        for k in ("decision_diff_frac", "mean_max_marginal_gap"):
            audit.eq(f"exp5 {sim} {k}", s5[sim][k]["mean"], stats.mean(r[k] for r in rows))


# --------------------------------------------------------------------------- paired stats

def paired_all(r1, r4, r5):
    out = {"note": "Paired over trials, first minus second; positive = first is worse "
                   "(higher Brier / log loss). se = sample sd / sqrt(n)."}
    out["experiment1"] = {
        "bn_paper_literal_minus_bn_exact": paired(r1, ["trial"], "bn_paper_literal", "bn_exact",
                                                  ["brier", "logloss", "entropy", "recall_at_k"]),
        "weighted_average_minus_bn_exact": paired(r1, ["trial"], "weighted_average", "bn_exact",
                                                  ["brier", "logloss", "entropy", "recall_at_k"]),
    }
    out["experiment4"] = {}
    for sweep in ("fp_shrink", "lethality_bias"):
        levels = []
        for lvl in dict.fromkeys(r["level"] for r in r4 if r["sweep"] == sweep):
            rows = [r for r in r4 if r["sweep"] == sweep and r["level"] == lvl]
            levels.append({"level": lvl, **paired(rows, ["trial"], "weighted_average", "bn_exact",
                                                  ["brier", "logloss"])})
        out["experiment4"][f"{sweep}_weighted_average_minus_bn_exact"] = levels
    out["experiment5"] = {}
    for sim in e5.SIMULATORS:
        rows = [r for r in r5 if r["simulator"] == sim]
        if sim == "independent":
            fac, joint, wrong = "factored_indep", "joint_indep", "factored_persist"
        else:
            fac, joint, wrong = "factored_persist", "joint_persist", "factored_indep"
        out["experiment5"][sim] = {
            "factored_minus_exact_matching_model":
                paired(rows, ["trial"], fac, joint, ["brier", "logloss", "recall_k"], "rule"),
            "wrong_minus_right_visit_model_factored":
                paired(rows, ["trial"], wrong, fac, ["brier", "logloss", "recall_k"], "rule"),
        }
    return out


# --------------------------------------------------------------------------- main

def settings(mod, names):
    return {n: getattr(mod, n) for n in names if hasattr(mod, n)}


def main():
    t_start = time.time()
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    steps = [("experiment1_revisits", e1.main), ("experiment2_bottleneck", e2.main),
             ("experiment4_misspecification", e4.main), ("experiment5_exact_joint", e5.main)]
    timings = {}
    for name, fn in steps:
        print(f"\n######## {name} ########", flush=True)
        t0 = time.time()
        fn()
        timings[name] = round(time.time() - t0, 2)

    print("\n######## per-trial raw results ########", flush=True)
    r1, r2, r4, r5 = raw_experiment1(), raw_experiment2(), raw_experiment4(), raw_experiment5()
    write_csv(os.path.join(RAW, "experiment1_trials.csv"), r1)
    write_csv(os.path.join(RAW, "experiment2_trials.csv"), r2)
    write_csv(os.path.join(RAW, "experiment4_trials.csv"), r4)
    write_csv(os.path.join(RAW, "experiment5_trials.csv"), r5)

    audit = Audit()
    audit_all(audit, r1, r2, r4, r5)
    print(f"audit: {audit.checks} summary means recomputed from raw CSVs, "
          f"{len(audit.failures)} mismatches")
    for msg in audit.failures:
        print("  MISMATCH", msg)

    pairs = paired_all(r1, r4, r5)
    with open(os.path.join("results", "paired_differences.json"), "w") as f:
        json.dump(pairs, f, indent=2)

    print("\n######## Experiment 5 sanity check ########", flush=True)
    worst, n_cases, _ = check5.run()
    check_ok = all(v <= check5.TOL for v in worst.values())
    print(f"max |exact joint - closed form| after one update: {worst}  "
          f"{'PASS' if check_ok else 'FAIL'}")

    outputs = []
    for root, _dirs, files in os.walk("results"):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if os.path.getmtime(p) >= t_start - 1 and not p.endswith("run_manifest.json"):
                outputs.append({"path": p.replace(os.sep, "/"), "sha256": sha256(p)})
    outputs.sort(key=lambda o: o["path"])

    dirty = git("status", "--porcelain", "--untracked-files=all")
    manifest = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_worktree_dirty": bool(dirty),
        "git_dirty_paths_excluding_results": [l for l in dirty.splitlines()
                                             if not l[3:].startswith("results/")],
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "command": "python run_all.py",
        "runtime_seconds": timings,
        "experiments": {
            "experiment1_revisits": {
                "script": "experiment1_revisits.py",
                "seeds": f"SEED0 + trial, trial = 0..{e1.TRIALS - 1}",
                **settings(e1, ["SEED0", "TRIALS", "ROUNDS", "P_HAZ", "LETHALITY",
                                "FALSE_POSITIVE", "MAX_OUT_LEN", "METHODS"]),
                "graph": "generate_random_facility(rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)",
                "planner_max_candidates": 250,
            },
            "experiment2_bottleneck": {
                "script": "experiment2_bottleneck.py",
                "seeds": f"SEED0 + trial, trial = 0..{e2.TRIALS - 1}",
                **settings(e2, ["SEED0", "TRIALS", "ROUNDS", "P_HAZ", "LETHALITY",
                                "FALSE_POSITIVE", "MAX_OUT_LEN"]),
                "graph": "generate_bottleneck_facility()",
                "planner_max_candidates": 250,
            },
            "experiment4_misspecification": {
                "script": "experiment4_misspecification.py",
                "seeds": "SEED0 + sweep_offset + level_index*1000 + trial; "
                         "sweep_offset = 0 (fp_shrink) or 500000 (lethality_bias)",
                **settings(e4, ["SEED0", "TRIALS", "ROUNDS", "P_HAZ", "TRUE_LETHALITY",
                                "TRUE_FALSE_POSITIVE", "MAX_OUT_LEN", "METHODS",
                                "FP_SHRINK_LEVELS", "L_BIAS_LEVELS"]),
                "graph": "generate_random_facility(rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)",
                "planner_max_candidates": 200,
            },
            "experiment5_exact_joint": {
                "script": "experiment5_exact_joint.py",
                "seeds": f"SEED0 + trial, trial = 0..{e5.TRIALS - 1}, same seeds for each simulator",
                **settings(e5, ["SEED0", "TRIALS", "ROUNDS", "P_HAZ", "LETHALITY",
                                "FALSE_POSITIVE", "MAX_OUT_LEN", "SIMULATORS", "RULES"]),
                "simulator_names": {"persistent": "once-per-round in the report"},
                "graph": "generate_random_facility(rows=4, cols=4, prune_prob=0.35, min_dead_ends=2)",
                "planner_max_candidates": 250,
            },
            "check_exp5_exact_vs_closed_form": {
                "script": "check_exp5_exact_vs_closed_form.py",
                "seed": check5.SEED, "cases": n_cases, "tolerance": check5.TOL,
                "max_abs_difference": worst, "passed": check_ok,
            },
        },
        "audit": {"checks": audit.checks, "tolerance": AUDIT_TOL,
                  "mismatches": audit.failures},
        "outputs": outputs,
    }
    with open(os.path.join("results", "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest: results/run_manifest.json ({len(outputs)} output files)")

    if audit.failures or not check_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
