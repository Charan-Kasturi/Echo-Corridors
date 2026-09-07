"""
make_print_figures.py
======================
Regenerates every figure used by the print report (report/echo_corridors_report.html)
as a strictly black-and-white, dash-differentiated SVG under results/figures/print/,
matching the monochrome print styling used throughout that document. Run this after
any of the three experiment scripts, so the print figures stay in sync with the
color ones used in README.md / report/REPORT.md.

Requires results/experiment{1,2,4}_summary.json to already exist (run the three
experiment*.py scripts first) plus make_diagrams.py / make_animation.py's bw outputs
(regenerated here directly).
"""
import json
import os
import statistics as stats

from src import svgplot
import make_diagrams
import make_animation

OUT = "results/figures/print"


def entropy_brier_recall_charts():
    with open("results/experiment1_summary.csv") as f:
        import csv
        rows = list(csv.DictReader(f))
    methods = ["weighted_average", "bn_paper_literal", "bn_exact"]
    labels = {"weighted_average": "Weighted-average [18]",
              "bn_paper_literal": "BN, paper's literal revisit rule",
              "bn_exact": "BN, corrected revisit rule (ours)"}
    rounds_x = [int(r["round"]) for r in rows]

    for metric, fname, ylabel in [
        ("entropy_mean", "experiment1_entropy_bw.svg", "Total map entropy (nats)"),
        ("brier_mean", "experiment1_brier_bw.svg", "Brier score (lower = more correct)"),
    ]:
        series = [{"label": labels[m], "x": rounds_x,
                   "y": [float(r[f"{m}_{metric}"]) for r in rows]} for m in methods]
        svgplot.line_chart(os.path.join(OUT, fname), "", "Search round", ylabel, series, bw=True)


def bottleneck_charts():
    import csv
    with open("results/experiment2_summary.csv") as f:
        rows = list(csv.DictReader(f))
    methods = ["naive_union", "corrected_fusion"]
    labels = {"naive_union": "Naive union (Algorithm 3, as specified)",
              "corrected_fusion": "Corrected sequential fusion (ours)"}
    rounds_x = [int(r["round"]) for r in rows]
    series = [{"label": labels[m], "x": rounds_x,
               "y": [float(r[f"{m}_bridge_brier_mean"]) for r in rows]} for m in methods]
    svgplot.line_chart(os.path.join(OUT, "experiment2_bridge_brier_bw.svg"), "",
                        "Search round", "Bridge-edge Brier score", series, bw=True)


def misspecification_chart():
    with open("results/experiment4_summary.json") as f:
        d = json.load(f)
    rows = d["fp_shrink_sweep"]
    methods = ["weighted_average", "bn_exact"]
    labels = {"weighted_average": "Weighted-average [18]", "bn_exact": "Bayesian network (ours, exact)"}
    true_fp = d["true_false_positive"]
    series = [{"label": labels[m], "x": [row["assumed_false_positive"] / true_fp for row in rows],
               "y": [row[f"{m}_brier_mean"] for row in rows]} for m in methods]
    svgplot.line_chart(os.path.join(OUT, "experiment4_fp_sweep_bw.svg"), "",
                        "Assumed false-positive rate as a fraction of the true rate",
                        "Final Brier score", series, bw=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    entropy_brier_recall_charts()
    bottleneck_charts()
    misspecification_chart()
    make_diagrams.bottleneck_diagram(os.path.join(OUT, "bottleneck_topology_bw.svg"), bw=True)
    make_diagrams.sample_facility_diagram(os.path.join(OUT, "sample_facility_bw.svg"), bw=True)
    make_animation.single_agent_animation(os.path.join(OUT, "sim_single_agent_bw.svg"), bw=True)
    make_animation.bottleneck_animation(os.path.join(OUT, "sim_bottleneck_bw.svg"), bw=True)
    print("print figures written to", OUT)
