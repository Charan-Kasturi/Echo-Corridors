# Echo Corridors

Path-based sensors report a single boolean observation for an entire
traversed route: an agent that returns means "no hazard triggered along
this path," one that does not means "something happened, but not where."
Fusing many such observations into a spatial hazard belief map, and
choosing routes that gather information fastest, is the problem this
project implements and stress-tests.

This is an independent, from-scratch implementation of the Bayesian
network belief-update method described in Srivastava, Kontoudis, Sofge &
Otte, *"Bayesian Networks for Path-Based Sensors: Gathering Information and
Path Planning in Communication Denied Environments"* (2026), evaluated on a
**new environment** the original paper does not use: graph-shaped corridor
networks (floor plans, tunnels, bridges) instead of an open continuous
grid. That change in environment is what surfaces the two structural gaps
documented in the report
([`report/echo_corridors_report.html`](report/echo_corridors_report.html),
[PDF](report/echo_corridors_report.pdf)):

1. The paper's own prescription for a path that revisits a corridor is
   mathematically under-specified, and its natural reading is measurably
   less accurate than the corrected closed form derived here.
2. The multi-robot algorithm assumes a disjoint search-region partition
   always exists; on a graph with a single shared bottleneck corridor, it
   provably doesn't, and the algorithm's specified merge step silently
   drops one agent's observation of the shared edge.

A third observation runs through both: the paper's sole convergence
metric (Shannon entropy) measures how *confident* a belief map is, not
whether it is *correct*, and both gaps above are invisible to it.

Two further results are reported as tentative:

4. Under sensor-reliability misspecification, the Bayesian-network method
   keeps its accuracy advantage over the weighted-average baseline at every
   false-positive misestimate tested and across lethality bias from -35% to
   +10%; the advantage disappears only when lethality is assumed
   near-certain.
5. Keeping one independent probability per corridor (a factored belief)
   instead of the exact joint posterior costs little accuracy when hazards
   act on every pass, about five times more when a corridor's hazard acts
   once per round, and in both cases changes the planner's route choice in
   a fifth to a third of rounds.

**Changelog.** 5 Nov 2026: Experiments 1 and 4 originally reused the
initial belief for planning, so the agent repeated one route every round.
Fixed; all affected results regenerated. Experiment 5 added.

## What it looks like

![Single-agent round trip, animated](results/figures/sim_single_agent.svg)
![Two-agent bottleneck round, animated](results/figures/sim_bottleneck.svg)

Two representative simulated rounds, replayed as looping SVG animations
(play natively here on GitHub and in any browser -- no GIF, see
`report/echo_corridors_report.html` Section 3.3 for why). Static,
non-animated diagrams of both environments are also in `results/figures/`.

Read the full write-up, with all figures and numbers, in
[`report/echo_corridors_report.html`](report/echo_corridors_report.html)
(or the [PDF](report/echo_corridors_report.pdf) -- same content, printed to
Letter paper). A longer, more informally written companion version is also
kept at [`report/REPORT.md`](report/REPORT.md) for quick reading directly
on GitHub.

## Repository layout

```
src/
  environment.py      graph environments: random pruned-grid facility,
                       fixed bottleneck facility, hazard planting
  inference.py         three belief-update rules + derivation of the
                       corrected multi-visit closed form
  planner.py           greedy expected-information-gain round-trip planner
  metrics.py            entropy, Brier score, log loss, recall@k
  svgplot.py            dependency-free SVG line-chart writer

experiment1_revisits.py          Finding 1: revisit handling
experiment2_bottleneck.py        Finding 2: multi-agent overlap on a bridge
experiment4_misspecification.py  Finding 4: sensor-rate misspecification
experiment5_exact_joint.py       Finding 5: factored vs exact joint belief (needs numpy)
check_exp5_exact_vs_closed_form.py  sanity check for Experiment 5
run_all.py                       runs Experiments 1, 2, 4, 5; raw CSVs, audit, manifest
make_diagrams.py                  static environment topology figures
make_animation.py                 animated round-trip replays (SVG, not GIF -- see report)
make_print_figures.py             black-and-white figure set for the print report

results/               generated CSV/JSON summaries + SVG figures
results/raw/           per-trial raw results (one row per trial and method)
results/run_manifest.json  commit, versions, seeds, settings, output hashes
report/echo_corridors_report.html   the report (open in a browser)
report/echo_corridors_report.pdf    the same report, printed to Letter paper
report/REPORT.md                    longer, informally written companion version
```

## Running it

Experiments 1, 2 and 4 need only Python 3.9+ and the standard library.
Experiment 5 (and therefore `run_all.py`) also needs numpy
(`pip install numpy`).

```bash
python run_all.py               # Experiments 1, 2, 4, 5 + raw CSVs, audit, manifest
python make_print_figures.py    # black-and-white figures for the print report
python make_diagrams.py
python make_animation.py
```

Each experiment script can also be run on its own and writes its own
CSV/JSON summaries and SVG figures into `results/`. `run_all.py` also
writes per-trial results to `results/raw/`, checks every summary mean
against them, writes the paired differences quoted in the report to
`results/paired_differences.json`, runs the Experiment 5 sanity check, and
records the git commit, Python and numpy versions, seeds, settings and
every output file in `results/run_manifest.json`. Random seeds are fixed in
each script, so re-running reproduces the numbers in the report.

## Method being implemented

Hazard existence per corridor `Z`, per-corridor destruction `Delta`, and
the path-level sensor trigger `Theta` form a small Bayesian network. Given
a survived or triggered observation over a path, the posterior over hazard
existence for each corridor on that path is computed in closed form
(`src/inference.py`); path selection greedily maximizes expected
information gain, i.e., expected entropy reduction of the belief map,
matching the objective in the reference paper's Eq. (1). See
`report/echo_corridors_report.html` Section 2.3 for the full derivation,
including the multi-visit generalization this project adds.

## License

MIT. See `LICENSE`.
