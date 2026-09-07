"""
metrics.py
==========
Scoring functions for hazard belief maps.

The reference paper evaluates convergence using only Shannon entropy of the
belief map and a count of destroyed agents (Section 6). Entropy measures how
*confident* a belief map is, not how *correct* it is: a belief map that has
converged to a confident but wrong answer has low entropy and looks, by the
paper's own metric, like a success. We add proper scoring-rule metrics
(Brier score, log loss) plus detection precision/recall against the actual
planted ground truth, so that "converged" and "correct" can be told apart.
"""
from __future__ import annotations

import math

EPS = 1e-9


def clamp(p: float) -> float:
    return min(1.0 - EPS, max(EPS, p))


def binary_entropy(p: float) -> float:
    p = clamp(p)
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def map_entropy(belief: dict) -> float:
    """Total Shannon entropy (nats) of the map, summed over edges -- valid
    because hazard existence is assumed independent across edges/cells
    (Section 3 of the reference paper)."""
    return sum(binary_entropy(p) for p in belief.values())


def brier_score(belief: dict, truth: dict) -> float:
    """Mean squared error between believed hazard probability and ground
    truth, averaged over all edges. Lower is better; 0 is perfect."""
    keys = list(belief.keys())
    if not keys:
        return 0.0
    return sum((belief[k] - truth[k]) ** 2 for k in keys) / len(keys)


def log_loss(belief: dict, truth: dict) -> float:
    keys = list(belief.keys())
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        p = clamp(belief[k])
        y = truth[k]
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(keys)


def precision_recall_at_k(belief: dict, truth: dict, k: int) -> tuple:
    """Precision/recall of the top-k highest-belief edges against the true
    hazard set (k defaults, in callers, to the true number of hazards)."""
    if k <= 0:
        return (1.0, 1.0)
    ranked = sorted(belief.keys(), key=lambda e: belief[e], reverse=True)[:k]
    true_positives = sum(1 for e in ranked if truth.get(e, 0) == 1)
    n_true = sum(1 for v in truth.values() if v == 1)
    precision = true_positives / k
    recall = true_positives / n_true if n_true > 0 else 1.0
    return (precision, recall)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)
