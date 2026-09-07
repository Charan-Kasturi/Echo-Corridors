"""
inference.py
=============
Belief-map update rules for a single path-based sensor observation.

Implements three update rules over a shared representation: `belief` is a
dict edge_id -> P(hazard = 1), and a traversed path is an ordered list of
edge ids (repeats allowed).

  1. weighted_average_update  -- the pre-existing method of Otte & Sofge
     (2021) [18] and its multi-robot extension [26], Eq. (2) of the
     reference paper: mutually-exclusive single-culprit histories, combined
     by relative likelihood.

  2. bn_exact_update -- our closed-form generalisation of the reference
     paper's Bayesian-network method (Section 4.2, Eqs. (5)-(7)) that
     correctly handles a path revisiting the same edge/cell multiple times,
     derived below.

  3. bn_literal_revisit_update -- a literal, best-effort implementation of
     the paper's own stated (but not formalised) prescription for repeat
     visits: Remark 1 says only that "the new likelihood P(Theta | Delta) is
     calculated by summing the probability of each instance," without
     giving the resulting expression or showing it is consistent with the
     rest of the Bayesian-network derivation. We take that sentence at face
     value -- an edge visited n times contributes n independent additive
     chances of triggering the sensor -- and show, in experiment 1, that it
     diverges materially from the exact closed form (2).

Derivation of the exact closed form (method 2)
-----------------------------------------------
Let a path visit distinct edges e = 1..m, with edge e visited n_e times.
Conditioned on hazard state z_e in {0,1}, each individual visit to e is an
independent Bernoulli survival trial with probability
    a_e = P(survive one visit | z_e = 1) = 1 - L        (lethality L)
    b_e = P(survive one visit | z_e = 0) = 1 - fp        (false-positive rate)
so surviving *all* n_e visits to e has probability a_e^{n_e} or b_e^{n_e}.
Because the agent is destroyed on first failure and simply does not return,
the path-based sensor observation Theta is a deterministic function of
whether *any* visit failed:

    P(Theta = 0 | z) = prod_e  (a_e^{n_e} if z_e=1 else b_e^{n_e})

Because different edges are hazard-independent, marginalising over the
unobserved z gives, per edge, a marginal single-edge "survive all my visits"
probability s_e = q_e * a_e^{n_e} + (1-q_e) * b_e^{n_e} (q_e = prior belief),
and P(Theta=0) = prod_e s_e. Standard Bayes then gives, for Theta = 0:

    P(z_e=1 | Theta=0) = q_e * a_e^{n_e} / s_e                         (*)

which is the direct multi-visit generalisation of the paper's Eq. (7), and,
for Theta = 1 (destroyed somewhere on the path):

    P(z_e=1 | Theta=1) = q_e * (1 - a_e^{n_e} * S / s_e) / (1 - S)     (**)

where S = prod_i s_i over all distinct edges on the path. Notably, (*) and
(**) depend only on the *count* of visits per edge, not their order along
the path -- because "at least one failure occurred somewhere" is a symmetric
function of the underlying independent per-visit outcomes. When n_e = 1 for
every edge (no revisits), (*) and (**) reduce exactly to the paper's Eqs.
(7) and (5)/(6).
"""
from __future__ import annotations

from collections import Counter


def _visit_counts(path):
    return Counter(path)


def path_survival_probability(belief: dict, path: list,
                               lethality: float, false_positive: float) -> float:
    """P(Theta = 0) under the current belief -- the probability that a
    round-trip along `path` (with repeats) is survived entirely."""
    a = 1.0 - lethality
    b = 1.0 - false_positive
    S = 1.0
    for e, n_e in _visit_counts(path).items():
        q = belief[e]
        S *= q * (a ** n_e) + (1 - q) * (b ** n_e)
    return S


def bn_exact_update(belief: dict, path: list, destroyed: bool,
                     lethality: float, false_positive: float) -> dict:
    """Closed-form Bayesian-network update, correct under arbitrary revisits.
    Returns a NEW belief dict (edges not on the path are copied unchanged).
    """
    new_belief = dict(belief)
    counts = _visit_counts(path)
    a = 1.0 - lethality
    b = 1.0 - false_positive

    s = {}
    for e, n_e in counts.items():
        q = belief[e]
        a_n = a ** n_e
        b_n = b ** n_e
        s[e] = q * a_n + (1 - q) * b_n

    S = 1.0
    for v in s.values():
        S *= v

    for e, n_e in counts.items():
        q = belief[e]
        a_n = a ** n_e
        if not destroyed:
            new_belief[e] = q * a_n / s[e] if s[e] > 0 else q
        else:
            denom = 1.0 - S
            if denom <= 1e-12:
                new_belief[e] = q  # degenerate: Theta=1 was essentially certain regardless
            else:
                new_belief[e] = q * (1.0 - a_n * (S / s[e])) / denom
    return new_belief


def bn_literal_revisit_update(belief: dict, path: list, destroyed: bool,
                               lethality: float, false_positive: float) -> dict:
    """Best-effort literal reading of the paper's Remark 1: a cell visited
    n times contributes an *additive* ("summed") destruction chance across
    instances, capped at 1, rather than the correct compounding of survival
    probabilities. The adjusted per-edge destruction probability is then
    plugged into the ordinary single-visit Bayesian-network update (the
    paper's Eqs. (4)-(7), i.e. this function with every n_e forced to 1).
    """
    new_belief = dict(belief)
    counts = _visit_counts(path)

    eff_L = {}
    eff_fp = {}
    for e, n_e in counts.items():
        eff_L[e] = min(1.0, n_e * lethality)
        eff_fp[e] = min(1.0, n_e * false_positive)

    s = {}
    for e in counts:
        q = belief[e]
        a = 1.0 - eff_L[e]
        b = 1.0 - eff_fp[e]
        s[e] = q * a + (1 - q) * b

    S = 1.0
    for v in s.values():
        S *= v

    for e in counts:
        q = belief[e]
        a = 1.0 - eff_L[e]
        if not destroyed:
            new_belief[e] = q * a / s[e] if s[e] > 0 else q
        else:
            denom = 1.0 - S
            if denom <= 1e-12:
                new_belief[e] = q
            else:
                new_belief[e] = q * (1.0 - a * (S / s[e])) / denom
    return new_belief


def weighted_average_update(belief: dict, path: list, destroyed: bool,
                             lethality: float, false_positive: float) -> dict:
    """Reference [18]/[26] weighted-average update, Eq. (2): mutually
    exclusive single-culprit histories (destruction happened at position k,
    for k = 1..len(path)) combined by their relative likelihood. Positions
    are taken literally as written along the path (this is exactly how a
    practitioner following [18] would apply it to a path that revisits
    cells -- the method was never extended for that case either)."""
    L, fp = lethality, false_positive
    a = 1.0 - L
    b = 1.0 - fp

    if not destroyed:
        # independent per-edge survival update (order-independent, same as
        # sequential application of Eq. (7) once per visit).
        new_belief = dict(belief)
        counts = _visit_counts(path)
        for e, n_e in counts.items():
            q = new_belief[e]
            a_n = a ** n_e
            b_n = b ** n_e
            s = q * a_n + (1 - q) * b_n
            new_belief[e] = q * a_n / s if s > 0 else q
        return new_belief

    # destroyed: build the ell mutually exclusive "destroyed at position k"
    # scenario weights, each scenario updating the path's cells consistently
    # with that history, then combine by relative likelihood.
    ell = len(path)
    surv_prob_at = []  # P(survive position j) under its own edge's CURRENT prior
    for e in path:
        q = belief[e]
        surv_prob_at.append(q * a + (1 - q) * b)

    weights = []
    prefix = 1.0
    for k in range(ell):
        e_k = path[k]
        q_k = belief[e_k]
        destroy_k = q_k * L + (1 - q_k) * fp
        weights.append(prefix * destroy_k)
        prefix *= surv_prob_at[k]
    total_w = sum(weights)
    if total_w <= 1e-12:
        return dict(belief)
    weights = [w / total_w for w in weights]

    # accumulate weighted posterior per distinct edge
    acc = {e: 0.0 for e in set(path)}
    for k in range(ell):
        e_k = path[k]
        q_k = belief[e_k]
        destroy_k = q_k * L + (1 - q_k) * fp
        post_k = (q_k * L) / destroy_k if destroy_k > 0 else q_k
        # for scenario k: edge at position k gets `post_k`; every OTHER
        # distinct edge on the path keeps its prior (unaffected in that
        # hypothetical history) unless it also equals e_k at another index.
        for e in acc:
            if e == e_k:
                acc[e] += weights[k] * post_k
            else:
                acc[e] += weights[k] * belief[e]

    new_belief = dict(belief)
    for e, v in acc.items():
        new_belief[e] = v
    return new_belief
