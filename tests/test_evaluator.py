"""Contract + behavior tests for the shared 5-layer Evaluator (final phase).

Run: python3 tests/test_evaluator.py
Encodes the §4 spec per Rule 25: the hard gate Features 1/2/3 use. Each layer is
checked on synthetic outcome series where the right answer is known, plus the
end-to-end candidate evaluation + population admission. Domain-agnostic (Rule 23):
everything operates on generic outcome series, no candle/order-book references.
"""
from __future__ import annotations

import os
import sys
import json
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.evaluator import (
    purged_walk_forward_splits, sharpe, max_drawdown, stability,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, min_track_record_length,
    expected_max_sharpe, discounted_ucb_score, sliding_window_ucb_score,
    nondominated_sort, pareto_front, cscv_pbo, anchor_check,
    Evaluator, ObjectiveSpec, DEFAULT_OBJECTIVES,
)

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


# ------------------------------------------------------------- Layer 0
def test_purged_walk_forward_no_leakage():
    """Every fold's train indices are strictly before its test indices, with an
    embargo gap — no future leaks into the past."""
    splits = purged_walk_forward_splits(200, n_splits=5, embargo=0.05)
    check(len(splits) >= 3, f"too few folds: {len(splits)}")
    gap = int(round(0.05 * 200))
    for tr, te in splits:
        check(tr.max() < te.min(), f"train overlaps/after test: {tr.max()} !< {te.min()}")
        check(te.min() - tr.max() - 1 >= gap - 1, "embargo gap not honored")
        check(len(set(tr) & set(te)) == 0, "train/test index overlap")
    print(f"  Layer0: {len(splits)} leakage-safe folds, embargo honored")


# ------------------------------------------------------------- Layer 1
def test_sharpe_and_drawdown_known():
    # constant positive series: high sharpe, zero drawdown
    pos = np.full(100, 0.1)
    check(sharpe(pos) > 1e6 or sharpe(pos) > 100, "constant positive should have huge sharpe")
    check(max_drawdown(pos) == 0.0, "monotone-up curve should have zero drawdown")
    # a dip creates drawdown
    series = np.array([1.0, 1.0, -3.0, 1.0])
    check(max_drawdown(series) > 0, "a negative outcome should create drawdown")
    # stability: a steady ramp is near 1
    check(stability(np.full(50, 0.2)) > 0.99, "steady accrual should be very stable")
    print("  Layer1: sharpe/drawdown/stability behave on known series")


def test_psr_dsr_ordering():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.15, 1.0, 500)   # clear positive edge, long sample
    weak = rng.normal(0.01, 1.0, 30)      # tiny edge, short sample
    check(probabilistic_sharpe_ratio(strong) > probabilistic_sharpe_ratio(weak),
          "PSR should rank the strong, long series above the weak, short one")
    check(0.0 <= probabilistic_sharpe_ratio(strong) <= 1.0, "PSR out of [0,1]")
    # deflation: more trials -> harder benchmark -> DSR <= PSR
    psr = probabilistic_sharpe_ratio(strong)
    dsr_many = deflated_sharpe_ratio(strong, n_trials=100, var_sr=1.0)
    check(dsr_many <= psr + 1e-9, f"DSR({dsr_many:.3f}) should be <= PSR({psr:.3f}) under many trials")
    check(expected_max_sharpe(100, 1.0) > expected_max_sharpe(2, 1.0),
          "expected max Sharpe should grow with trials")
    print(f"  Layer1: PSR strong>weak; DSR deflates under selection (PSR {psr:.3f} -> DSR {dsr_many:.3f})")


def test_min_trl():
    rng = np.random.default_rng(1)
    good = rng.normal(0.2, 1.0, 400)
    trl = min_track_record_length(good)
    check(math.isfinite(trl) and trl > 0, f"minTRL should be a positive finite number: {trl}")
    # a series with no edge -> infinite minTRL
    flat = rng.normal(0.0, 1.0, 400)
    check(min_track_record_length(flat, sr_benchmark=0.5) == float("inf"),
          "no-edge series vs high benchmark should need infinite track record")
    print(f"  Layer1: minTRL finite for an edge ({trl:.0f} obs), inf without")


# ------------------------------------------------------------- Layer 2
def test_ucb_recency_and_exploration():
    decayed = np.concatenate([np.full(50, 1.0), np.full(50, -1.0)])  # recently bad
    improving = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])  # recently good
    check(discounted_ucb_score(improving) > discounted_ucb_score(decayed),
          "discounted UCB should favor the recently-improving stream (pathway death)")
    # exploration bonus: a short stream gets a larger bonus than a long one at same mean
    short = np.full(5, 0.0)
    long = np.full(500, 0.0)
    check(sliding_window_ucb_score(short, window=1000) > sliding_window_ucb_score(long, window=1000),
          "less-seen candidate should get a larger exploration bonus (curiosity)")
    print("  Layer2: discounted-UCB recency + exploration bonus behave")


# ------------------------------------------------------------- Layer 3
def test_pareto_and_pbo():
    # objectives (maximize): a dominates b; c is on the front (trades off)
    obj = np.array([[1.0, 1.0],   # 0
                    [0.5, 0.5],   # 1 dominated by 0
                    [0.2, 2.0],   # 2 non-dominated (high obj2)
                    [2.0, 0.1]])  # 3 non-dominated (high obj1)
    front = set(pareto_front(obj))
    check(1 not in front, "dominated candidate leaked into the Pareto front")
    check({0, 2, 3} <= front, f"front missing non-dominated candidates: {front}")
    fronts = nondominated_sort(obj)
    check([1] in [f for f in fronts], "candidate 1 should be in a later front")

    # PBO: random strategies (no real edge) -> high overfit probability
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (240, 6))
    pbo_noise = cscv_pbo(noise, n_blocks=8)
    # one genuinely good strategy added -> PBO should drop
    edge = noise.copy(); edge[:, 0] += 0.6
    pbo_edge = cscv_pbo(edge, n_blocks=8)
    check(0.0 <= pbo_noise <= 1.0, "PBO out of [0,1]")
    check(pbo_edge <= pbo_noise + 1e-9,
          f"a genuine-edge strategy should not raise PBO ({pbo_edge:.2f} vs {pbo_noise:.2f})")
    print(f"  Layer3: Pareto sort correct; PBO noise={pbo_noise:.2f} >= edge={pbo_edge:.2f}")


# ------------------------------------------------------------- Layer 4
def test_anchor_check():
    check(anchor_check([[0.1, 0.2], [0.05, 0.1]], min_mean=0.0), "sane anchors rejected")
    check(not anchor_check([[0.1, 0.2], [-0.5, -0.4]], min_mean=0.0),
          "a failing anchor episode should reject the candidate")
    print("  Layer4: anchor gate accepts sane, rejects failing episodes")


# --------------------------------------------------- end-to-end Evaluator
def _edge_candidate(train, test):
    """A toy candidate with a real edge: outcome = mean-reverting score with a
    positive bias (generic, no domain meaning)."""
    rng = np.random.default_rng(int(abs(test.sum())) % 10000)
    return 0.12 + rng.normal(0, 1.0, test.shape[0])


def _noise_candidate(train, test):
    rng = np.random.default_rng(int(abs(test.sum())) % 10000 + 1)
    return rng.normal(0, 1.0, test.shape[0])


def test_evaluate_candidate_end_to_end():
    X = np.cumsum(np.random.default_rng(0).normal(size=(600, 2)), axis=0)
    ev = Evaluator(n_splits=6, embargo=0.02, psr_threshold=0.9, dsr_threshold=0.9)
    rep_edge = ev.evaluate_candidate(_edge_candidate, X, n_trials=1)
    rep_noise = ev.evaluate_candidate(_noise_candidate, X, n_trials=1)
    check(rep_edge.n_obs > 0, "no outcomes produced for edge candidate")
    check(rep_edge.sharpe > rep_noise.sharpe, "edge candidate should out-Sharpe noise")
    check(rep_edge.passed, f"edge candidate should pass Layers 1-2: {rep_edge.reasons}")
    check(not rep_noise.passed, "pure-noise candidate should NOT pass significance")
    json.dumps(rep_edge.to_dict())  # report must serialize (dashboard/audit)
    print(f"  e2e: edge passed (PSR {rep_edge.psr:.3f}), noise rejected ({rep_noise.reasons})")


def test_select_admission_gate():
    """Population gate: only Pareto-optimal, significant, anchor-surviving, non-
    overfit candidates are admitted."""
    X = np.cumsum(np.random.default_rng(2).normal(size=(600, 2)), axis=0)
    ev = Evaluator(n_splits=6, psr_threshold=0.9, dsr_threshold=0.9)
    reports = [ev.evaluate_candidate(_edge_candidate, X),
               ev.evaluate_candidate(_noise_candidate, X)]
    # per-strategy per-period matrix for PBO (edge col + noise col)
    rng = np.random.default_rng(5)
    perf = np.column_stack([0.12 + rng.normal(0, 1, 240), rng.normal(0, 1, 240)])
    anchors = [[[0.1, 0.2, 0.1]], [[-0.3, -0.2]]]  # edge survives anchor, noise fails
    res = ev.select(reports, perf_matrix=perf, anchors=anchors)
    check(0 in res["pareto_front"], "edge candidate should be on the Pareto front")
    check(res["admitted"] == [0] or res["admitted"] == [],
          f"only the edge candidate may be admitted, got {res['admitted']}")
    check(1 not in res["admitted"], "noise candidate must never be admitted")
    json.dumps(res)
    print(f"  select: admitted={res['admitted']} pbo={res['pbo']} front={res['pareto_front']}")


def test_pluggable_objectives():
    """§0 tension: the objective set is pluggable (domain-agnostic core)."""
    spec = ObjectiveSpec((("mean", lambda x: float(np.mean(x)) if len(x) else 0.0),))
    ev = Evaluator(objectives=spec)
    rep = ev.evaluate_outcomes(np.full(50, 0.1))
    check(list(rep.objectives.keys()) == ["mean"], "custom ObjectiveSpec not applied")
    check(set(o[0] for o in DEFAULT_OBJECTIVES.metrics) ==
          {"mean", "sharpe", "neg_drawdown", "stability"}, "default objectives drifted")
    print("  objectives: pluggable ObjectiveSpec works (resolves §0 tension)")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "evaluator.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in evaluator.py: {hits}")
    print("  domain-independence: evaluator has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — 5-layer Evaluator (final phase): contract tests")
    print("=" * 70)
    test_purged_walk_forward_no_leakage()
    test_sharpe_and_drawdown_known()
    test_psr_dsr_ordering()
    test_min_trl()
    test_ucb_recency_and_exploration()
    test_pareto_and_pbo()
    test_anchor_check()
    test_evaluate_candidate_end_to_end()
    test_select_admission_gate()
    test_pluggable_objectives()
    test_domain_independence()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
