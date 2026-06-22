"""Tests for W3 (genome metadata/typing) + W1 (Hyperband/Successive-Halving).

Run: python3 tests/test_scheduler.py  (light stack).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb  # noqa: F401  (registers the bank)
from pattern_brain import genome
from pattern_brain.search import DAGSearch, _skill
from pattern_brain.scheduler import SuccessiveHalving, Hyperband
from pattern_brain.leaderboard import Leaderboard

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _ar_data(T=160, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng.normal(scale=0.4)
    return x.reshape(-1, 1)


# ----------------------------------------------------------------- W3 genome
def test_genome_metadata():
    lib = genome.genome_library()
    check(len(lib) > 150, f"genome library should cover the whole bank, got {len(lib)}")
    # cost classes are sane
    check(genome.node_cost("drift_forecast") == "low", "a light forecaster should be low cost")
    if "lstm_forecaster" in lib:                          # torch present
        check(genome.node_cost("lstm_forecaster") == "high", "a torch node should be high cost")
    check(genome.node_cost("gaussian_hmm") in ("med", "low"), "hmm cost should be med/low")
    # roles
    check(genome.role("drift_forecast") == "predictor", "a forecaster is a predictor")
    check(genome.role("difference") == "feature", "a signal transform is a feature")
    print(f"  genome: {len(lib)} nodes carded; cost/role derived correctly")


def test_compatibility_predicate():
    # layer ordering respected; never chain out of a terminal
    check(genome.compatible("difference", "drift_forecast"), "feature→predictor should be allowed")
    check(not genome.compatible("drift_forecast", "difference"), "predictor→feature should be blocked (backwards)")
    # a terminal (decision/rl) cannot be upstream
    terms = [nt for nt in pb.all_node_types() if genome.role(nt) == "terminal"]
    if terms:
        check(not genome.compatible(terms[0], "drift_forecast"),
              "a terminal node must not feed another node")
    check(genome.compatible("unknown_a", "unknown_b"), "unknown nodes default to allowed (no silent exclude)")
    print(f"  compatibility: layer-order + no-terminal-upstream enforced ({len(terms)} terminals)")


# ----------------------------------------------------------------- W1 search
def _search(seed=0):
    X = _ar_data(seed=seed)
    lb = Leaderboard(":memory:")
    pool = ["drift_forecast", "theta_forecast", "naive_mean_forecast",
            "autoregressive_forecast", "ewma_forecast", "seasonal_naive_forecast"]
    pool = [p for p in pool if p in pb.all_node_types()] or None
    return DAGSearch(X, leaderboard=lb, base_pool=pool, min_train=40, n_samples=40, seed=seed)


def test_budgeted_scoring_cheaper():
    """A small budget evaluates fewer forecast steps than the full budget."""
    s = _search()
    spec = s._random_spec()
    cheap = s.score_spec_budgeted(spec, frac=0.4, n_samples=12)
    full = s.score_spec_budgeted(spec, frac=1.0, n_samples=40)
    check(cheap.get("n", 0) <= full.get("n", 0),
          f"cheap rung should score on fewer steps ({cheap.get('n')} vs {full.get('n')})")
    check(cheap.get("n", 0) >= 1 and full.get("n", 0) >= 1, "both budgets should produce forecasts")
    print(f"  budgeted scoring: cheap n={cheap.get('n')} <= full n={full.get('n')}")


def test_successive_halving_reduces_and_picks():
    s = _search()
    sh = SuccessiveHalving(s, eta=3, seed=1)
    rep = sh.run(n_configs=9)
    rungs = rep["rungs"]
    check(len(rungs) == 3, f"expected a 3-rung ladder, got {len(rungs)}")
    # candidates shrink down the ladder (9 → ~3 → final), by ~1/eta each non-final rung
    check(rungs[0]["evaluated"] >= rungs[1]["evaluated"] >= 1,
          f"rungs should narrow: {[r['evaluated'] for r in rungs]}")
    check(rungs[1]["evaluated"] <= max(1, -(-rungs[0]["evaluated"] // 3)),
          "rung 1 should keep ~1/eta of rung 0")
    check(rep["best_spec"] is not None, "SH should return a best spec")
    # cost saving: total evals < evaluating ALL candidates at the full rung
    n0 = rungs[0]["evaluated"]
    check(rep["total_evaluations"] < n0 * len(rungs),
          f"SH should cost less than full-grid ({rep['total_evaluations']} vs {n0*len(rungs)})")
    print(f"  successive-halving: ladder {[r['evaluated'] for r in rungs]}, "
          f"{rep['total_evaluations']} evals, best picked")


def test_hyperband_runs_and_records():
    s = _search(seed=2)
    rep = Hyperband(s, eta=3, seed=2).run(max_configs=9, n_brackets=2)
    check(len(rep["brackets"]) == 2, "hyperband should run 2 brackets")
    check(rep["total_evaluations"] > 0, "hyperband should evaluate candidates")
    # the final rungs record to the leaderboard → best() is populated
    best = rep["best"]
    check(best is not None and "crps_skill" in best, "hyperband best should be on the leaderboard")
    print(f"  hyperband: 2 brackets, {rep['total_evaluations']} evals, best skill="
          f"{round(float(best.get('crps_skill') or 0),4)}")


def main():
    print("=" * 70)
    print("Pattern Brain — W3 genome + W1 Hyperband/Successive-Halving: tests")
    print("=" * 70)
    test_genome_metadata()
    test_compatibility_predicate()
    test_budgeted_scoring_cheaper()
    test_successive_halving_reduces_and_picks()
    test_hyperband_runs_and_records()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
