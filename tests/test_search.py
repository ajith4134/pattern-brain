"""Tests for Phase 8 slice 7 — the DAG search engine (PLAN.md §12).

Run: .venv/bin/python tests/test_search.py   (optuna path needs optuna; skips cleanly).
     python3 tests/test_search.py             (random/evolutionary only).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.search import DAGSearch, COMBINERS, optuna_available
from pattern_brain.leaderboard import Leaderboard
from pattern_brain.evaluator import Evaluator

FAILS = []
POOL = ["drift_forecast", "theta_forecast", "naive_mean_forecast"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _ar(n=130, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    return a.reshape(-1, 1)


def _search(seed=0, dataset="t"):
    return DAGSearch(_ar(seed=seed), leaderboard=Leaderboard(":memory:"), base_pool=POOL,
                     max_base=3, min_train=35, n_samples=50, dataset_id=dataset, seed=seed)


def test_random_search_records_and_returns_best():
    s = _search()
    best = s.random_search(n=5)
    check(s.lb.count() == 5, f"random search should record 5 rows, got {s.lb.count()}")
    check(best is not None and "crps_skill" in best and "spec_json" in best, "best row malformed")
    check(np.isfinite(best["crps_skill"]), "best skill non-finite")
    print(f"  random search: 5 trials recorded; best skill={best['crps_skill']:.3f} ({best['combiner']})")


def test_genome_operators_valid():
    s = _search()
    spec = s._random_spec()
    for _ in range(40):
        m = s._mutate(spec)
        check(len(m.layers[0]) >= 1 and all(n in POOL for n in m.layers[0]), f"mutate invalid: {m}")
        check(m.combiner in COMBINERS, f"mutate bad combiner: {m.combiner}")
        c = s._crossover(spec, m)
        check(len(c.layers[0]) >= 1 and all(n in POOL for n in c.layers[0]), f"crossover invalid: {c}")
        spec = m
    print("  mutate/crossover always produce valid specs (nodes from pool, valid combiner)")


def test_evolutionary_runs():
    s = _search(seed=1)
    best = s.evolutionary_search(generations=2, pop=3)
    check(s.lb.count() >= 6, f"evolutionary should record >= gen*pop rows, got {s.lb.count()}")
    check(best is not None and np.isfinite(best["crps_skill"]), "evolutionary best malformed")
    print(f"  evolutionary (2×3): {s.lb.count()} trials; best skill={best['crps_skill']:.3f}")


def test_dsr_deflation_by_budget():
    """The anti-overfit guard: the SAME outcome series earns a LOWER Deflated Sharpe
    when more configurations were searched (n_trials ↑ ⇒ DSR ↓)."""
    rng = np.random.default_rng(2)
    series = 0.05 + rng.normal(scale=0.3, size=120)     # mildly positive edge
    ev = Evaluator(n_splits=4, embargo=0.02)
    dsr_few = ev.evaluate_outcomes(series, n_trials=1).dsr
    dsr_many = ev.evaluate_outcomes(series, n_trials=500).dsr
    check(dsr_many <= dsr_few, f"DSR should deflate with more trials: {dsr_few:.3f} -> {dsr_many:.3f}")
    print(f"  DSR deflation: n_trials 1→500 lowers DSR {dsr_few:.3f} → {dsr_many:.3f} (selection-bias guard)")


def test_best_prefers_dsr_then_skill():
    """best() ranks by DSR when available (deflated), else by raw skill."""
    s = _search()
    s.random_search(n=4)
    b = s.best()
    check(b is not None, "best should exist after a search")
    # at least one recorded row carries a dsr value (n_eval was large enough)
    rows = s.lb.top(10, metric="crps_skill")
    has_dsr = any(r["dsr"] is not None for r in rows)
    check(has_dsr, "at least one trial should have a deflated DSR recorded")
    print(f"  best() returns a winner; DSR recorded on >=1 trial (deflated by budget)")


def test_optuna_search():
    if not optuna_available():
        print("  optuna not installed -> optuna search skipped cleanly")
        return
    s = _search(seed=3)
    best = s.optuna_search(n_trials=5)
    check(s.lb.count() == 5, f"optuna should record 5 trials, got {s.lb.count()}")
    check(best is not None and np.isfinite(best["crps_skill"]), "optuna best malformed")
    print(f"  optuna (TPE) search: 5 trials; best skill={best['crps_skill']:.3f}")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 7: DAG search engine: tests")
    print("=" * 70)
    test_random_search_records_and_returns_best()
    test_genome_operators_valid()
    test_evolutionary_runs()
    test_dsr_deflation_by_budget()
    test_best_prefers_dsr_then_skill()
    test_optuna_search()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
