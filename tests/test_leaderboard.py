"""Tests for Phase 8 slice 6 — the Leaderboard (PLAN.md §12).

Run: python3 tests/test_leaderboard.py  (light stack — stdlib sqlite + numpy).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.leaderboard import Leaderboard, _PROJECT_ROOT
from pattern_brain.network import DAGSpec

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _score(skill, crps, calib=0.05, dsr=None, n=50):
    return {"n": n, "mean_crps": crps, "baseline_crps": crps + skill,
            "crps_skill": skill, "calib_ks": calib, "calibrated": calib < 0.1}


def test_record_and_rank():
    lb = Leaderboard(":memory:")
    specs = {
        "weak": DAGSpec([["drift_forecast"]], "mixture"),
        "mid": DAGSpec([["drift_forecast", "theta_forecast"]], "mixture"),
        "best": DAGSpec([["drift_forecast", "theta_forecast"]], "stacked"),
    }
    lb.record(specs["weak"], _score(0.01, 0.40))
    lb.record(specs["mid"], _score(0.10, 0.30))
    lb.record(specs["best"], _score(0.25, 0.22))
    check(lb.count() == 3, f"expected 3 rows, got {lb.count()}")
    top = lb.top(3, metric="crps_skill")
    check([r["crps_skill"] for r in top] == [0.25, 0.10, 0.01], "skill ranking wrong")
    top_crps = lb.top(3, metric="mean_crps")
    check(top_crps[0]["mean_crps"] == 0.22, "mean_crps ranking should be ascending (lower better)")
    check(lb.best()["combiner"] == "stacked", "best should be the stacked DAG")
    print("  record + rank by skill (desc) and crps (asc); best = stacked DAG")


def test_dedupe_keeps_best_per_spec():
    lb = Leaderboard(":memory:")
    spec = DAGSpec([["drift_forecast"]], "mixture")
    lb.record(spec, _score(0.05, 0.35))
    lb.record(spec, _score(0.20, 0.25))                 # same spec, better score (re-run)
    top = lb.top(10, dedupe=True)
    check(len(top) == 1 and top[0]["crps_skill"] == 0.20, "dedupe should keep the best per spec")
    check(len(lb.top(10, dedupe=False)) == 2, "non-dedupe should keep both rows")
    print("  dedupe keeps the best row per distinct spec")


def test_dsr_ranking_and_summary():
    lb = Leaderboard(":memory:")
    a = DAGSpec([["drift_forecast"]], "mixture")
    b = DAGSpec([["theta_forecast"]], "stacked")
    lb.record(a, _score(0.30, 0.20), dsr=0.40)          # high skill but NOT deflated-significant
    lb.record(b, _score(0.12, 0.28), dsr=0.97)          # lower skill but survives deflation
    top_dsr = lb.top(5, metric="dsr")
    check(top_dsr[0]["combiner"] == "stacked", "DSR ranking must prefer the deflated-significant DAG")
    s = lb.summary()
    check(s["n_runs"] == 2 and s["n_beat_baseline"] == 2 and s["n_dsr_significant"] == 1,
          f"summary wrong: {s}")
    print(f"  DSR ranking favors the deflated-significant DAG; summary={s['n_dsr_significant']} significant")


def test_persistence_and_rule1_guard():
    db = os.path.join(_PROJECT_ROOT, "data", "leaderboard_test.sqlite")
    if os.path.exists(db):
        os.remove(db)
    lb = Leaderboard(db)
    rid = lb.record(DAGSpec([["drift_forecast"]], "mixture"), _score(0.1, 0.3))
    lb.close()
    lb2 = Leaderboard(db)                                # reopen → row persisted
    check(lb2.get(rid) is not None and lb2.count() == 1, "row did not persist across reopen")
    check(db.startswith(_PROJECT_ROOT + os.sep), "db must be inside the project folder")
    lb2.close()
    os.remove(db)
    try:
        Leaderboard("/tmp/pb_outside_leaderboard.sqlite")
        check(False, "Rule 1: outside-folder db should be refused")
    except ValueError:
        pass
    print("  persists across reopen; Rule-1 guard refuses outside-folder db")


def test_end_to_end_with_real_dag():
    """A real StackedDAG.score() row records and ranks correctly."""
    import numpy as np
    from pattern_brain.network import StackedDAG
    rng = np.random.default_rng(0)
    a = np.zeros(160)
    for t in range(1, 160):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    X = a.reshape(-1, 1)
    lb = Leaderboard(":memory:")
    spec = DAGSpec([["drift_forecast", "theta_forecast"]], "stacked")
    score = StackedDAG(spec, min_train=40, n_samples=120).score(X)
    rid = lb.record(spec, score, dataset_id="ar1")
    row = lb.get(rid)
    check(row["n_eval"] == score["n"] and abs(row["crps_skill"] - score["crps_skill"]) < 1e-9,
          "real DAG score did not record faithfully")
    print(f"  end-to-end: real DAG (skill={score['crps_skill']:.3f}) recorded + retrievable")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 6: Leaderboard: tests")
    print("=" * 70)
    test_record_and_rank()
    test_dedupe_keeps_best_per_spec()
    test_dsr_ranking_and_summary()
    test_persistence_and_rule1_guard()
    test_end_to_end_with_real_dag()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
