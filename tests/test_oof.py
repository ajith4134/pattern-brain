"""Tests for Phase 8 slice 4 — Belief→feature projection + OOF harness (PLAN.md §12).

Run: python3 tests/test_oof.py  (light stack — numpy/scipy/sklearn).
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import create
from pattern_brain import belief_features as BF
from pattern_brain import oof as OOF

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_belief_to_vector_fixed_width():
    x = np.cumsum(np.random.default_rng(0).normal(size=120)).reshape(-1, 1)
    for nt in ["naive_mean_forecast", "isolation_forest", "ridge_regression",
               "gaussian_hmm", "hurst_exponent"]:
        b = create(nt).predict(x)
        v = BF.belief_to_vector(b)
        check(v.shape == (BF.BELIEF_VECTOR_WIDTH,), f"{nt} vector width {v.shape} != 2")
        check(np.all(np.isfinite(v)), f"{nt} non-finite projection {v}")
    # forecast primary scalar == its point estimate
    b = create("naive_mean_forecast").predict(x)
    check(abs(BF.belief_to_vector(b)[0] - BF.belief_to_point(b)) < 1e-9,
          "forecast vector primary should equal the point estimate")
    print("  belief_to_vector: fixed width 2, finite, forecast primary == point")


def test_projection_completeness():
    info = BF.assert_projection_completeness()
    check(info["complete"], "projection completeness must hold (fallback guarantees it)")
    for t in ["forecast", "anomaly", "equation", "regime", "signal"]:
        check(t in info["dedicated"], f"core type {t} should have a dedicated projector")
    print(f"  completeness: {len(info['dedicated'])} dedicated, "
          f"fallback-only={info['fallback_only']}")


def test_belief_to_samples_uses_node_uncertainty():
    x = np.cumsum(np.random.default_rng(1).normal(size=150)).reshape(-1, 1)
    rng = np.random.default_rng(2)
    # quantile forecaster carries a quantile dict -> samples span it
    q = create("quantile_forecaster").predict(x)
    sq = BF.belief_to_samples(q, x[:, 0], 500, rng)
    check(sq.shape == (500,) and np.std(sq) > 0, "quantile samples should be a spread")
    # point-only forecaster -> falls back to point ± past volatility (still a spread)
    pt = create("naive_mean_forecast").predict(x)
    sp = BF.belief_to_samples(pt, x[:, 0], 500, rng)
    check(sp.shape == (500,) and np.std(sp) > 0, "point-only fallback should still spread")
    print(f"  belief_to_samples: quantile std={np.std(sq):.3f}, fallback std={np.std(sp):.3f}")


def test_oof_runs_and_is_leakage_safe():
    """The OOF point forecast at index t must not change when the FUTURE (>t) is
    perturbed — predict(X[:t]) sees only the past."""
    rng = np.random.default_rng(3)
    X = np.cumsum(rng.normal(size=200)).reshape(-1, 1)
    h = OOF.OOFHarness(min_train=40, stride=5, n_samples=100, seed=0)
    r1 = h.node_oof("drift_forecast", X)
    check(r1.index.size > 0 and r1.samples.shape == (r1.index.size, 100),
          f"OOF shapes wrong: idx {r1.index.size}, samples {r1.samples.shape}")
    check(r1.index.max() < len(X), "OOF must not evaluate beyond the series")
    X2 = X.copy(); X2[150:] += 40.0
    r2 = h.node_oof("drift_forecast", X2)
    early = r1.index < 145
    check(np.allclose(r1.point[early], r2.point[early], atol=0, rtol=0),
          "LOOK-AHEAD: OOF predictions before t changed when the future was perturbed")
    print(f"  OOF: {r1.index.size} causal one-step forecasts; future perturbation "
          f"leaves earlier predictions identical (no leakage)")


def test_score_node_and_sweep():
    rng = np.random.default_rng(4)
    # AR(1) mean-reverting series (has structure a forecaster can exploit)
    n = 240; a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    X = a.reshape(-1, 1)
    h = OOF.OOFHarness(min_train=40, stride=4, n_samples=150, seed=1)
    s = h.score_node("drift_forecast", X)
    for k in ["mean_crps", "crps_skill", "calib_ks", "point_mae", "n"]:
        check(k in s and (s["n"] == 0 or np.isfinite(s[k])), f"score_node missing/non-finite {k}")
    check(s["n"] > 0, "score_node produced no evaluations")
    table = h.sweep(X, ["naive_mean_forecast", "drift_forecast", "theta_forecast"])
    check(len(table) == 3, "sweep should return a row per node")
    skills = [r["crps_skill"] for r in table if np.isfinite(r["crps_skill"])]
    check(skills == sorted(skills, reverse=True), "sweep must be sorted by skill (best first)")
    print(f"  score/sweep: best={table[0]['node_type']} skill={table[0]['crps_skill']:.3f}, "
          f"mean_crps={table[0]['mean_crps']:.3f} (n={table[0]['n']})")


def test_oof_cache_inside_folder():
    rng = np.random.default_rng(5)
    X = np.cumsum(rng.normal(size=120)).reshape(-1, 1)
    root = os.path.join(OOF._PROJECT_ROOT, "data", "oof_cache_test")
    shutil.rmtree(root, ignore_errors=True)
    h = OOF.OOFHarness(min_train=30, stride=5, n_samples=60, cache_dir=root)
    r1 = h.node_oof("drift_forecast", X)
    key = h._key("drift_forecast", X)
    check(os.path.exists(os.path.join(root, f"{key}.npz")), "OOF cache file not written")
    r2 = h.node_oof("drift_forecast", X)                     # cache hit
    check(np.allclose(r1.point, r2.point) and np.array_equal(r1.index, r2.index),
          "cached OOF differs from computed")
    shutil.rmtree(root, ignore_errors=True)
    try:
        OOF.OOFHarness(cache_dir="/tmp/pb_oof_outside")
        check(False, "Rule 1: outside-folder OOF cache should be refused")
    except ValueError:
        pass
    print("  OOF cache: in-folder content-addressed + Rule-1 guard")


def test_forecast_node_discovery():
    fc = OOF.forecast_node_types()
    check(len(fc) > 5 and "drift_forecast" in fc, f"forecast-node discovery looks wrong: {len(fc)}")
    print(f"  forecast-node discovery: {len(fc)} nodes emit 'forecast'")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 4: Belief→feature + OOF harness: tests")
    print("=" * 70)
    test_belief_to_vector_fixed_width()
    test_projection_completeness()
    test_belief_to_samples_uses_node_uncertainty()
    test_oof_runs_and_is_leakage_safe()
    test_score_node_and_sweep()
    test_oof_cache_inside_folder()
    test_forecast_node_discovery()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
