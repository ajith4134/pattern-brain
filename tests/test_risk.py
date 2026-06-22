"""Tests for the risk/volatility layer (PLAN.md §14, idea 1).

Run: python3 tests/test_risk.py  (light stack — numpy).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain import risk as R

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_risk_signals_basic():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 0.02, 5000)                     # ~2% vol return distribution
    sig = R.risk_signals(s, target_vol=0.01, var_level=0.05, capital=10_000)
    check(abs(sig["pred_vol"] - 0.02) < 0.002, f"pred_vol should be ~0.02, got {sig['pred_vol']}")
    check(sig["var"] > 0 and sig["es"] >= sig["var"], "ES should be >= VaR (deeper tail)")
    check(0 <= sig["position_fraction"] <= 3.0, "position fraction out of bounds")
    check(abs(sig["capital_at_risk"] - 10_000 * sig["var"]) < 1e-6, "capital-at-risk wrong")
    print(f"  risk signals: pred_vol={sig['pred_vol']:.3f}, VaR={sig['var']:.3f}, "
          f"ES={sig['es']:.3f}, size={sig['position_fraction']:.2f}")


def test_vol_targeting_sizes_down_in_high_vol():
    rng = np.random.default_rng(1)
    calm = R.risk_signals(rng.normal(0, 0.005, 4000), target_vol=0.01)
    storm = R.risk_signals(rng.normal(0, 0.05, 4000), target_vol=0.01)
    check(calm["position_fraction"] > storm["position_fraction"],
          "vol-targeting must size DOWN when forecast vol is higher")
    print(f"  vol-targeting: calm size={calm['position_fraction']:.2f} > storm "
          f"size={storm['position_fraction']:.2f}")


def test_vol_forecast_skill_detects_a_good_forecaster():
    """A forecast that tracks the true latent vol should beat a flat baseline on
    both correlation and QLIKE; a useless flat forecast should not."""
    rng = np.random.default_rng(2)
    T = 400
    latent = 0.01 + 0.02 * np.abs(np.sin(np.linspace(0, 8 * np.pi, T)))   # time-varying vol
    realized = rng.normal(0, latent)                    # returns with that vol
    good = latent + rng.normal(0, 0.001, T)             # tracks latent vol
    flat = np.full(T, float(np.mean(latent)))           # constant baseline
    ev = R.evaluate_vol_forecast(good, realized, baseline_vol=flat)
    check(ev["corr_with_realized"] > 0.2, f"good vol forecast should correlate, got {ev['corr_with_realized']:.2f}")
    check(ev["qlike_skill"] > 0, f"good forecast should beat flat baseline (QLIKE), got {ev['qlike_skill']:.3f}")
    ev_flat = R.evaluate_vol_forecast(flat, realized, baseline_vol=flat)
    check(abs(ev_flat["qlike_skill"]) < 1e-6, "flat-vs-flat skill should be ~0")
    print(f"  vol-forecast skill: good corr={ev['corr_with_realized']:.2f}, "
          f"QLIKE-skill={ev['qlike_skill']:.3f} > flat(0)")


def test_realized_vol_causal():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 1, 200)
    v1 = R.realized_vol(r, window=10)
    r2 = r.copy(); r2[150:] += 10
    v2 = R.realized_vol(r2, window=10)
    check(np.allclose(v1[:140], v2[:140]), "realized_vol must be causal (no look-ahead)")
    print("  realized_vol is causal (future perturbation leaves earlier values unchanged)")


def main():
    print("=" * 70)
    print("Pattern Brain — risk/volatility layer: tests")
    print("=" * 70)
    test_risk_signals_basic()
    test_vol_targeting_sizes_down_in_high_vol()
    test_vol_forecast_skill_detects_a_good_forecaster()
    test_realized_vol_causal()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
