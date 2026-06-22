"""Tests for Phase 8 slice 9 — the capstone freeze→forward-test (PLAN.md §12.1).

Run: python3 tests/test_capstone.py  (offline/synthetic; ccxt live path guarded).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.capstone import run_capstone
from pattern_brain.adapters import crypto

FAILS = []
POOL = ["drift_forecast", "theta_forecast", "naive_mean_forecast"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _matrix(n=200, seed=0):
    """A (T,3) where col 0 is an AR(1) 'return' (mildly predictable) + 2 features."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.5 * r[t - 1] + rng.normal(scale=0.4)
    f1 = np.cumsum(r)
    f2 = rng.normal(size=n)
    return np.column_stack([r, f1, f2])


def _run(seed=0):
    return run_capstone(_matrix(seed=seed), holdout_frac=0.3, strategy="random", budget=6,
                        base_pool=POOL, min_train=40, n_samples=40, seed=seed)


def test_capstone_report_shape():
    rep = _run()
    for k in ("n_total", "n_history", "n_holdout", "history_skill", "best_spec",
              "forward_crps", "forward_skill", "forward_calibrated", "forward_dsr",
              "net_directional_hit", "baseline_directional_hit", "next_move", "verdict"):
        check(k in rep, f"report missing {k}")
    check(rep["n_history"] == int(round(rep["n_total"] * 0.7)), "history split should be 70%")
    check(0 < rep["n_holdout"] < rep["n_total"], "holdout must be a non-trivial unseen tail")
    check(rep["verdict"] in ("PASS", "WEAK", "FAIL"), f"bad verdict {rep['verdict']}")
    print(f"  report ok: {rep['n_history']} hist / {rep['n_holdout']} holdout, "
          f"forward_skill={rep['forward_skill']:.3f}, verdict={rep['verdict']}")


def test_history_only_selection_and_next_move():
    rep = _run()
    # the search saw only the history block (by construction n_history rows)
    check(rep["n_history"] + rep["n_holdout"] <= rep["n_total"], "history+holdout exceeds total")
    nm = rep["next_move"]
    check(nm["q10"] <= nm["q50"] <= nm["q90"], f"next-move quantiles unordered: {nm}")
    check(0.0 <= nm["prob_up"] <= 1.0, "prob_up out of range")
    print(f"  next move: mean={nm['mean']:.4f}, P(up)={nm['prob_up']:.2f} "
          f"[{nm['q10']:.3f}, {nm['q90']:.3f}]")


def test_capstone_deterministic():
    a, b = _run(seed=1), _run(seed=1)
    check(abs(a["forward_skill"] - b["forward_skill"]) < 1e-9 and a["verdict"] == b["verdict"],
          "capstone must be reproducible for a fixed seed")
    print("  capstone reproducible under a fixed seed (honest, re-runnable science)")


def test_crypto_synthetic_and_cache_guard():
    d = crypto.load_symbol("SYNTHCOIN/USD", "1h", limit=300, prefer_live=False, use_cache=False, allow_synthetic=True)
    check(d["source"] == "synthetic", f"offline load should be synthetic, got {d['source']}")
    check(d["matrix"].ndim == 2 and d["matrix"].shape[1] >= 1 and np.all(np.isfinite(d["matrix"])),
          "synthetic crypto matrix invalid")
    check("last_close" in d and d["n_candles"] > 0, "crypto meta incomplete")
    # cache read path: write a fake cache, then load reads it
    cp = crypto._cache_path("TEST/USD", "1h", "kraken")
    np.savez(cp, ohlcv=crypto.synthetic_ohlcv(T=300, seed=5))
    d2 = crypto.load_symbol("TEST/USD", "1h", limit=300, prefer_live=False)
    check(d2["source"] == "cache", f"should read cache, got {d2['source']}")
    os.remove(cp)
    check(cp.startswith(crypto._CACHE_DIR + os.sep), "cache must be inside the project folder")
    print(f"  crypto: synthetic fallback + in-folder cache read (Rule 1); features={d['feature_names']}")


def test_fetch_crypto_symbol_tool():
    from pattern_brain.agent.tools import Toolbox
    out = Toolbox().fetch_crypto_symbol("SYNTHCOIN/USD", "1h", limit=300, prefer_live=False)
    check(out["source"] == "synthetic" and out["X"].shape[1] >= 1, "tool returned bad data")
    check("last_close" in out and len(out["feature_names"]) >= 1, "tool meta incomplete")
    print(f"  agent tool fetch_crypto_symbol → (T,D)={out['shape']}, source={out['source']}")


def test_run_crypto_capstone_offline():
    rep = crypto.run_crypto_capstone("SYNTHCOIN/USD", "1h", limit=260, prefer_live=False,
                                     holdout_frac=0.3, strategy="random", budget=5,
                                     base_pool=POOL, min_train=45, n_samples=40, seed=0)
    check(rep["source"] == "synthetic" and rep["verdict"] in ("PASS", "WEAK", "FAIL"),
          "crypto capstone offline failed")
    check(rep["next_price_est"] is not None, "should estimate next price from last_close")
    print(f"  run_crypto_capstone (offline): verdict={rep['verdict']}, "
          f"next_price_est={rep['next_price_est']:.1f} (last_close={rep['last_close']:.1f})")


def test_run_crypto_sweep_offline():
    res = crypto.run_crypto_sweep(symbols=("SYNTHA/USD", "SYNTHB/USD"), timeframes=("1h",),
                                  prefer_live=False, holdout_frac=0.3, strategy="random",
                                  budget=4, base_pool=POOL, min_train=45, n_samples=30, seed=0)
    agg = res["aggregate"]
    check(agg["n_runs"] == 2 and agg["n_failed"] == 0, f"sweep should run 2 ok, got {agg}")
    check(agg["median_forward_skill"] is not None and 0 <= agg["n_pass"] <= 2, "sweep agg malformed")
    check(len(res["runs"]) == 2, "sweep should report a row per symbol×timeframe")
    print(f"  sweep (offline, 2 runs): median skill={agg['median_forward_skill']:.3f}, "
          f"pass={agg['n_pass']}/{agg['n_runs']}")


def test_volatility_target_offline():
    rep = crypto.run_crypto_capstone("SYNTHA/USD", prefer_live=False, target="volatility",
                                     holdout_frac=0.3, strategy="random", budget=4,
                                     base_pool=POOL, min_train=45, n_samples=30, seed=0)
    check(rep.get("target") == "volatility", "report should carry the volatility target")
    check(rep["next_price_est"] is None, "vol target must NOT produce a price estimate")
    check(rep["verdict"] in ("PASS", "WEAK", "FAIL"), f"bad vol verdict {rep['verdict']}")
    print(f"  volatility target: verdict={rep['verdict']}, skill={rep['forward_skill']:.3f} "
          f"(no price est, correct)")


def test_ccxt_live_guarded():
    if not crypto.ccxt_available():
        print("  ccxt not installed -> live fetch skipped cleanly")
        return
    try:
        d = crypto.load_symbol("BTC/USD", "1h", limit=200, prefer_live=True)
    except Exception as e:
        print(f"  ccxt live fetch unavailable ({type(e).__name__}) -> skipped (offline ok)")
        return
    check(d["source"] in ("live", "cache") and d["matrix"].shape[0] > 50, "live crypto load invalid")
    print(f"  ccxt LIVE: {d['n_candles']} {d['symbol']} candles ({d['source']}), last_close={d['last_close']:.1f}")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 9: capstone freeze→forward-test: tests")
    print("=" * 70)
    test_capstone_report_shape()
    test_history_only_selection_and_next_move()
    test_capstone_deterministic()
    test_crypto_synthetic_and_cache_guard()
    test_fetch_crypto_symbol_tool()
    test_run_crypto_capstone_offline()
    test_run_crypto_sweep_offline()
    test_volatility_target_offline()
    test_ccxt_live_guarded()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
