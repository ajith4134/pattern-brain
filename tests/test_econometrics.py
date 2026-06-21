"""Tests for Phase 7a econometrics nodes (PLAN.md §11, Block 57). Light stack only."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
ECON = ["arma_forecast", "arima_forecast", "sarima_forecast", "var_forecast",
        "garch_forecast", "egarch_forecast", "ewma_volatility",
        "holt_winters_forecast", "prophet_like_forecast"]
VOL = ["garch_forecast", "egarch_forecast", "ewma_volatility"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _series(seed=1, n=360):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (0.02 * t + 3 * np.sin(2 * np.pi * t / 24)
            + np.cumsum(rng.normal(scale=0.5, size=n)))


def test_registered_run_conform():
    x = _series().reshape(-1, 1)
    xD = np.column_stack([x[:, 0], np.roll(x[:, 0], 1)])
    for n in ECON:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(xD if n == "var_forecast" else x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        pred = b.payload["next_vector"][0]
        check(np.isfinite(pred), f"{n} non-finite prediction {pred}")
    print(f"  all {len(ECON)} econometrics nodes registered, run, conform, finite preds")


def test_volatility_models_positive_variance():
    x = _series(2).reshape(-1, 1)
    for n in VOL:
        p = create(n).predict(x).payload
        check(p["variance"] > 0 and np.isfinite(p["variance"]), f"{n} bad variance {p.get('variance')}")
        check(abs(p["volatility"] - np.sqrt(p["variance"])) < 1e-6, f"{n} vol != sqrt(var)")
    g = create("garch_forecast").predict(x).payload
    check(0 <= g["persistence"] < 1.0, f"GARCH persistence out of range: {g['persistence']}")
    print(f"  volatility: all positive variance; GARCH persistence={g['persistence']:.2f} in [0,1)")


def test_var_returns_full_vector():
    x = _series(3)
    xD = np.column_stack([x, 0.5 * x + 1.0, np.roll(x, 2)])
    nv = create("var_forecast").predict(xD).payload["next_vector"]
    check(len(nv) == 3, f"VAR should forecast all 3 channels, got {len(nv)}")
    check(all(np.isfinite(v) for v in nv), "VAR produced non-finite forecast")
    print(f"  var: VAR(1) forecasts the full {len(nv)}-channel vector")


def test_seasonal_models_detect_period():
    # strong period-24 seasonality -> SARIMA/Holt-Winters/Prophet should detect it
    t = np.arange(300)
    x = (5 * np.sin(2 * np.pi * t / 24) + np.cumsum(np.random.default_rng(4).normal(scale=.2, size=300))).reshape(-1, 1)
    found = []
    for n in ("sarima_forecast", "holt_winters_forecast", "prophet_like_forecast"):
        s = create(n).predict(x).payload.get("season", 0)
        found.append(s)
    check(any(abs(s - 24) <= 2 for s in found), f"no node detected the ~24 period: {found}")
    print(f"  seasonal: detected periods {found} (expected ~24)")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "econometrics.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in econometrics.py: {hits}")
    print("  domain-independence: econometrics.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7a econometrics nodes: tests")
    print("=" * 70)
    test_registered_run_conform()
    test_volatility_models_positive_variance()
    test_var_returns_full_vector()
    test_seasonal_models_detect_period()
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
