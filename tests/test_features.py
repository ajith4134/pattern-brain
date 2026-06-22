"""Tests for Phase 8 slice 3 — the Feature Factory + Feature Store (PLAN.md §12).

Run: python3 tests/test_features.py  (light stack — numpy).
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.node import Node
from pattern_brain import features as F

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_transform_shape_and_validity():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=200)).reshape(-1, 1)
    fac = F.FeatureFactory()
    M, names = fac.transform(X)
    check(M.shape[0] == 200, f"feature matrix T changed: {M.shape}")
    check(M.shape[1] == len(names), f"cols {M.shape[1]} != names {len(names)}")
    check(M.shape[1] == 1 + len(F.CORE_FEATURES), f"expected raw+features cols, got {M.shape[1]}")
    Node._validate(M)                                   # must satisfy the core input contract
    check(np.all(np.isfinite(M)), "feature matrix has non-finite values")
    print(f"  transform → finite ({M.shape}) passing Node._validate; {len(names)} columns")


def test_causality_no_lookahead():
    """THE critical test: a feature at time t must use only data ≤ t. Perturb the
    FUTURE and every feature at earlier times must be byte-identical."""
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.normal(size=300)).reshape(-1, 1)
    fac = F.FeatureFactory()
    F1, _ = fac.transform(X)
    X2 = X.copy()
    X2[160:] += 50.0                                    # change only the future
    F2, _ = fac.transform(X2)
    check(np.allclose(F1[:150], F2[:150], atol=0, rtol=0),
          "LOOK-AHEAD: early features changed when only the future was perturbed")
    print("  causality: perturbing t≥160 leaves all features at t≤150 identical (no look-ahead)")


def test_specific_features_correct():
    x = np.arange(1, 21, dtype=float)
    fac = F.FeatureFactory(features=["lag1", "return", "roll_mean_5"], keep_raw=True)
    M, names = fac.transform(x.reshape(-1, 1))
    col = {n: M[:, i] for i, n in enumerate(names)}
    check(col["lag1"][5] == x[4], f"lag1 wrong: {col['lag1'][5]} != {x[4]}")
    check(col["lag1"][0] == x[0], "lag1[0] should be x[0] (earliest past)")
    check(abs(col["return"][7] - (x[7] - x[6])) < 1e-9, "return should be first difference")
    check(abs(col["roll_mean_5"][9] - np.mean(x[5:10])) < 1e-9, "roll_mean_5 wrong at t=9")
    check(abs(col["roll_mean_5"][2] - np.mean(x[:3])) < 1e-9, "roll_mean_5 should expand for warm-up")
    print("  lag/return/rolling-mean compute the correct causal values")


def test_unknown_feature_rejected_and_fracdiff_finite():
    try:
        F.FeatureFactory(features=["does_not_exist"])
        check(False, "unknown feature should raise")
    except ValueError:
        pass
    rng = np.random.default_rng(2)
    X = np.cumsum(rng.normal(size=120)).reshape(-1, 1)
    M, names = F.FeatureFactory(features=["fracdiff_04"]).transform(X)
    check(np.all(np.isfinite(M)) and "fracdiff_04" in names, "fracdiff column missing/non-finite")
    print("  unknown feature rejected; fractional-diff column finite")


def test_feature_store_caches_inside_folder():
    rng = np.random.default_rng(3)
    X = np.cumsum(rng.normal(size=150)).reshape(-1, 1)
    fac = F.FeatureFactory()
    root = os.path.join(F._PROJECT_ROOT, "data", "feature_store_test")
    shutil.rmtree(root, ignore_errors=True)
    store = F.FeatureStore(root=root)
    m1, n1 = store.get_or_compute(X, fac)
    key = store.key(X, fac)
    cached_path = os.path.join(store.root, f"{key}.npz")
    check(os.path.exists(cached_path), "feature store did not write a cache file")
    check(cached_path.startswith(F._PROJECT_ROOT + os.sep), "cache file escaped the project folder")
    m2, n2 = store.get_or_compute(X, fac)               # cache hit
    check(np.allclose(m1, m2) and n1 == n2, "cached result differs from computed")
    shutil.rmtree(root, ignore_errors=True)
    # Rule 1 guard: outside-folder root refused
    try:
        F.FeatureStore(root="/tmp/pb_outside_store")
        check(False, "Rule 1: outside-folder store root should be refused")
    except ValueError:
        pass
    print("  feature store caches inside the folder + Rule-1 guard refuses outside paths")


def test_feeds_the_bank():
    rng = np.random.default_rng(4)
    X = np.cumsum(rng.normal(size=180)).reshape(-1, 1)
    M, _ = F.FeatureFactory().transform(X)
    from pattern_brain.registry import create
    belief = create("ridge_regression").predict(M)
    check(belief.type == "equation" and np.isfinite(belief.confidence),
          "feature matrix failed to run through a bank node")
    print(f"  feature matrix runs through a bank node (Rule 21): belief={belief.type}")


def test_rule23_no_domain_identifiers():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book", "rsi", "macd",
                 "funding", "vwap", "crypto", "binance")
    path = os.path.join(F._PROJECT_ROOT, "pattern_brain", "features.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain identifiers in core features.py: {hits}")
    print("  Rule 23: core features.py carries no domain identifiers")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 3: Feature Factory + Store: tests")
    print("=" * 70)
    test_transform_shape_and_validity()
    test_causality_no_lookahead()
    test_specific_features_correct()
    test_unknown_feature_rejected_and_fracdiff_finite()
    test_feature_store_caches_inside_folder()
    test_feeds_the_bank()
    test_rule23_no_domain_identifiers()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
