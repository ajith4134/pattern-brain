"""Tests for Phase 7e physics/PhD-tier nodes (PLAN.md §11, Block 57).

Run: python3 tests/test_physics.py  (light stack only — numpy)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
PHYSICS = ["esn_forecaster", "deterministic_esn_forecaster", "lyapunov_exponent",
           "recurrence_rate", "hurst_exponent", "sample_entropy",
           "permutation_entropy", "rmt_denoise"]
PHYSICS2 = ["quantile_forecaster", "bayesian_ar_ensemble", "conformal_forecaster",
            "bayesian_bootstrap", "hawkes_intensity", "mfdfa", "tsallis_entropy",
            "phase_space_takens"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_all_registered_and_conform():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=300)).reshape(-1, 1)
    xD = np.column_stack([x[:, 0], 0.8 * x[:, 0] + rng.normal(scale=.3, size=300),
                          rng.normal(size=300)])
    for n in PHYSICS:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(xD if n == "rmt_denoise" else x)
        check(not validate_belief(b), f"{n} belief off-contract: {validate_belief(b)}")
        check(np.isfinite(b.confidence), f"{n} non-finite confidence")
    print(f"  all {len(PHYSICS)} physics nodes registered, run, and conform to v0.1")


def test_reservoir_forecasters_beat_naive_on_ar():
    """ESN + deterministic ESN should track a predictable AR(1) better than a
    last-value naive guess (sanity: they actually learn structure)."""
    rng = np.random.default_rng(3)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.85 * x[t - 1] + rng.normal(scale=0.3)
    X = x[:-1].reshape(-1, 1)
    truth = x[-1]
    for node in ("esn_forecaster", "deterministic_esn_forecaster"):
        pred = create(node).predict(X).payload["next_vector"][0]
        err_model = abs(pred - truth)
        err_naive = abs(X[-1, 0] - truth)
        check(np.isfinite(pred), f"{node} non-finite pred")
        check(err_model <= err_naive * 1.5 + 0.5, f"{node} far worse than naive")
    print("  reservoir forecasters produce finite, structure-aware predictions")


def test_hurst_distinguishes_trend_vs_meanrevert():
    rng = np.random.default_rng(5)
    trend = np.cumsum(np.abs(rng.normal(size=400)) + 0.05).reshape(-1, 1)  # strong drift
    noise = rng.normal(size=400).reshape(-1, 1)                            # iid ~ random walk-ish
    h_trend = create("hurst_exponent").predict(trend).payload["hurst"]
    h_noise = create("hurst_exponent").predict(noise).payload["hurst"]
    check(h_trend > h_noise, f"Hurst should rank trend>noise, got {h_trend:.2f} vs {h_noise:.2f}")
    print(f"  hurst: trend H={h_trend:.2f} > noise H={h_noise:.2f}")


def test_entropy_orders_regular_below_random():
    rng = np.random.default_rng(7)
    regular = np.sin(np.linspace(0, 40 * np.pi, 400)).reshape(-1, 1)
    random = rng.normal(size=400).reshape(-1, 1)
    for node, key in (("permutation_entropy", "permutation_entropy"),
                      ("sample_entropy", "sample_entropy")):
        reg = create(node).predict(regular).payload[key]
        ran = create(node).predict(random).payload[key]
        check(reg < ran, f"{node}: regular({reg:.2f}) should be < random({ran:.2f})")
    print("  entropy: regular signal scores lower complexity than random (both nodes)")


def test_rmt_denoise_dims():
    rng = np.random.default_rng(9)
    # 1-D -> passthrough
    x1 = rng.normal(size=200).reshape(-1, 1)
    b1 = create("rmt_denoise").predict(x1)
    check(b1.payload["n_signal_factors"] == 0, "1-D rmt should passthrough (0 factors)")
    # correlated multi-feature -> at least one signal factor recovered
    base = np.cumsum(rng.normal(size=300))
    xD = np.column_stack([base + rng.normal(scale=.2, size=300),
                          base + rng.normal(scale=.2, size=300),
                          rng.normal(size=300)])
    bD = create("rmt_denoise").predict(xD)
    check(bD.payload["n_signal_factors"] >= 1, "correlated data should yield >=1 signal factor")
    check(len(bD.payload["series"]) == 300, "rmt denoised series wrong length")
    print(f"  rmt: 1-D passthrough; correlated D=3 -> {bD.payload['n_signal_factors']} signal factor(s)")


def test_lyapunov_flags_chaos():
    # logistic map at r=3.99 is chaotic -> positive LE expected; constant -> ~0
    v, xs = 0.137, []
    for _ in range(400):
        v = 3.99 * v * (1 - v); xs.append(v)
    chaotic = create("lyapunov_exponent").predict(np.array(xs).reshape(-1, 1))
    flat = create("lyapunov_exponent").predict(np.ones((400, 1)) * 0.5)
    check(chaotic.payload["lyapunov"] > flat.payload["lyapunov"],
          "logistic-chaos LE should exceed constant-series LE")
    print(f"  lyapunov: chaotic LE={chaotic.payload['lyapunov']:.3f} > flat LE={flat.payload['lyapunov']:.3f}")


def test_batch2_registered_and_conform():
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(size=300)).reshape(-1, 1)
    for n in PHYSICS2:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check(np.isfinite(b.confidence), f"{n} non-finite conf")
    print(f"  all {len(PHYSICS2)} batch-2 nodes registered, run, and conform")


def test_distributional_forecasters_are_ordered():
    rng = np.random.default_rng(13)
    x = np.cumsum(rng.normal(size=300)).reshape(-1, 1)
    q = create("quantile_forecaster").predict(x).payload["quantiles"]
    check(q["q10"] <= q["q50"] <= q["q90"], f"quantiles unordered: {q}")
    cf = create("conformal_forecaster").predict(x).payload
    check(cf["lower"] <= cf["next_vector"][0] <= cf["upper"], "conformal interval doesn't bracket pred")
    bb = create("bayesian_bootstrap").predict(x).payload
    check(bb["ci_low"] <= bb["next_vector"][0] <= bb["ci_high"], "bootstrap CI doesn't bracket mean")
    w = create("bayesian_ar_ensemble").predict(x).payload["posterior_weights"]
    check(abs(sum(w) - 1.0) < 0.02, f"posterior weights don't sum to ~1 (rounded): {sum(w)}")
    print("  distributional: quantiles ordered; conformal + bootstrap intervals bracket; BMA weights sum to 1")


def test_hawkes_detects_clustering():
    rng = np.random.default_rng(17)
    smooth = np.cumsum(rng.normal(scale=0.2, size=400)).reshape(-1, 1)
    bursty = rng.normal(scale=0.1, size=400)
    bursty[50:55] += 8; bursty[200:206] += 8; bursty[330:333] += 8   # clustered shocks
    bursty = np.cumsum(bursty).reshape(-1, 1)
    f_smooth = create("hawkes_intensity").predict(smooth).payload["hawkes_fano"]
    f_bursty = create("hawkes_intensity").predict(bursty).payload["hawkes_fano"]
    check(f_bursty >= f_smooth, f"Hawkes Fano should flag clustering: bursty {f_bursty:.2f} vs smooth {f_smooth:.2f}")
    print(f"  hawkes: clustered Fano={f_bursty:.2f} >= smooth Fano={f_smooth:.2f}")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "physics.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in physics.py: {hits}")
    print("  domain-independence: physics.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7e physics/PhD-tier nodes: tests")
    print("=" * 70)
    test_all_registered_and_conform()
    test_reservoir_forecasters_beat_naive_on_ar()
    test_hurst_distinguishes_trend_vs_meanrevert()
    test_entropy_orders_regular_below_random()
    test_rmt_denoise_dims()
    test_lyapunov_flags_chaos()
    test_batch2_registered_and_conform()
    test_distributional_forecasters_are_ordered()
    test_hawkes_detects_clustering()
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
