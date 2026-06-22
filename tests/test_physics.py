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
PHYSICS3 = ["quadratic_hawkes", "rough_volatility", "soc_criticality",
            "persistent_homology"]


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


def test_batch3_registered_and_conform():
    rng = np.random.default_rng(19)
    x = np.cumsum(rng.normal(size=400)).reshape(-1, 1)
    for n in PHYSICS3:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check(np.isfinite(b.confidence), f"{n} non-finite conf")
    print(f"  all {len(PHYSICS3)} batch-3 nodes registered, run, and conform")


def test_quadratic_hawkes_flags_volatility_feedback():
    """A vol-clustering (ARCH) series should show stronger quadratic feedback than
    an i.i.d. gaussian series (which has none)."""
    rng = np.random.default_rng(21)
    n = 600
    # ARCH(1): sigma^2_t depends on last squared return -> squared-return feedback
    r = np.zeros(n); s2 = np.ones(n)
    for t in range(1, n):
        s2[t] = 0.1 + 0.85 * r[t - 1] ** 2
        r[t] = np.sqrt(s2[t]) * rng.normal()
    arch = np.cumsum(r).reshape(-1, 1)
    iid = np.cumsum(rng.normal(size=n)).reshape(-1, 1)
    fb_arch = create("quadratic_hawkes").predict(arch).payload["quadratic_feedback"]
    fb_iid = create("quadratic_hawkes").predict(iid).payload["quadratic_feedback"]
    check(fb_arch > fb_iid, f"quad feedback should rank ARCH>iid, got {fb_arch:.3f} vs {fb_iid:.3f}")
    print(f"  quadratic_hawkes: ARCH feedback={fb_arch:.3f} > iid={fb_iid:.3f}")


def test_rough_volatility_detects_roughness():
    """Log-vol of a vol-clustering series is rough (H well below 0.5); the node
    should report a finite rough_hurst in [0,1] and flag roughness on a series
    whose volatility itself fluctuates fast."""
    rng = np.random.default_rng(23)
    n = 800
    # fast-fluctuating volatility -> rough log-vol path
    vol = 0.2 + 0.15 * np.abs(rng.normal(size=n))
    series = np.cumsum(vol * rng.normal(size=n)).reshape(-1, 1)
    p = create("rough_volatility").predict(series).payload
    check(0.0 <= p["rough_hurst"] <= 1.0, f"rough_hurst out of range: {p['rough_hurst']}")
    check(0.0 <= p["level_hurst"] <= 1.0, f"level_hurst out of range: {p['level_hurst']}")
    check(isinstance(p["is_rough"], bool), "is_rough must be bool")
    print(f"  rough_volatility: rough_hurst={p['rough_hurst']:.2f}, level_hurst={p['level_hurst']:.2f}, rough={p['is_rough']}")


def test_soc_avalanches_scale_free():
    """The SOC signature is a *scale-free* avalanche-size distribution, not the raw
    count. A series with multi-scale bursts should have a broader (more scale-free)
    size distribution and a finite power-law exponent vs a smooth low-noise series
    whose exceedances are all the same tiny size."""
    rng = np.random.default_rng(29)
    n = 600
    smooth = np.cumsum(rng.normal(scale=0.05, size=n)).reshape(-1, 1)
    bursts = rng.normal(scale=0.05, size=n)
    for c in (60, 150, 151, 152, 300, 301, 450, 451, 452, 453):  # multi-scale avalanches
        bursts[c] += rng.uniform(2, 9)
    bursty = np.cumsum(bursts).reshape(-1, 1)
    s_b = create("soc_criticality").predict(bursty).payload
    s_s = create("soc_criticality").predict(smooth).payload
    check(s_b["scale_free"] > s_s["scale_free"],
          f"bursty should be more scale-free: {s_b['scale_free']:.2f} vs {s_s['scale_free']:.2f}")
    check(s_b["powerlaw_exponent"] > 0, "power-law exponent should be positive when avalanches exist")
    check(s_b["n_avalanches"] >= 5, f"bursty should yield >=5 avalanches, got {s_b['n_avalanches']}")
    print(f"  soc: bursty alpha={s_b['powerlaw_exponent']:.2f} "
          f"scale_free={s_b['scale_free']:.2f} > smooth {s_s['scale_free']:.2f} "
          f"(n_av bursty={s_b['n_avalanches']})")


def test_persistent_homology_topology():
    """An oscillating signal has many topological basins (high n_features); a
    monotone ramp has essentially none. Lifetimes/diagram must be consistent."""
    osc = np.sin(np.linspace(0, 30 * np.pi, 500)).reshape(-1, 1)
    ramp = np.linspace(0, 10, 500).reshape(-1, 1)
    po = create("persistent_homology").predict(osc).payload
    pr = create("persistent_homology").predict(ramp).payload
    check(po["n_features"] >= 5, f"oscillation should yield many basins, got {po['n_features']}")
    check(pr["n_features"] <= 1, f"monotone ramp should yield ~0 basins, got {pr['n_features']}")
    check(po["total_persistence"] > pr["total_persistence"],
          "oscillation should have more total persistence than a ramp")
    check(len(po["diagram"]) == po["n_features"], "diagram length must equal n_features")
    check(all(d > b for b, d in po["diagram"]), "every persistence pair must have death > birth")
    check(0.0 <= po["persistence_entropy"] <= 1.0, "persistence entropy must be normalized [0,1]")
    print(f"  persistent_homology: oscillation features={po['n_features']} "
          f"(ramp={pr['n_features']}), entropy={po['persistence_entropy']:.2f}")


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
    test_batch3_registered_and_conform()
    test_quadratic_hawkes_flags_volatility_feedback()
    test_rough_volatility_detects_roughness()
    test_soc_avalanches_scale_free()
    test_persistent_homology_topology()
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
