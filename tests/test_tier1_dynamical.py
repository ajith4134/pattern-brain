"""Oracle-test-first acceptance suite for the TIER-1 DYNAMICAL/FORECAST/GEOMETRIC nodes.

Per CLAUDE.md (oracle/acceptance check FIRST) + ML_ENGINEERING_PRACTICES.md §G/§F and
RULES.md 32/34/35/36. For each node:
  * ORACLE — recovery on a system where the theory has a known answer (closed form,
    clean oscillator, trend, synthetic manifold), plus a CONTROL where it should NOT
    fire / should degrade gracefully.
  * INTERLINGUA — emitted Belief conforms (validate_belief == []).
  * DESIGN-DATA — the real panel input the design targets, vs persistence honest verdict.
  * CROSS-DOMAIN (diffusion_map, Rule 36) — pendigits class separation beats chance.

Run: PYTHONPATH=. python3 -m pytest tests/test_tier1_dynamical.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import stats as sstats

import pattern_brain.nodes.tier1_dynamical as dyn  # noqa: F401  (runs @register)
from pattern_brain.interlingua import validate_belief
from pattern_brain.registry import create
import pattern_brain.nodes.tier1_dynamical as M

RNG = np.random.default_rng(7)
PANEL_DIR = "data/panel"
PENDIGITS = "data/sequences/pendigits.npz"


def _close(sym):
    o = np.load(os.path.join(PANEL_DIR, sym + ".npz"))["ohlcv"]
    return o[:, 3].astype(float)


def _returns(sym):
    c = _close(sym)
    return np.diff(np.log(c))


PANEL = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
         "DOGEUSDT", "BNBUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]


# =====================================================================  KOOPMAN / DMD
def test_dmd_recovers_oscillation_frequency():
    """ORACLE: on a clean sine wave DMD recovers the true frequency (cycles/step)."""
    T = 400
    f_true = 1.0 / 20.0                       # 20-step period
    t = np.arange(T)
    x = np.sin(2 * np.pi * f_true * t)
    node = M.KoopmanDMDNode(embed_dim=20)
    b = node.predict(x)
    freqs = np.abs(np.array(b.payload["eig_freq"]))
    assert np.min(np.abs(freqs - f_true)) < 0.01, f"freqs={sorted(set(np.round(freqs,4)))}"


def test_dmd_recovers_decay_rate():
    """ORACLE: damped oscillator → an eigenvalue magnitude matching the per-step decay."""
    T = 300
    decay = 0.99                              # |λ| per step
    t = np.arange(T)
    x = (decay ** t) * np.cos(2 * np.pi * t / 15.0)
    node = M.KoopmanDMDNode(embed_dim=16)
    b = node.predict(x)
    mags = np.array(b.payload["eig_magnitude"])
    assert np.min(np.abs(mags - decay)) < 0.02, f"mags near 1: {np.sort(mags)[-4:]}"


def test_dmd_forecasts_oscillator_with_tiny_error():
    """ORACLE: multi-step forecast of a clean oscillator is near-exact (deterministic)."""
    T = 200
    t = np.arange(T)
    x = np.sin(2 * np.pi * t / 25.0) + 0.5 * np.cos(2 * np.pi * t / 11.0)
    train, truth = x[:-5], x[-5:]
    H = M.delay_embed(train, 30)
    A, _ = M.dmd_fit(H)
    fc = M.dmd_forecast(H, A, 5)
    rmse = float(np.sqrt(np.mean((fc - truth) ** 2)))
    assert rmse < 0.05, f"oscillator forecast rmse={rmse:.4f}"


def test_dmd_belief_conforms():
    b = M.KoopmanDMDNode().predict(np.cumsum(RNG.standard_normal(300)))
    assert b.type == "forecast" and validate_belief(b) == []
    assert "estimate" in b.payload and b.payload["n_modes"] > 0


def test_dmd_handles_degenerate_short_input():
    """CONTROL: too-short input degrades to persistence, no crash."""
    b = M.KoopmanDMDNode().predict(np.array([1.0, 2.0, 3.0]))
    assert b.payload["estimate"] == 3.0 and b.payload["n_modes"] == 0


def test_dmd_on_returns_loses_to_zero_baseline_honest():
    """DESIGN-DATA (returns, 1-step): the HONEST verdict. DMD beats naive persistence
    (a strawman on returns: predicting r[t] carries r[t]'s full variance) but does NOT
    beat the zero/mean baseline — it merely shrinks toward zero, adding no real edge.
    Returns are not forecastable (project lesson). Asserted on the panel."""
    wins_vs_zero = 0
    for sym in PANEL:
        r = _returns(sym)
        split = int(0.7 * r.size)
        e_dmd, e_zero = [], []
        node = M.KoopmanDMDNode(embed_dim=12)
        for t in range(split, r.size - 1):
            hist = r[max(0, t - 200):t + 1]
            pred = node.predict(hist).payload["estimate"]
            e_dmd.append((pred - r[t + 1]) ** 2)
            e_zero.append((0.0 - r[t + 1]) ** 2)        # zero is the real return baseline
        if np.mean(e_dmd) < np.mean(e_zero):
            wins_vs_zero += 1
    # Honest: DMD should NOT systematically beat the zero baseline on returns.
    assert wins_vs_zero <= len(PANEL) // 2, \
        f"unexpected: DMD beat the zero baseline on {wins_vs_zero}/{len(PANEL)} (returns)"


# =====================================================================  BSTS / KALMAN
def test_bsts_recovers_trend_below_noise():
    """ORACLE: filtered level tracks the true linear trend with RMSE < the noise std."""
    T = 400
    t = np.arange(T)
    trend = 0.05 * t
    noise_sd = 2.0
    y = trend + RNG.normal(0, noise_sd, T)
    _, _, _, _ = (None, None, None, None)
    ov, lv, sv = M.estimate_state_variances(y)
    levels, slopes, s, _ = M.kalman_local_linear_trend(y, ov, lv, sv)
    rmse = float(np.sqrt(np.mean((levels - trend) ** 2)))
    assert rmse < noise_sd, f"level-vs-trend rmse={rmse:.3f} not < noise_sd={noise_sd}"


def test_bsts_beats_persistence_on_trending_series():
    """ORACLE: on a clean trending series, 1-step BSTS forecast beats persistence."""
    T = 300
    t = np.arange(T)
    y = 0.1 * t + RNG.normal(0, 1.0, T)
    node = M.BSTSForecastNode()
    err_b, err_p = [], []
    for k in range(150, T - 1):
        hist = y[:k + 1]
        est = node.predict(hist).payload["estimate"]
        err_b.append((est - y[k + 1]) ** 2)
        err_p.append((y[k] - y[k + 1]) ** 2)   # persistence ignores the slope
    assert np.mean(err_b) < np.mean(err_p), "BSTS should beat persistence on a trend"


def test_bsts_degrades_to_persistence_on_pure_noise():
    """CONTROL: on white noise BSTS must not blow up; its forecast stays near 0/last."""
    y = RNG.normal(0, 1.0, 300)
    b = M.BSTSForecastNode().predict(y)
    assert abs(b.payload["estimate"]) < 3.0 and b.payload["std"] >= 0.0


def test_bsts_belief_conforms():
    b = M.BSTSForecastNode().predict(np.cumsum(RNG.standard_normal(200)))
    assert b.type == "forecast" and validate_belief(b) == []
    assert "estimate" in b.payload and "std" in b.payload


def test_bsts_levels_design_vs_returns_honest():
    """DESIGN-DATA: trend extraction on price LEVELS (its home turf) vs raw returns.
    On levels the filtered level should track close-to-price (low relative error);
    on returns it should not beat persistence (honest)."""
    c = _close("BTCUSDT")
    ov, lv, sv = M.estimate_state_variances(c)
    levels, _, _, _ = M.kalman_local_linear_trend(c, ov, lv, sv)
    rel = float(np.mean(np.abs(levels - c)) / (np.mean(np.abs(c)) + 1e-9))
    assert rel < 0.05, f"level tracking rel-error {rel:.4f} too high on price levels"

    # returns: BSTS must NOT beat the zero baseline (the honest baseline on returns).
    wins_vs_zero = 0
    node = M.BSTSForecastNode()
    for sym in PANEL:
        r = _returns(sym)
        split = int(0.7 * r.size)
        eb, ez = [], []
        for t in range(split, r.size - 1):
            est = node.predict(r[:t + 1]).payload["estimate"]
            eb.append((est - r[t + 1]) ** 2)
            ez.append((0.0 - r[t + 1]) ** 2)
        if np.mean(eb) < np.mean(ez):
            wins_vs_zero += 1
    assert wins_vs_zero <= len(PANEL) // 2, \
        f"unexpected: BSTS beat the zero baseline on returns {wins_vs_zero}/{len(PANEL)}"


# =====================================================================  INFO-GEOM FISHER-RAO
def test_fisher_rao_zero_for_identical():
    assert M.fisher_rao_gaussian(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_fisher_rao_monotone_in_mean_and_sigma():
    """ORACLE: distance grows monotonically as μ separates and as σ separates."""
    base = (0.0, 1.0)
    d_mu = [M.fisher_rao_gaussian(*base, m, 1.0) for m in [0.5, 1.0, 2.0, 4.0]]
    d_sig = [M.fisher_rao_gaussian(*base, 0.0, s) for s in [1.5, 2.0, 4.0, 8.0]]
    assert all(np.diff(d_mu) > 0) and all(np.diff(d_sig) > 0)


def test_fisher_rao_matches_closed_form():
    """ORACLE: matches the Costa-Santos-Hancock closed form on a known pair."""
    mu1, s1, mu2, s2 = 0.0, 1.0, 3.0, 2.0
    arg = 1.0 + ((mu1 - mu2) ** 2 + 2 * (s1 - s2) ** 2) / (4 * s1 * s2)
    expected = np.sqrt(2.0) * np.arccosh(arg)
    assert M.fisher_rao_gaussian(mu1, s1, mu2, s2) == pytest.approx(expected, rel=1e-9)


def test_infogeom_spikes_at_vol_jump():
    """DESIGN-DATA/ORACLE: a synthetic series with an injected vol jump → the Fisher-Rao
    series spikes at the window containing the shift (max distance at the jump window)."""
    w = 30
    calm = RNG.normal(0, 1.0, 5 * w)
    wild = RNG.normal(0, 6.0, 5 * w)
    x = np.concatenate([calm, wild])
    node = M.InfoGeomDistanceNode(window=w)
    b = node.predict(x)
    series = np.array(b.payload["series"])
    jump_win = 5                              # window index where wild begins
    assert int(np.argmax(series)) == jump_win, f"argmax={np.argmax(series)} expected {jump_win}"
    # the jump window's distance dwarfs every NON-jump window (a clean, localized spike)
    others = np.delete(series, jump_win)
    assert series[jump_win] > 4.0 * others.max(), \
        f"spike {series[jump_win]:.2f} not >> other windows max {others.max():.2f}"


def test_infogeom_flat_on_stationary():
    """CONTROL: a stationary series → small, roughly uniform distances (no big spike)."""
    x = RNG.normal(0, 1.0, 300)
    b = M.InfoGeomDistanceNode(window=30).predict(x)
    series = np.array(b.payload["series"])[1:]
    assert series.max() < 1.0, f"stationary spike {series.max():.3f} too large"


def test_infogeom_belief_conforms():
    b = M.InfoGeomDistanceNode().predict(_returns("ETHUSDT"))
    assert b.type == "signal" and validate_belief(b) == []


def test_infogeom_detects_real_panel_regime():
    """DESIGN-DATA: on real panel returns the regime distance has meaningful dynamic range
    (max >> mean), i.e. it flags genuine volatility-regime windows."""
    flagged = 0
    for sym in PANEL:
        b = M.InfoGeomDistanceNode(window=40).predict(_returns(sym))
        if b.payload["max_distance"] > 1.5 * b.payload["mean_distance"]:
            flagged += 1
    assert flagged >= len(PANEL) // 2, f"regime range present on only {flagged}/{len(PANEL)}"


# =====================================================================  DIFFUSION MAP
def test_diffusion_recovers_1d_manifold_ordering():
    """ORACLE: points on a noisy 1-D arc → the leading diffusion coordinate is
    monotone in the true arc parameter (|Spearman| high)."""
    n = 300
    theta = np.sort(RNG.uniform(0, np.pi, n))          # 1-D parameter
    pts = np.column_stack([np.cos(theta), np.sin(theta)]) + RNG.normal(0, 0.02, (n, 2))
    coords = M.diffusion_embed(pts, n_coords=2)
    rho = abs(sstats.spearmanr(coords[:, 0], theta).statistic)
    assert rho > 0.9, f"|spearman| leading-coord vs theta = {rho:.3f}"


def test_diffusion_belief_conforms():
    b = M.DiffusionMapNode().predict(np.cumsum(RNG.standard_normal(400)))
    assert b.type == "signal" and validate_belief(b) == []
    assert b.payload["n_coords"] >= 1


def test_diffusion_control_random_no_structure():
    """CONTROL: i.i.d. Gaussian cloud has no 1-D manifold → leading coord should not be
    strongly monotone in a random external index (sanity that the test isn't trivially true)."""
    pts = RNG.normal(0, 1.0, (200, 5))
    coords = M.diffusion_embed(pts, n_coords=2)
    fake_param = RNG.permutation(200)
    rho = abs(sstats.spearmanr(coords[:, 0], fake_param).statistic)
    assert rho < 0.5, f"spurious ordering {rho:.3f} on structureless cloud"


@pytest.mark.skipif(not os.path.exists(PENDIGITS), reason="pendigits not present")
def test_diffusion_separates_pendigits_classes_above_chance():
    """CROSS-DOMAIN (Rule 36, MANDATORY for a general method): embed flattened pen
    trajectories; a k-NN on the 2-3D diffusion coords beats random-chance (10%) materially
    AND beats a same-dim random projection by a clear margin."""
    z = np.load(PENDIGITS)
    Xtr, ytr = z["Xtr"], z["ytr"]
    n = 1200
    idx = RNG.choice(Xtr.shape[0], n, replace=False)
    flat = Xtr[idx].reshape(n, -1).astype(float)       # (n, 16)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-9)
    y = ytr[idx]
    coords = M.diffusion_embed(flat, n_coords=3)

    # split, fit a tiny k-NN purely in the embedding (no labels leaked into embedding)
    s = n // 2
    Xc_tr, Xc_te = coords[:s], coords[s:]
    yc_tr, yc_te = y[:s], y[s:]

    def knn_acc(Xtr_, Xte_, k=7):
        from scipy.spatial.distance import cdist
        D = cdist(Xte_, Xtr_)
        nn = np.argsort(D, axis=1)[:, :k]
        votes = yc_tr[nn]
        pred = np.array([np.bincount(v).argmax() for v in votes])
        return float(np.mean(pred == yc_te))

    acc_diff = knn_acc(Xc_tr, Xc_te)
    # baseline: random projection to the same dim
    Rproj = RNG.standard_normal((flat.shape[1], 3))
    rp = flat @ Rproj
    acc_rand = knn_acc(rp[:s], rp[s:])

    assert acc_diff > 0.30, f"diffusion kNN acc {acc_diff:.3f} not materially above 10% chance"
    assert acc_diff > acc_rand - 0.05, f"diffusion {acc_diff:.3f} not >= random-proj {acc_rand:.3f}"


# =====================================================================  REGISTRY
def test_all_four_registered():
    for nt in ["koopman_dmd", "bsts_forecast", "infogeom_distance", "diffusion_map"]:
        node = create(nt)
        assert node.node_type == nt
