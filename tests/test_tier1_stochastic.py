"""Oracle + design-data tests for the TIER-1 STOCHASTIC-PROCESS / PHYSICS nodes.

Per CLAUDE.md the ORACLE/acceptance check is written FIRST (this file) and the code in
``pattern_brain/nodes/tier1_stochastic.py`` is written until it passes. Each node gets:
  * a RECOVERY oracle (simulate the process with KNOWN parameters → recover them within tol),
  * a CONTROL oracle (the degenerate/null case behaves correctly — e.g. zero jumps on Gaussian,
    higher percolation threshold for independent assets),
  * a contract test (registered, Belief validates against the interlingua).

Design-data verdict tests (panel returns / RV / universe cross-section) live in
``MODEL_REPORT_tier1_stochastic.md`` as a runnable script section; the lightweight design
checks here exercise the nodes on the real corpus so the tests fail if the data path breaks.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Import so @register runs and the node classes are exercised through the public Node API.
import pattern_brain.nodes.tier1_stochastic as ts  # noqa: F401
from pattern_brain.interlingua import validate_belief
from pattern_brain.registry import create

_PANEL = os.path.join(os.path.dirname(__file__), "..", "data", "panel")
_UNIVERSE = os.path.join(os.path.dirname(__file__), "..", "data", "universe", "returns.npz")
_SYMBOLS = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT",
            "LINKUSDT", "LTCUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]


# ----------------------------------------------------------------- data helpers
def _returns(symbol: str) -> np.ndarray:
    ohlcv = np.load(os.path.join(_PANEL, f"{symbol}.npz"))["ohlcv"]
    close = ohlcv[:, 3].astype(float)
    return np.diff(np.log(close))


def _rv(symbol: str) -> np.ndarray:
    r = _returns(symbol)
    return r ** 2


# ----------------------------------------------------------------- simulators
def _sim_jump_diffusion(n, mu, sigma, lam, jmean, jstd, seed=0):
    rng = np.random.default_rng(seed)
    diffusion = rng.normal(mu, sigma, n)
    jump_mask = rng.random(n) < lam
    jumps = np.where(jump_mask, rng.normal(jmean, jstd, n), 0.0)
    return diffusion + jumps, jump_mask


def _sim_cir(n, kappa, theta, xi, v0=None, dt=1.0, seed=0):
    """Euler-Maruyama CIR variance path dv = kappa(theta-v)dt + xi*sqrt(v) dW (full truncation)."""
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = theta if v0 is None else v0
    for t in range(1, n):
        vp = max(v[t - 1], 0.0)
        v[t] = vp + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * rng.normal()
        v[t] = max(v[t], 0.0)
    return v


def _sim_sv(n, mu, phi, sigma_eta, seed=0):
    """Stochastic-vol: h_t = mu + phi(h_{t-1}-mu) + sigma_eta*eta; y_t = exp(h_t/2)*eps."""
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = mu
    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * rng.normal()
    y = np.exp(h / 2.0) * rng.normal(size=n)
    true_vol = np.exp(h / 2.0)
    return y, true_vol


def _sim_block_corr(n_assets, n_obs, block_size, rho, seed=0):
    """Block-correlated returns: assets in the same block share a common factor (corr ~ rho)."""
    rng = np.random.default_rng(seed)
    cols = []
    for start in range(0, n_assets, block_size):
        k = min(block_size, n_assets - start)
        common = rng.normal(size=(n_obs, 1))
        idio = rng.normal(size=(n_obs, k))
        block = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
        cols.append(block)
    return np.hstack(cols)[:, :n_assets]


# ================================================================ 1. jump_diffusion
class TestJumpDiffusion:
    def test_oracle_recovers_lambda_and_separates_jumps(self):
        lam = 0.05
        X, true_mask = _sim_jump_diffusion(2000, mu=0.0, sigma=0.01, lam=lam,
                                           jmean=0.0, jstd=0.08, seed=1)
        b = create("jump_diffusion").predict(X.reshape(-1, 1))
        p = b.payload
        # recover intensity within tolerance
        assert abs(p["lambda"] - lam) < 0.03, f"lambda {p['lambda']} vs {lam}"
        # high precision/recall separating labeled jumps from diffusion
        pred = np.array(p["series"], dtype=bool)
        tp = int(np.sum(pred & true_mask))
        precision = tp / max(int(pred.sum()), 1)
        recall = tp / max(int(true_mask.sum()), 1)
        assert precision > 0.6, f"precision {precision}"
        assert recall > 0.6, f"recall {recall}"
        # diffusion vol recovered (jumps removed)
        assert abs(p["diffusion_vol"] - 0.01) < 0.005, p["diffusion_vol"]

    def test_control_no_jumps_on_pure_gaussian(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0.0, 0.01, 2000).reshape(-1, 1)
        b = create("jump_diffusion").predict(X)
        # near-zero false jumps on clean Gaussian (threshold detector should be conservative)
        assert b.payload["lambda"] < 0.02, b.payload["lambda"]
        assert b.payload["n_jumps"] <= 10, b.payload["n_jumps"]

    def test_contract_and_real_data(self):
        for sym in _SYMBOLS[:3]:
            b = create("jump_diffusion").predict(_returns(sym).reshape(-1, 1))
            assert b.type == "signal"
            assert validate_belief(b) == []
            assert len(b.payload["series"]) == len(_returns(sym))


# ================================================================ 2. heston_vol
class TestHestonVol:
    def test_oracle_recovers_theta_and_kappa(self):
        kappa, theta, xi = 0.1, 0.04, 0.05
        v = _sim_cir(4000, kappa, theta, xi, v0=theta, seed=2)
        b = create("heston_vol").predict(v.reshape(-1, 1))
        p = b.payload
        # theta is the long-run level == mean RV; recover it tightly
        assert abs(p["theta"] - theta) < 0.01, f"theta {p['theta']} vs {theta}"
        # kappa (mean-reversion speed) recovered to the right order of magnitude
        assert 0.02 < p["kappa"] < 0.5, f"kappa {p['kappa']} vs {kappa}"

    def test_control_fast_vs_slow_reversion(self):
        v_fast = _sim_cir(4000, kappa=0.5, theta=0.04, xi=0.04, v0=0.04, seed=3)
        v_slow = _sim_cir(4000, kappa=0.02, theta=0.04, xi=0.04, v0=0.04, seed=3)
        k_fast = create("heston_vol").predict(v_fast.reshape(-1, 1)).payload["kappa"]
        k_slow = create("heston_vol").predict(v_slow.reshape(-1, 1)).payload["kappa"]
        # faster true mean reversion → larger estimated kappa (monotone control)
        assert k_fast > k_slow, f"k_fast {k_fast} !> k_slow {k_slow}"

    def test_contract_and_real_rv(self):
        b = create("heston_vol").predict(_rv("ETHUSDT").reshape(-1, 1))
        assert b.type == "forecast"
        assert validate_belief(b) == []
        assert b.payload["next_variance"] >= 0.0


# ================================================================ 3. sv_particle
class TestSVParticle:
    def test_oracle_filtered_tracks_true_vol(self):
        y, true_vol = _sim_sv(1500, mu=-9.0, phi=0.95, sigma_eta=0.3, seed=4)
        b = create("sv_particle").predict(y.reshape(-1, 1))
        filt = np.array(b.payload["series"], dtype=float)
        # filtered vol correlates strongly with the TRUE latent vol
        corr = float(np.corrcoef(filt, true_vol)[0, 1])
        assert corr > 0.7, f"corr {corr}"

    def test_control_degenerate_phi_zero(self):
        # phi=0 → no persistence; filter must still run and produce finite, positive vol
        y, _ = _sim_sv(800, mu=-9.0, phi=0.0, sigma_eta=0.3, seed=5)
        b = create("sv_particle").predict(y.reshape(-1, 1))
        filt = np.array(b.payload["series"], dtype=float)
        assert np.all(np.isfinite(filt))
        assert np.all(filt >= 0.0)

    def test_contract_and_real_data(self):
        b = create("sv_particle").predict(_returns("BTCUSDT").reshape(-1, 1))
        assert b.type in ("forecast", "signal")
        assert validate_belief(b) == []
        assert len(b.payload["series"]) == len(_returns("BTCUSDT"))


# ================================================================ 4. percolation_risk
class TestPercolationRisk:
    def test_oracle_block_corr_higher_threshold_than_independent(self):
        block = _sim_block_corr(40, 500, block_size=10, rho=0.7, seed=6)
        indep = np.random.default_rng(6).normal(size=(500, 40))
        tb = create("percolation_risk").predict(block).payload["percolation_threshold"]
        ti = create("percolation_risk").predict(indep).payload["percolation_threshold"]
        # the block-correlated system retains a giant component to a HIGHER threshold
        assert tb > ti, f"block thr {tb} !> indep thr {ti}"

    def test_control_order_parameter_monotone_in_connectivity(self):
        node = create("percolation_risk")
        high = _sim_block_corr(40, 500, block_size=40, rho=0.8, seed=8)   # one big block
        low = _sim_block_corr(40, 500, block_size=4, rho=0.3, seed=8)     # small weak blocks
        g_high = node.predict(high).payload["giant_component_frac"]
        g_low = node.predict(low).payload["giant_component_frac"]
        # more connectivity → larger giant-component fraction at the same threshold sweep
        assert g_high >= g_low, f"g_high {g_high} !>= g_low {g_low}"

    def test_contract_and_real_universe(self):
        R = np.load(_UNIVERSE)["returns"]
        b = create("percolation_risk").predict(R)
        assert b.type == "signal"
        assert validate_belief(b) == []
        assert len(b.payload["series"]) > 0
        assert 0.0 <= b.payload["giant_component_frac"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
