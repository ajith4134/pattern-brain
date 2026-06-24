"""Oracle tests for the TIER-1 ALLOCATION/CONTROL nodes — written to define the
acceptance contract (CLAUDE.md Phase 4: oracle FIRST). Each node has a TRUE-POSITIVE
oracle (it does the thing) + a CONTROL (it does NOT fire when it shouldn't), then a
REAL-data verdict vs baselines on the design-input data (ML_ENGINEERING §G).

Run: PYTHONPATH=. python3 -m pytest tests/test_tier1_allocation.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import pattern_brain.nodes.tier1_allocation  # noqa: F401  (runs @register)
from pattern_brain.nodes.tier1_allocation import (
    hrp_weights, recursive_bisection, _cov_corr,
    cvar_min_weights,
    merton_fraction, merton_weights,
    mpc_position_path, naive_position_path, _turnover,
)
from pattern_brain.registry import create
from pattern_brain.interlingua import validate_belief

_UNIVERSE = "data/universe/returns.npz"
_UNIVERSE_DAILY = "data/universe_daily/returns.npz"


# ---------------------------------------------------------------- synthetic data
def _block_correlated_returns(rng, T=2000, blocks=(20, 20, 20), rho=0.85):
    """Returns with `blocks` highly-correlated groups, blocks ~independent of each
    other. A clustering allocator should spread weight ACROSS blocks."""
    cols = []
    labels = []
    for bi, k in enumerate(blocks):
        common = rng.standard_normal((T, 1))
        idio = rng.standard_normal((T, k))
        block = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
        cols.append(block)
        labels += [bi] * k
    return np.hstack(cols), np.array(labels)


# ================================================================== HRP oracles
def test_hrp_weights_simplex_and_positive():
    """TRUE-POSITIVE: HRP weights are long-only and sum to 1 on block data."""
    rng = np.random.default_rng(0)
    X, _ = _block_correlated_returns(rng)
    w = hrp_weights(X)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0.0)
    assert np.all(np.isfinite(w))


def test_hrp_diversifies_across_blocks():
    """TRUE-POSITIVE: HRP spreads weight across all correlation blocks rather than
    piling into one — each block gets a comparable share of total weight."""
    rng = np.random.default_rng(1)
    X, labels = _block_correlated_returns(rng, blocks=(20, 20, 20))
    w = hrp_weights(X)
    block_w = np.array([w[labels == b].sum() for b in np.unique(labels)])
    # 3 independent blocks → each should hold a meaningful (>15%) share.
    assert block_w.min() > 0.15, f"block weights {block_w}"
    assert block_w.max() < 0.60, f"one block dominates: {block_w}"


def test_hrp_no_blowup_under_singular_cov():
    """CONTROL/ADVERSARIAL: a near-singular covariance (duplicated assets) does NOT
    produce inf/nan or a degenerate all-in-one weight — the no-inversion guarantee."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((500, 5))
    X = np.hstack([base, base + 1e-9 * rng.standard_normal((500, 5))])  # rank-deficient
    w = hrp_weights(X)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.max() < 0.95, "should not collapse to a single asset"


def test_hrp_inverse_variance_within_block():
    """TRUE-POSITIVE: within an uncorrelated pair, HRP gives MORE weight to the
    lower-variance asset (its inverse-variance bisection)."""
    rng = np.random.default_rng(3)
    lo = 1.0 * rng.standard_normal((4000, 1))
    hi = 4.0 * rng.standard_normal((4000, 1))
    X = np.hstack([lo, hi])
    w = hrp_weights(X)
    assert w[0] > w[1], f"low-var asset should get more weight: {w}"


# ================================================================== CVaR oracles
def test_cvar_underweights_fat_tailed_asset_vs_mean_variance():
    """TRUE-POSITIVE: with one crash-prone (fat-tailed, same mean/vol) asset, CVaR-opt
    assigns it LESS weight than a min-variance optimizer does (tail awareness)."""
    rng = np.random.default_rng(10)
    T = 4000
    a = 0.01 * rng.standard_normal(T)                       # benign
    b = 0.01 * rng.standard_normal(T)                       # benign
    # crash-prone: same ~vol but heavy left tail (occasional big losses)
    c = 0.007 * rng.standard_normal(T)
    crash = rng.random(T) < 0.02
    c[crash] -= 0.10
    X = np.column_stack([a, b, c])
    w_cvar, _ = cvar_min_weights(X, alpha=0.95)
    # min-variance baseline (closed form, long-only via clip+renorm)
    cov, _ = _cov_corr(X)
    inv = np.linalg.solve(cov, np.ones(3))
    w_mv = np.clip(inv, 0, None); w_mv /= w_mv.sum()
    assert w_cvar[2] < w_mv[2] - 1e-3, (
        f"CVaR weight on crash asset {w_cvar[2]:.3f} should be < mean-var {w_mv[2]:.3f}")


def test_cvar_recovers_equal_weights_iid_symmetric():
    """CONTROL: with i.i.d. symmetric identical assets, CVaR-opt has no tail reason to
    prefer any one — weights are ~equal."""
    rng = np.random.default_rng(11)
    X = 0.01 * rng.standard_normal((4000, 4))
    w, _ = cvar_min_weights(X, alpha=0.95)
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(w >= -1e-9)
    assert np.max(np.abs(w - 0.25)) < 0.12, f"should be ~equal: {w}"


# ================================================================== Merton oracles
def test_merton_closed_form_exact():
    """TRUE-POSITIVE: given known μ, σ, γ the node returns (μ−r)/(γσ²) exactly."""
    for mu, sigma, gamma, r in [(0.10, 0.20, 3.0, 0.02),
                                (0.05, 0.15, 5.0, 0.0),
                                (0.12, 0.30, 2.0, 0.01)]:
        expected = (mu - r) / (gamma * sigma ** 2)
        assert abs(merton_fraction(mu, sigma, gamma, r) - expected) < 1e-12


def test_merton_monotonic_in_mu_sigma_gamma():
    """CONTROL: fraction rises with μ, falls with σ² and with γ."""
    base = merton_fraction(0.10, 0.20, 3.0, 0.0)
    assert merton_fraction(0.20, 0.20, 3.0, 0.0) > base      # more μ → more
    assert merton_fraction(0.10, 0.40, 3.0, 0.0) < base      # more σ → less
    assert merton_fraction(0.10, 0.20, 6.0, 0.0) < base      # more γ → less


def test_merton_node_recovers_estimated_fraction():
    """TRUE-POSITIVE: on a single simulated asset the node's reported fraction equals
    (μ̂−r)/(γσ̂²) from its own μ,σ estimates."""
    rng = np.random.default_rng(20)
    x = 0.001 + 0.02 * rng.standard_normal((5000, 1))
    b = create("merton_alloc").process(x)
    mu, sigma, gamma = b.payload["mu"][0], b.payload["sigma"][0], b.payload["gamma"]
    expected = (mu - b.payload["r"]) / (gamma * sigma ** 2)
    assert abs(b.payload["fraction"] - expected) < 1e-9


# ================================================================== MPC oracles
def _mean_reverting_signal(rng, T=300, phi=0.9):
    """An AR(1) mean-reverting expected-return signal path."""
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + 0.3 * rng.standard_normal()
    return s


def test_mpc_tracks_but_smooths_signal():
    """TRUE-POSITIVE: MPC follows the signal (positive correlation) yet has LESS
    turnover than naive sign-following on a mean-reverting signal + quadratic cost."""
    rng = np.random.default_rng(30)
    s = _mean_reverting_signal(rng)
    mpc = mpc_position_path(s, risk=0.5, cost=2.0)
    naive = naive_position_path(s)
    corr = np.corrcoef(mpc, s)[0, 1]
    assert corr > 0.5, f"MPC should track the signal, corr={corr:.2f}"
    assert _turnover(mpc) < _turnover(naive), "MPC must smooth (less turnover)"


def test_mpc_smoothing_increases_with_cost():
    """CONTROL/ABLATION: raising the transaction-cost weight monotonically REDUCES
    turnover (more smoothing)."""
    rng = np.random.default_rng(31)
    s = _mean_reverting_signal(rng)
    turns = [_turnover(mpc_position_path(s, risk=0.5, cost=c))
             for c in (0.1, 1.0, 5.0, 20.0)]
    assert all(turns[i] >= turns[i + 1] - 1e-9 for i in range(len(turns) - 1)), turns
    assert turns[-1] < turns[0], f"high cost should smooth more: {turns}"


# ================================================================== interlingua
@pytest.mark.parametrize("nt", ["hrp_alloc", "cvar_opt", "merton_alloc", "mpc_position"])
def test_belief_conforms_interlingua(nt):
    rng = np.random.default_rng(99)
    X = 0.01 * rng.standard_normal((400, 8))
    b = create(nt).process(X)
    assert validate_belief(b) == [], f"{nt} belief off-contract"
    assert b.type in ("decision", "action")
    assert b.source == nt


@pytest.mark.parametrize("nt", ["hrp_alloc", "cvar_opt"])
def test_allocator_weights_in_payload_sum_to_one(nt):
    rng = np.random.default_rng(98)
    X = 0.01 * rng.standard_normal((400, 6))
    b = create(nt).process(X)
    w = np.array(b.payload["weights"])
    assert abs(w.sum() - 1.0) < 1e-6 and np.all(w >= -1e-9)


# ================================================================== REAL-data verdicts
def _walk_forward_oos_var(X, weight_fn, train=500, step=120):
    """Mean OOS portfolio variance: fit weights on a trailing window, measure the
    realized variance of the held-out next block. Lower = better risk control."""
    T = X.shape[0]
    oos = []
    t = train
    while t + step <= T:
        w = weight_fn(X[t - train:t])
        port = X[t:t + step] @ w
        oos.append(np.var(port))
        t += step
    return float(np.mean(oos)) if oos else float("nan")


def _equal_weights(Xtr):
    return np.ones(Xtr.shape[1]) / Xtr.shape[1]


def _inverse_variance(Xtr):
    v = 1.0 / np.clip(Xtr.var(axis=0), 1e-18, None)
    return v / v.sum()


@pytest.mark.skipif(not os.path.exists(_UNIVERSE), reason="universe data absent")
def test_hrp_real_oos_variance_vs_baselines():
    """REAL: HRP should reduce (or at least match) OOS portfolio variance vs
    equal-weight on data/universe/returns. Reported either way (honest verdict)."""
    X = np.load(_UNIVERSE)["returns"].astype(float)
    v_hrp = _walk_forward_oos_var(X, hrp_weights)
    v_eq = _walk_forward_oos_var(X, _equal_weights)
    v_iv = _walk_forward_oos_var(X, _inverse_variance)
    print(f"\n[HRP universe OOS var] hrp={v_hrp:.3e} equal={v_eq:.3e} invvar={v_iv:.3e}")
    assert np.isfinite(v_hrp)
    # KEEP-as-utility bar: HRP beats naive equal-weight on OOS risk.
    assert v_hrp <= v_eq * 1.05, f"HRP {v_hrp:.3e} not competitive vs equal {v_eq:.3e}"


@pytest.mark.skipif(not os.path.exists(_UNIVERSE_DAILY), reason="daily data absent")
def test_hrp_adversarial_multiregime():
    """ADVERSARIAL: HRP stays finite/valid across the 2022 bear→2024 rally span."""
    X = np.load(_UNIVERSE_DAILY)["returns"].astype(float)
    v_hrp = _walk_forward_oos_var(X, hrp_weights, train=300, step=60)
    v_eq = _walk_forward_oos_var(X, _equal_weights, train=300, step=60)
    print(f"\n[HRP daily/multiregime OOS var] hrp={v_hrp:.3e} equal={v_eq:.3e}")
    assert np.isfinite(v_hrp)


@pytest.mark.skipif(not os.path.exists(_UNIVERSE), reason="universe data absent")
def test_cvar_real_oos_tail_loss_vs_baselines():
    """REAL: CVaR-opt should reduce OOS tail loss (mean of worst-5% portfolio
    returns) vs equal-weight on universe returns."""
    X = np.load(_UNIVERSE)["returns"].astype(float)

    def oos_tail(weight_fn, train=500, step=120):
        T = X.shape[0]; t = train; tails = []
        while t + step <= T:
            w = weight_fn(X[t - train:t])
            port = X[t:t + step] @ w
            q = np.quantile(port, 0.05)
            tails.append(port[port <= q].mean())
            t += step
        return float(np.mean(tails))

    cvar_w = lambda Xtr: cvar_min_weights(Xtr, alpha=0.95)[0]
    tl_cvar = oos_tail(cvar_w)
    tl_eq = oos_tail(_equal_weights)
    print(f"\n[CVaR universe OOS tail-loss] cvar={tl_cvar:.3e} equal={tl_eq:.3e}")
    # less negative tail loss = better. KEEP-as-utility bar: beats equal-weight.
    assert tl_cvar >= tl_eq - 1e-4, f"CVaR tail {tl_cvar:.3e} worse than equal {tl_eq:.3e}"
