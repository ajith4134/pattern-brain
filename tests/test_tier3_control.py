"""Oracle-first tests for the TIER-3 decision/control nodes (oracle BEFORE code).

Per CLAUDE.md Phase-4 / HIGH_QUALITY_CODE_INSTRUCTIONS #1: write the executable
numerical-equivalence check FIRST, then code until it passes. These tests pin the
closed-form contract for each node against an independent reference:

  * log_utility   — f* = mu / sigma^2 (fractional-Kelly), sign logic, clipping.
  * lagrange_opt  — the analytic mean-variance Lagrange solution on a known (Sigma, mu).
  * hjb_control   — value iteration converges (Bellman residual -> 0) and picks the
                    obviously-optimal action on a toy reward.

Plus a real-data sanity check (Rule 32/34): these are SIZING/ALLOCATION/CONTROL
utilities, judged by DESIGN-JOB correctness (does the sizing reduce variance / does
the allocation match the analytic optimum), not by 1-step alpha. Domain-agnostic
(Rule 23): generic (T, D) input, no candle/orderbook knowledge.

Run: PYTHONPATH=. python3 -m pytest tests/test_tier3_control.py -q
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from pattern_brain.belief import Belief
from pattern_brain.nodes import tier3_control as t3
from pattern_brain.registry import create, _REGISTRY


# --------------------------------------------------------------------------- #
# log_utility — f* = mu / sigma^2 (fractional Kelly), clipped to [-1, 1].
# --------------------------------------------------------------------------- #
def _kelly_fraction_oracle(returns: np.ndarray, clip: float = 1.0) -> float:
    """Independent reference: growth-optimal fraction f* = mu / sigma^2, clipped."""
    r = np.asarray(returns, dtype=float).ravel()
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    f = mu / var if var > 0 else 0.0
    return float(np.clip(f, -clip, clip))


def test_log_utility_matches_kelly_oracle_positive_drift():
    rng = np.random.default_rng(0)
    # mu > 0: a positive-drift series should size LONG with f* = mu/sigma^2.
    r = rng.normal(0.02, 0.05, size=400)
    X = r.reshape(-1, 1)
    node = create("log_utility")
    belief = node.predict(X)
    f_oracle = _kelly_fraction_oracle(r)
    assert belief.type == "decision"
    assert belief.payload["action"] == "long"
    assert belief.payload["z"] == pytest.approx(f_oracle, abs=1e-9)
    assert f_oracle > 0


def test_log_utility_matches_kelly_oracle_negative_drift():
    rng = np.random.default_rng(1)
    r = rng.normal(-0.02, 0.05, size=400)
    X = r.reshape(-1, 1)
    belief = create("log_utility").predict(X)
    f_oracle = _kelly_fraction_oracle(r)
    assert belief.payload["action"] == "short"
    assert belief.payload["z"] == pytest.approx(f_oracle, abs=1e-9)
    assert f_oracle < 0


def test_log_utility_clips_to_unit_interval():
    # Tiny variance, non-trivial mean -> raw f* explodes; node must clip to [-1, 1].
    r = np.full(200, 0.10) + np.array([1e-6, -1e-6] * 100)
    belief = create("log_utility").predict(r.reshape(-1, 1))
    assert -1.0 <= belief.payload["z"] <= 1.0
    assert belief.payload["action"] == "long"


def test_log_utility_flat_on_exactly_zero_drift():
    # Exactly-zero-mean series -> f* = mu/sigma^2 = 0 -> flat action. (Note: with a
    # small sigma even a tiny non-zero mu correctly implies large Kelly leverage and
    # clips to +/-1 — that is the criterion working, not a flat bet.)
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.05, size=2000)
    r = r - r.mean()                           # force mu == 0 exactly
    belief = create("log_utility").predict(r.reshape(-1, 1))
    assert belief.payload["z"] == pytest.approx(0.0, abs=1e-9)
    assert belief.payload["action"] == "flat"
    # multi-asset input: f* computed on the cross-sectional mean series.
    X = rng.normal(0.0, 0.05, size=(300, 4))
    b2 = create("log_utility").predict(X)
    assert b2.type == "decision" and "z" in b2.payload


# --------------------------------------------------------------------------- #
# lagrange_opt — closed-form mean-variance Lagrange solution.
# min wᵀΣw − λ·wᵀμ s.t. 1ᵀw = 1.
# Stationarity: 2Σw − λμ − 2γ1 = 0  ->  w = Σ⁻¹(½λμ + γ1),
# γ chosen so 1ᵀw = 1:  γ = (1 − ½λ·(1ᵀΣ⁻¹μ)) / (1ᵀΣ⁻¹1).
# --------------------------------------------------------------------------- #
def _lagrange_weights_oracle(cov, mu, lam):
    """Independent closed-form reference (uses inv intentionally as the oracle)."""
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    inv = np.linalg.inv(cov)
    one = np.ones(cov.shape[0])
    A = float(one @ inv @ one)
    B = float(one @ inv @ mu)
    gamma = (1.0 - 0.5 * lam * B) / A
    w = inv @ (0.5 * lam * mu + gamma * one)
    return w


def test_lagrange_opt_matches_closed_form_on_known_problem():
    # A known SPD covariance + return vector; compare node weights to the oracle.
    cov = np.array([[0.10, 0.02, 0.00],
                    [0.02, 0.08, 0.01],
                    [0.00, 0.01, 0.05]])
    mu = np.array([0.05, 0.03, 0.04])
    lam = 1.0
    w_oracle = _lagrange_weights_oracle(cov, mu, lam)
    w_node = t3.lagrange_mean_variance_weights(cov, mu, risk_aversion_lambda=lam)
    assert np.allclose(w_node, w_oracle, atol=1e-9)
    assert w_node.sum() == pytest.approx(1.0, abs=1e-9)   # post-condition: budget


def test_lagrange_opt_lambda_zero_is_min_variance():
    # lambda=0 removes the return term -> pure global minimum-variance portfolio
    # w = Σ⁻¹1 / (1ᵀΣ⁻¹1).
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    mu = np.array([0.02, 0.05])
    w = t3.lagrange_mean_variance_weights(cov, mu, risk_aversion_lambda=0.0)
    inv = np.linalg.inv(cov)
    one = np.ones(2)
    w_minvar = (inv @ one) / (one @ inv @ one)
    assert np.allclose(w, w_minvar, atol=1e-10)


def test_lagrange_opt_degrades_to_single_asset():
    # 1-D input: nothing to allocate across -> a single-asset decision.
    rng = np.random.default_rng(3)
    X = rng.normal(0.01, 0.05, size=(300, 1))
    belief = create("lagrange_opt").predict(X)
    assert belief.type == "decision"
    assert belief.payload["action"] in ("long", "short", "flat")
    w = np.asarray(belief.payload["weights"], dtype=float)
    assert w.shape == (1,)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_lagrange_opt_emits_decision_with_allocation():
    rng = np.random.default_rng(4)
    X = rng.normal(0.0, 0.02, size=(400, 3))
    belief = create("lagrange_opt").predict(X)
    assert belief.type == "decision"
    w = np.asarray(belief.payload["weights"], dtype=float)
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert belief.payload["action"] in ("long", "short", "flat")


# --------------------------------------------------------------------------- #
# hjb_control — Bellman value iteration on a discretized state (z-score grid).
# --------------------------------------------------------------------------- #
def test_hjb_value_iteration_converges():
    # The fixed-point V must satisfy the Bellman equation: residual -> 0.
    grid = np.linspace(-3.0, 3.0, 41)
    # Toy reward: holding pays the mean-reverting pull toward 0 (|z| small good),
    # exiting pays 0. Transition: deterministic shrink toward 0 (kappa).
    V, policy, residual = t3.value_iterate_exit_or_hold(
        grid=grid, kappa=0.3, gamma=0.95, hold_reward=lambda z: -z * z,
        exit_reward=0.0, max_iter=2000, tol=1e-10)
    assert residual < 1e-6                       # converged Bellman fixed point


def test_hjb_picks_obviously_optimal_action():
    # With hold_reward = -z^2 (penalty grows with |z|) and a mean-reverting
    # transition, far-from-equilibrium states should EXIT (avoid the penalty),
    # near-equilibrium states should HOLD (reward ~0, future gain from reversion).
    grid = np.linspace(-3.0, 3.0, 61)
    V, policy, residual = t3.value_iterate_exit_or_hold(
        grid=grid, kappa=0.3, gamma=0.95, hold_reward=lambda z: -z * z,
        exit_reward=0.0, max_iter=2000, tol=1e-12)
    # near 0 -> hold; the extreme grid points -> exit.
    i_center = int(np.argmin(np.abs(grid)))
    assert policy[i_center] == "hold"
    assert policy[0] == "exit" and policy[-1] == "exit"


def test_hjb_control_node_emits_decision():
    rng = np.random.default_rng(5)
    # Mean-reverting series so the control surface is meaningful.
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
    belief = create("hjb_control").predict(x.reshape(-1, 1))
    assert belief.type == "decision"
    assert belief.payload["action"] in ("hold", "exit")
    assert "z" in belief.payload


# --------------------------------------------------------------------------- #
# Registration + Node-contract conformance.
# --------------------------------------------------------------------------- #
def test_all_three_registered():
    for nt in ("log_utility", "lagrange_opt", "hjb_control"):
        assert nt in _REGISTRY, f"{nt} not registered"
        assert _REGISTRY[nt].layer == "decision"


def test_beliefs_conform_to_interlingua_decision_type():
    from pattern_brain.interlingua import validate_belief
    rng = np.random.default_rng(6)
    X = rng.normal(0.0, 0.03, size=(300, 3))
    for nt in ("log_utility", "lagrange_opt", "hjb_control"):
        b = create(nt).predict(X)
        assert isinstance(b, Belief) and b.type == "decision"
        assert validate_belief(b) == []      # conforms to the 'decision' schema


# --------------------------------------------------------------------------- #
# Real-data DESIGN-JOB sanity (Rule 32/34): not 1-step alpha — does the tool do
# its job? log_utility sizing should reduce variance vs full exposure; lagrange
# min-variance allocation should beat equal-weight variance OOS.
# --------------------------------------------------------------------------- #
_SCRATCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "_test_tier3_ctrl")


def _load_panel_returns():
    """Load panel close-to-close log-returns as a (T, n_assets) matrix. Skips if
    the panel is absent (keeps the unit oracle tests runnable anywhere)."""
    from pattern_brain.datasets import available_datasets, load_series
    names = [n for n in available_datasets() if not n.startswith("kraken:")]
    if not names:
        return None, None
    cols = []
    used = []
    for n in names:
        r = load_series(n, target="logreturn").ravel()
        cols.append(r)
        used.append(n)
    m = min(len(c) for c in cols)
    R = np.column_stack([c[-m:] for c in cols])
    return R, used


def test_realdata_log_utility_sizing_reduces_variance():
    R, names = _load_panel_returns()
    if R is None:
        pytest.skip("panel data not present")
    os.makedirs(_SCRATCH, exist_ok=True)
    try:
        # DESIGN JOB: walk-forward Kelly sizing of a single series should produce a
        # sized P&L whose variance is <= full-exposure variance (it scales down when
        # the edge/variance ratio is small). Judge on the design metric, not alpha.
        col = R[:, 0]
        split = len(col) // 2
        node = create("log_utility")
        sized = []
        full = []
        for t in range(split, len(col) - 1):
            window = col[max(0, t - 200):t].reshape(-1, 1)
            f = float(node.predict(window).payload["z"])
            sized.append(f * col[t + 1])         # sized next-step return
            full.append(col[t + 1])              # full-exposure next-step return
        sized = np.asarray(sized)
        full = np.asarray(full)
        # Fractional-Kelly with |f|<=1 cannot increase variance beyond full exposure.
        assert sized.var() <= full.var() + 1e-12
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


def test_realdata_lagrange_min_variance_beats_equal_weight():
    R, names = _load_panel_returns()
    if R is None or R.shape[1] < 3:
        pytest.skip("panel data not present / too few assets")
    os.makedirs(_SCRATCH, exist_ok=True)
    try:
        # DESIGN JOB: min-variance allocation (lambda=0) estimated on TRAIN must
        # deliver LOWER realised portfolio variance OOS than equal-weight (the
        # textbook guarantee of the Markowitz minimum-variance portfolio).
        split = R.shape[0] // 2
        train, test = R[:split], R[split:]
        cov = np.cov(train, rowvar=False)
        w_mv = t3.lagrange_mean_variance_weights(cov, train.mean(axis=0),
                                                 risk_aversion_lambda=0.0)
        n = R.shape[1]
        w_eq = np.ones(n) / n
        var_mv = float((test @ w_mv).var())
        var_eq = float((test @ w_eq).var())
        assert var_mv <= var_eq + 1e-12
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)
