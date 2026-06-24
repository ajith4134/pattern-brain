"""TIER-3 decision/control foundations (Bellman / HJB / Lagrange / log-utility).

These three nodes are the *mathematical ancestors* of the decision & sizing
machinery the bank already uses (Kelly, Markowitz, optimal-stopping trails). They
are CONTROL UTILITIES — sizing / allocation / exit policies — not 1-step
forecasters; judge them by DESIGN-JOB correctness (does the optimum match the
known closed form / does the sizing reduce variance), exactly as RMT and
signatures are judged (Rule 34). Each emits the generic ``decision`` belief
(interlingua-conformant: optional keys ``action``/``vote``/``z``/``predictions``);
none invents a new belief type.

Domain-agnostic (Rule 23): every node consumes a generic finite ``(T, D)`` array
and knows nothing about candles or order books.

Numerical policy (CLAUDE.md Phase 3): float64 throughout; ``np.linalg.solve`` is
used for every linear system (never ``inv``); the closed forms are derived in the
docstrings so the code is checkable against the spec.

Nodes
-----
* ``log_utility`` (LogUtilitySizingNode) — Bernoulli expected/log-utility position
  sizing, the ancestor of Kelly: f* = mu / sigma^2, clipped to [-1, 1].
* ``lagrange_opt`` (LagrangeMeanVarianceNode) — Lagrangian constrained
  mean-variance allocation (the math under Markowitz), closed-form via ``solve``.
* ``hjb_control`` (HJBControlNode) — Bellman dynamic-programming / discretized HJB
  optimal hold-vs-exit policy via value iteration on a z-score state grid.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

# Below this |f*| / |signal| the decision is "flat" / "hold" rather than a
# directional bet (avoids labelling pure noise as a position).
_FLAT_EPS = 1e-3


def _cross_section_mean(X: np.ndarray) -> np.ndarray:
    """Collapse a (T, D) matrix to a 1-D (T,) series by the cross-feature mean.

    The same convention the existing decision nodes use (decision.py): a node that
    needs a single series treats the equal-weighted cross-section as that series.
    """
    return np.asarray(X, dtype=float).mean(axis=1)


# ===========================================================================
# 1. log_utility — growth-optimal (fractional-Kelly) position sizing
# ===========================================================================
def kelly_fraction(returns: np.ndarray, clip: float = 1.0) -> float:
    """Growth-optimal investment fraction f* = mu / sigma^2, clipped to [-clip, clip].

    Derivation
    ----------
    For a one-period return ``r`` and an invested fraction ``f``, expected
    log-utility (the Kelly criterion's objective, the ancestor of which is the
    Bernoulli log-utility of wealth) is

        E[log(1 + f r)] ≈ f mu − (1/2) f^2 sigma^2     (2nd-order expansion)

    Maximising over ``f`` gives d/df = mu − f sigma^2 = 0, hence the
    growth-optimal fraction

        f* = mu / sigma^2.

    Parameters
    ----------
    returns : array-like, shape (T,)
        Finite one-period returns (any scale). ``sigma^2`` uses the sample
        variance (``ddof=1``).
    clip : float, default 1.0
        Symmetric leverage bound; the returned fraction lies in [-clip, clip].

    Returns
    -------
    float
        f* in [-clip, clip]. Returns 0.0 when the variance is non-positive
        (constant series) — no edge can be estimated, so no position is taken.

    Post-conditions
    ---------------
    ``-clip <= f* <= clip``; ``sign(f*) == sign(mu)`` whenever ``sigma^2 > 0``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    mu = float(r.mean())
    var = float(r.var(ddof=1)) if r.size > 1 else 0.0
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    f = mu / var
    return float(np.clip(f, -clip, clip))


@register
class LogUtilitySizingNode(Node):
    """Bernoulli expected/log-utility sizing — the growth-optimal (fractional-Kelly)
    fraction f* = mu/sigma^2 from the window's return estimates, clipped to [-1, 1].

    Emits a ``decision`` belief: ``action`` (long/flat/short by the sign of f*) and
    ``z`` = f* (the sizing fraction). Confidence = |f*| (how strong the sized bet is).
    """

    layer = "decision"
    node_type = "log_utility"
    cost = "low"

    def __init__(self, clip: float = 1.0, **kw):
        super().__init__(clip=clip, **kw)
        self.clip = float(clip)

    def _predict(self, X: np.ndarray) -> Belief:
        series = _cross_section_mean(X)
        f = kelly_fraction(series, clip=self.clip)
        if f > _FLAT_EPS:
            action = "long"
        elif f < -_FLAT_EPS:
            action = "short"
        else:
            action = "flat"
        return Belief("decision", {"action": action, "z": f}, abs(f), self.name)


# ===========================================================================
# 2. lagrange_opt — Lagrangian constrained mean-variance allocation
# ===========================================================================
def lagrange_mean_variance_weights(cov: np.ndarray, mu: np.ndarray,
                                   risk_aversion_lambda: float = 1.0) -> np.ndarray:
    """Closed-form mean-variance allocation under a full-investment constraint.

    Problem (the math under Markowitz)::

        minimise   w^T Σ w − λ · w^T μ      subject to   1^T w = 1.

    Lagrangian ``L = w^T Σ w − λ w^T μ − 2γ(1^T w − 1)`` (the ``2`` keeps the
    algebra clean). Stationarity ``∂L/∂w = 0`` gives

        2 Σ w − λ μ − 2γ 1 = 0   ⇒   w = Σ^{-1}( (1/2)λ μ + γ 1 ).

    Imposing ``1^T w = 1`` solves the multiplier in closed form. With
    ``a = Σ^{-1}1`` and ``b = Σ^{-1}μ`` (obtained by ``solve``, never ``inv``)::

        γ = ( 1 − (1/2)λ · 1^T b ) / ( 1^T a ),
        w = (1/2)λ · b + γ · a.

    ``λ = 0`` recovers the global minimum-variance portfolio
    ``w = Σ^{-1}1 / (1^T Σ^{-1}1)``.

    Parameters
    ----------
    cov : array-like, shape (n, n)
        Symmetric positive-definite covariance. Solved with ``np.linalg.solve``;
        a singular ``Σ`` falls back to the pseudo-inverse (flagged numerically by
        the LinAlgError path), never explicit inversion of a well-posed system.
    mu : array-like, shape (n,)
        Expected-return (signal) vector.
    risk_aversion_lambda : float, default 1.0
        Return-tilt strength ``λ ≥ 0``. 0 = pure minimum-variance.

    Returns
    -------
    np.ndarray, shape (n,)
        Portfolio weights summing to 1.

    Post-conditions
    ---------------
    ``w.sum() == 1`` (to numerical tolerance). For ``n == 1`` returns ``[1.0]``.
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float).ravel()
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    one = np.ones(n)

    def _solve(rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(cov, rhs)          # stable: solve, not inv
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov) @ rhs          # ill-conditioned fallback

    a = _solve(one)                                   # Σ^{-1} 1
    b = _solve(mu)                                     # Σ^{-1} μ
    A = float(one @ a)
    if abs(A) < 1e-12:                                 # degenerate Σ^{-1}1 — equal weight
        return one / n
    B = float(one @ b)
    gamma = (1.0 - 0.5 * risk_aversion_lambda * B) / A
    w = 0.5 * risk_aversion_lambda * b + gamma * a
    s = w.sum()
    return w / s if abs(s) > 1e-12 else one / n


@register
class LagrangeMeanVarianceNode(Node):
    """Lagrangian constrained mean-variance allocation (the math under Markowitz).

    Estimates ``μ`` (column means) and ``Σ`` (sample covariance) from the window,
    then solves ``min wᵀΣw − λ·wᵀμ s.t. Σwᵢ=1`` via the closed-form Lagrange
    multiplier (``np.linalg.solve``). A 1-D input degrades gracefully to a single
    asset (``w = [1]``). Emits a ``decision`` belief carrying the ``weights`` in the
    payload; ``action`` is set by the dominant-weight asset's drift sign.
    """

    layer = "decision"
    node_type = "lagrange_opt"
    cost = "low"

    def __init__(self, risk_aversion_lambda: float = 1.0, **kw):
        super().__init__(risk_aversion_lambda=risk_aversion_lambda, **kw)
        self.risk_aversion_lambda = float(risk_aversion_lambda)

    def _predict(self, X: np.ndarray) -> Belief:
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        mu = X.mean(axis=0)
        if n == 1:
            w = np.array([1.0])
            drift = float(mu[0])
        else:
            cov = np.cov(X, rowvar=False)
            # guard a (near-)singular sample covariance with a tiny ridge
            cov = cov + 1e-10 * np.eye(n)
            w = lagrange_mean_variance_weights(cov, mu,
                                               risk_aversion_lambda=self.risk_aversion_lambda)
            drift = float(w @ mu)                      # expected portfolio drift
        if drift > _FLAT_EPS:
            action = "long"
        elif drift < -_FLAT_EPS:
            action = "short"
        else:
            action = "flat"
        conf = float(np.tanh(abs(drift) / (abs(X).mean() + 1e-12)))
        return Belief("decision",
                      {"action": action, "weights": w.tolist(), "z": drift},
                      conf, self.name)


# ===========================================================================
# 3. hjb_control — Bellman / discretized-HJB optimal hold-vs-exit policy
# ===========================================================================
def value_iterate_exit_or_hold(
    grid: np.ndarray,
    kappa: float,
    gamma: float,
    hold_reward: Callable[[float], float],
    exit_reward: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Value iteration for a 2-action (hold / exit) control on a discretized state.

    This is the discrete Bellman form of the HJB optimal-stopping equation: on a
    grid of state values (a z-score grid), the optimal value satisfies

        V(s) = max( exit_reward,                       # stop now (absorbing)
                    hold_reward(s) + γ · V(f(s)) )      # continue one step

    where the state transition ``f(s) = (1 − κ) s`` is deterministic
    mean-reversion toward the equilibrium 0 with reversion rate ``κ ∈ (0, 1]``.
    ``V(f(s))`` is read off the grid by nearest-neighbour lookup. Iterating the
    Bellman operator is a γ-contraction, so it converges to the unique fixed point;
    the returned ``residual`` is the final ``max_s |V_{k+1}(s) − V_k(s)|`` and goes
    to 0.

    Parameters
    ----------
    grid : array-like, shape (G,)
        Discretized state values (e.g. ``np.linspace(-3, 3, 61)`` z-scores).
    kappa : float in (0, 1]
        Mean-reversion rate of the deterministic transition ``s -> (1−κ) s``.
    gamma : float in [0, 1)
        Discount factor.
    hold_reward : callable float -> float
        Per-step reward for holding at state ``s``.
    exit_reward : float, default 0.0
        Terminal reward for exiting (absorbing).
    max_iter : int
        Iteration cap.
    tol : float
        Convergence tolerance on the Bellman residual.

    Returns
    -------
    (V, policy, residual)
        ``V`` : np.ndarray (G,) — the converged value function.
        ``policy`` : np.ndarray (G,) of {"hold", "exit"} — the optimal action per state.
        ``residual`` : float — the final Bellman residual (→ 0 at convergence).

    Pre-conditions
    --------------
    ``0 < kappa <= 1``; ``0 <= gamma < 1``; ``grid`` is sorted and non-empty.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    if grid.size == 0:
        raise ValueError("grid must be non-empty")
    if not (0.0 < kappa <= 1.0):
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    rewards = np.array([float(hold_reward(float(s))) for s in grid])
    # Precompute the next-state grid index for each state (deterministic transition).
    nxt = (1.0 - kappa) * grid
    nxt_idx = np.abs(grid[:, None] - nxt[None, :]).argmin(axis=0)  # nearest grid point

    V = np.zeros(grid.size)
    residual = float("inf")
    for _ in range(max_iter):
        hold_value = rewards + gamma * V[nxt_idx]
        V_new = np.maximum(exit_reward, hold_value)
        residual = float(np.max(np.abs(V_new - V)))
        V = V_new
        if residual < tol:
            break
    hold_value = rewards + gamma * V[nxt_idx]
    policy = np.where(hold_value >= exit_reward, "hold", "exit")
    return V, policy, residual


@register
class HJBControlNode(Node):
    """Bellman dynamic-programming / discretized-HJB optimal hold-vs-exit control.

    Standardizes the window's cross-section mean to a z-score, discretizes the
    z-axis into a grid, and runs value iteration (``value_iterate_exit_or_hold``)
    on a mean-reversion reward (holding far from equilibrium is penalised; the
    transition pulls the state back toward 0). The optimal action for the CURRENT
    z-state is emitted as a ``decision`` belief (``action`` ∈ {hold, exit}).

    This single node covers both the Bellman (TIER-3) and HJB (D10) concepts: the
    discrete value-iteration *is* the HJB optimal-stopping equation on a grid.
    """

    layer = "decision"
    node_type = "hjb_control"
    cost = "low"

    def __init__(self, n_grid: int = 41, z_max: float = 3.0, kappa: float = 0.3,
                 gamma: float = 0.95, penalty: float = 1.0, **kw):
        super().__init__(n_grid=n_grid, z_max=z_max, kappa=kappa, gamma=gamma,
                         penalty=penalty, **kw)
        self.n_grid = int(n_grid)
        self.z_max = float(z_max)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.penalty = float(penalty)

    def _predict(self, X: np.ndarray) -> Belief:
        series = _cross_section_mean(X)
        mu = float(series.mean())
        sd = float(series.std()) + 1e-12
        z_now = float((series[-1] - mu) / sd)
        z_clipped = float(np.clip(z_now, -self.z_max, self.z_max))

        grid = np.linspace(-self.z_max, self.z_max, self.n_grid)
        # holding pays the (negative) mean-reversion penalty; exiting pays 0.
        V, policy, residual = value_iterate_exit_or_hold(
            grid=grid, kappa=self.kappa, gamma=self.gamma,
            hold_reward=lambda s, p=self.penalty: -p * s * s, exit_reward=0.0,
            max_iter=1000, tol=1e-10)
        i = int(np.abs(grid - z_clipped).argmin())
        action = str(policy[i])
        # confidence = how decisively hold beats exit (or vice versa) at this state.
        hold_value = -self.penalty * z_clipped * z_clipped + self.gamma * float(
            V[int(np.abs(grid - (1.0 - self.kappa) * z_clipped).argmin())])
        margin = abs(hold_value - 0.0)
        conf = float(np.tanh(margin))
        return Belief("decision",
                      {"action": action, "z": z_now,
                       "value": float(V[i]), "bellman_residual": residual},
                      conf, self.name)
