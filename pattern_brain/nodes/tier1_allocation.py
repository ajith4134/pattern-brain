"""TIER-1 ALLOCATION / CONTROL nodes (CONCEPT_EQUATION_BANK.md), built oracle-first.

Light-stack only (numpy/scipy/sklearn), each behind the generic Node interface
(Rule 23) and emitting an existing v0.1 interlingua belief type ('decision'/'action').
Domain-agnostic: a generic (T, D) cross-section / return matrix in, a Belief out —
no candle/orderbook knowledge. These are ALLOCATORS / SIZERS / CONTROLLERS, judged
by correctness + risk reduction (KEEP-as-utility), not by standalone Sharpe.

Built:
  * ``hrp_alloc``    — Hierarchical Risk Parity (Lopez de Prado 2016): hierarchical
    clustering of the correlation matrix → quasi-diagonalization → recursive
    bisection → long-only weights summing to 1. NO matrix inversion ⇒ robust under
    ill-conditioned / near-singular covariance.
  * ``cvar_opt``     — Conditional Value-at-Risk optimization (Rockafellar-Uryasev
    1999 LP reformulation via scipy.optimize.linprog). Minimizes CVaR_α of the
    long-only portfolio — tail-aware, so it under-weights crash-prone assets.
  * ``merton_alloc`` — Merton (1969) optimal risky fraction w* = (μ−r)/(γσ²).
    Closed-form sizing per asset from estimated μ, σ.
  * ``mpc_position`` — Model Predictive Control / LQG position sizing: a rolling
    finite-horizon QP minimizing −expected_pnl + risk·position² + cost·Δposition²
    given a predicted signal path. Tracks the signal but SMOOTHS it (less turnover);
    smoothing grows with the transaction-cost weight.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

from ..belief import Belief
from ..node import Node
from ..registry import register


# ====================================================================== shared
def _cov_corr(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample covariance and correlation of a (T, D) return matrix (float64).

    Variances are floored strictly positive so a constant/degenerate column does
    not produce a zero on the diagonal (which would break the distance metric and
    the inverse-variance weighting downstream).
    """
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov).astype(float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return cov, corr


# ====================================================================== HRP
def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado's correlation distance d_ij = sqrt(0.5 * (1 - ρ_ij)).

    ρ=1 ⇒ d=0 (identical), ρ=−1 ⇒ d=1 (anti-correlated, maximally distant). A
    proper metric on the correlation matrix, the input to hierarchical clustering.
    """
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return 0.5 * (dist + dist.T)            # enforce exact symmetry for squareform


def quasi_diagonal_order(link: np.ndarray, n: int) -> list[int]:
    """Recover the leaf order of a SciPy linkage so correlated assets sit adjacent.

    Walks the linkage tree replacing each merged-cluster id (>= n) by its two
    children until only original leaves (< n) remain — the quasi-diagonalization
    step of HRP. Returns the ordered list of leaf indices.
    """
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n:
        new: list[int] = []
        for idx in order:
            if idx < n:
                new.append(idx)
            else:
                row = link[idx - n]
                new.extend([int(row[0]), int(row[1])])
        order = new
    return order


def _inverse_variance_weights(cov_block: np.ndarray) -> np.ndarray:
    """Inverse-variance ("naive risk parity") weights for one cluster block."""
    ivp = 1.0 / np.clip(np.diag(cov_block), 1e-18, None)
    return ivp / ivp.sum()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Variance of an inverse-variance-weighted sub-portfolio over ``items``."""
    sub = cov[np.ix_(items, items)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """HRP recursive bisection: split the quasi-diagonal order in half repeatedly,
    allocating risk between the two halves inversely to their cluster variance.

    No matrix inversion anywhere — only diagonal (inverse-variance) terms — which
    is exactly why HRP is robust to an ill-conditioned/near-singular covariance.
    Returns weights aligned to the ORIGINAL asset indices, summing to 1.
    """
    n = cov.shape[0]
    w = np.ones(n, dtype=float)
    clusters = [list(order)]
    while clusters:
        nxt: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right + 1e-18)   # less risk → more weight
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    return w / w.sum()


def hrp_weights(X: np.ndarray) -> np.ndarray:
    """Full Hierarchical Risk Parity pipeline on a (T, D) return matrix.

    corr/cov → correlation distance → single-linkage clustering →
    quasi-diagonalization → recursive bisection → long-only weights summing to 1.
    Degenerate D∈{0,1} returns the trivial weight vector.
    """
    cov, corr = _cov_corr(X)
    n = cov.shape[0]
    if n <= 1:
        return np.ones(n, dtype=float)
    dist = correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    order = quasi_diagonal_order(link, n)
    return recursive_bisection(cov, order)


@register
class HRPAllocNode(Node):
    """Hierarchical Risk Parity allocator (Lopez de Prado 2016). Input (T, D) is a
    cross-section of asset returns; emits a `decision` whose payload ``weights`` are
    long-only and sum to 1. Diversifies across correlation clusters WITHOUT inverting
    the covariance, so it stays stable where mean-variance blows up. KEEP-as-utility."""

    layer = "decision"
    node_type = "hrp_alloc"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        w = hrp_weights(X)
        n = w.size
        # effective number of bets (inverse Herfindahl) — a diversification readout.
        enb = float(1.0 / np.sum(w ** 2)) if n else 0.0
        conf = float(np.clip(X.shape[0] / 500.0, 0.2, 0.9))
        return Belief(
            type="decision",
            payload={
                "weights": w.tolist(),
                "n_assets": int(n),
                "effective_n_bets": enb,
                "max_weight": float(w.max()) if n else 0.0,
                "method": "hierarchical_risk_parity",
            },
            confidence=conf,
        )


# ====================================================================== CVaR (RU)
def cvar_min_weights(R: np.ndarray, alpha: float = 0.95,
                     mu_target: float | None = None) -> tuple[np.ndarray, float]:
    """Minimum-CVaR long-only weights via the Rockafellar-Uryasev (1999) LP.

    Decision vector z = [w (D), VaR (1), u (T)]. Minimize
        VaR + 1/((1-α)T) Σ_t u_t
    s.t.  u_t ≥ -(R_t · w) - VaR,  u_t ≥ 0,  Σ w = 1,  w ≥ 0,
    where R_t is the return row for scenario t (loss = -R_t·w). The optimum's
    objective value IS the minimized CVaR_α. Returns (weights, cvar). Falls back to
    equal weights if the LP fails to solve.
    """
    R = np.asarray(R, dtype=float)
    T, D = R.shape
    n = D + 1 + T
    c = np.zeros(n)
    c[D] = 1.0                                   # VaR
    c[D + 1:] = 1.0 / ((1.0 - alpha) * T)        # u_t average
    # u_t + R_t·w + VaR >= 0  →  -R_t·w - VaR - u_t <= 0
    A_ub = np.zeros((T, n))
    A_ub[:, :D] = -R
    A_ub[:, D] = -1.0
    A_ub[np.arange(T), D + 1 + np.arange(T)] = -1.0
    b_ub = np.zeros(T)
    A_eq = np.zeros((1, n)); A_eq[0, :D] = 1.0   # Σ w = 1
    b_eq = np.array([1.0])
    if mu_target is not None:                    # optional return floor: μ·w >= target
        mu = R.mean(axis=0)
        row = np.zeros((1, n)); row[0, :D] = -mu
        A_ub = np.vstack([A_ub, row])
        b_ub = np.concatenate([b_ub, [-mu_target]])
    bounds = [(0.0, 1.0)] * D + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        w = np.ones(D) / D
        losses = -R @ w
        var = float(np.quantile(losses, alpha))
        return w, float(losses[losses >= var].mean()) if np.any(losses >= var) else var
    w = np.clip(res.x[:D], 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(D) / D
    return w, float(res.fun)


@register
class CVaROptNode(Node):
    """Conditional-Value-at-Risk minimizing allocator (Rockafellar-Uryasev LP, D1
    risk). Input (T, D) return rows; emits a `decision` with long-only ``weights``
    that minimize CVaR_α (expected loss in the worst (1-α) tail). Tail-aware: it
    under-weights crash-prone assets a mean-variance optimizer would still hold.
    KEEP-as-utility (tail risk control)."""

    layer = "decision"
    node_type = "cvar_opt"
    requires_y = False
    is_transformer = False
    cost = "med"

    _ALPHA = 0.95

    def _predict(self, X: np.ndarray) -> Belief:
        D = X.shape[1]
        if D < 2 or X.shape[0] < 5:
            w = np.ones(D) / max(D, 1)
            return Belief(type="decision",
                          payload={"weights": w.tolist(), "cvar": 0.0,
                                   "alpha": self._ALPHA, "n_assets": int(D),
                                   "note": "needs D>=2 and T>=5"},
                          confidence=0.1)
        w, cvar = cvar_min_weights(X, alpha=self._ALPHA)
        enb = float(1.0 / np.sum(w ** 2))
        conf = float(np.clip(X.shape[0] / 500.0, 0.2, 0.9))
        return Belief(
            type="decision",
            payload={
                "weights": w.tolist(),
                "cvar": cvar,
                "alpha": self._ALPHA,
                "n_assets": int(D),
                "effective_n_bets": enb,
                "method": "cvar_rockafellar_uryasev",
            },
            confidence=conf,
        )


# ====================================================================== Merton
def merton_fraction(mu: float, sigma: float, gamma: float, r: float = 0.0) -> float:
    """Merton (1969) optimal fraction in the risky asset: w* = (μ − r)/(γ σ²).

    μ excess of the risk-free rate r, divided by risk aversion γ times variance σ².
    Rises with the risk premium, falls with variance and risk aversion. σ→0 caps at
    a large finite value rather than dividing by zero.
    """
    var = max(float(sigma) ** 2, 1e-18)
    return float((mu - r) / (max(float(gamma), 1e-9) * var))


def merton_weights(X: np.ndarray, gamma: float, r: float = 0.0) -> np.ndarray:
    """Per-asset Merton fractions for each column of a (T, D) return matrix.

    Estimates μ, σ per column from the sample, applies w*=(μ−r)/(γσ²) elementwise.
    Returned UNNORMALIZED (these are independent risky fractions, not a simplex);
    the node also reports the normalized long-only tilt for convenience.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
    var = np.clip(sigma ** 2, 1e-18, None)
    return (mu - r) / (max(float(gamma), 1e-9) * var)


@register
class MertonAllocNode(Node):
    """Merton optimal risky-fraction sizer (D1 sizing). Input (T, D) returns;
    estimates μ, σ per asset and emits a `decision` carrying the closed-form
    fraction(s) w*=(μ−r)/(γσ²) plus {mu, sigma, gamma}. With D==1 the scalar
    fraction is the headline; with D>1 a per-asset vector + a normalized long-only
    tilt are given. Judged by closed-form correctness. KEEP-as-utility."""

    layer = "decision"
    node_type = "merton_alloc"
    requires_y = False
    is_transformer = False
    cost = "low"

    _GAMMA = 3.0
    _R = 0.0

    def _predict(self, X: np.ndarray) -> Belief:
        gamma, r = self._GAMMA, self._R
        fracs = merton_weights(X, gamma=gamma, r=r)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
        pos = np.clip(fracs, 0.0, None)                 # long-only normalized tilt
        weights = (pos / pos.sum()) if pos.sum() > 0 else np.ones(fracs.size) / fracs.size
        conf = float(np.clip(X.shape[0] / 500.0, 0.2, 0.9))
        return Belief(
            type="decision",
            payload={
                "fraction": float(fracs[0]),
                "fractions": fracs.tolist(),
                "weights": weights.tolist(),
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "gamma": float(gamma),
                "r": float(r),
                "method": "merton_optimal_fraction",
            },
            confidence=conf,
        )


# ====================================================================== MPC / LQG
def mpc_position_path(signal: np.ndarray, risk: float, cost: float,
                      pos_bound: float = 5.0) -> np.ndarray:
    """Finite-horizon MPC/LQG position path for a predicted expected-return signal.

    Minimizes over the whole horizon
        Σ_t [ -signal_t · p_t + risk · p_t² ] + Σ_t cost · (p_t − p_{t-1})²
    a strictly convex quadratic in the position path p (p_{-1}=0). Solving ∇=0 gives
    a tridiagonal linear system H p = signal solved with np.linalg.solve (no inv):
        H = (2·risk) I + cost · L,  L the second-difference (turnover) operator.
    Larger ``cost`` ⇒ heavier off-diagonal coupling ⇒ a SMOOTHER path (less
    turnover). Positions are clipped to ±pos_bound. Returns the position path.
    """
    s = np.asarray(signal, dtype=float).ravel()
    T = s.size
    if T == 0:
        return s
    L = np.zeros((T, T))                            # turnover (second-difference) operator
    for t in range(T):
        L[t, t] += 2.0
        if t > 0:
            L[t, t - 1] -= 1.0; L[t - 1, t] -= 1.0
    L[T - 1, T - 1] -= 1.0                          # free terminal (no p_T penalty)
    H = 2.0 * risk * np.eye(T) + cost * L
    p = np.linalg.solve(H, s)                       # ∂/∂p [ -s·p + risk p² + cost Δp² ] = 0
    return np.clip(p, -pos_bound, pos_bound)


def naive_position_path(signal: np.ndarray, pos_bound: float = 5.0) -> np.ndarray:
    """Cost-blind benchmark: position proportional to the signal (sign-following)."""
    s = np.asarray(signal, dtype=float).ravel()
    scale = pos_bound / (np.max(np.abs(s)) + 1e-12)
    return np.clip(s * scale, -pos_bound, pos_bound)


def _turnover(path: np.ndarray) -> float:
    """Total absolute change of a position path (p_{-1}=0)."""
    p = np.asarray(path, dtype=float).ravel()
    return float(np.sum(np.abs(np.diff(np.concatenate([[0.0], p])))))


@register
class MPCPositionNode(Node):
    """Model-Predictive-Control / LQG position sizer (D1 control). Column 0 is the
    predicted expected-return signal path; emits a `decision` whose ``positions`` are
    the horizon-optimal target path that tracks the signal while penalizing turnover
    (cost·Δposition²) and exposure (risk·position²). Smoother (lower turnover) than
    naive sign-following; smoothing increases with the cost weight. KEEP-as-utility."""

    layer = "decision"
    node_type = "mpc_position"
    requires_y = False
    is_transformer = False
    cost = "low"

    _RISK = 0.5
    _COST = 2.0
    _MAX_N = 2000

    def _predict(self, X: np.ndarray) -> Belief:
        s = X[:, 0].astype(float)
        if s.size > self._MAX_N:
            s = s[-self._MAX_N:]
        # standardize the raw signal so the risk/cost weights are scale-stable.
        sd = float(np.std(s)) or 1.0
        s_std = (s - np.mean(s)) / sd
        path = mpc_position_path(s_std, risk=self._RISK, cost=self._COST)
        naive = naive_position_path(s_std)
        conf = float(np.clip(s.size / 500.0, 0.2, 0.9))
        return Belief(
            type="decision",
            payload={
                "positions": path.tolist(),
                "turnover": _turnover(path),
                "naive_turnover": _turnover(naive),
                "risk_weight": self._RISK,
                "cost_weight": self._COST,
                "mean_abs_position": float(np.mean(np.abs(path))),
                "method": "mpc_lqg_position",
            },
            confidence=conf,
        )
