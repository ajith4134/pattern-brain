"""TIER-1 DYNAMICAL / FORECAST / GEOMETRIC nodes (CONCEPT_EQUATION_BANK.md family).

Light-stack only (numpy/scipy/sklearn), each behind the generic Node interface
(Rule 23) and emitting an existing v0.1 interlingua belief type. Domain-agnostic:
a generic (T, D) float64 sequence in, a Belief out — no candle/orderbook knowledge.

Built here:
  * ``koopman_dmd`` — Koopman / Dynamic Mode Decomposition. Delay-embed the series,
    fit the linear operator A on time-shifted snapshots (X' ≈ A X) by least squares,
    eigen-decompose A for modes/eigenvalues, forecast by linear evolution. On a clean
    oscillator it recovers the frequency and decay rate; on noisy returns it loses to
    persistence (a forecaster on returns — project lesson).
  * ``bsts_forecast`` — Bayesian Structural Time Series local-linear-trend state-space
    model run through a light-stack Kalman filter (level + slope). Emits the next-step
    forecast {estimate, std}. Excels at trend extraction on a LEVEL series; degrades to
    ~persistence on noisy returns.
  * ``infogeom_distance`` — Information Geometry / Fisher-Rao distance between rolling
    Gaussian return distributions. Closed-form geodesic distance between consecutive
    windows → a regime-change distance signal that spikes at vol/mean shifts. A utility.
  * ``diffusion_map`` — Diffusion-map manifold embedding. Gaussian-kernel affinity →
    row-normalized Markov matrix → eigen-decomposition → low-dim diffusion coordinates
    of delay-embedded points. Recovers a 1-D manifold's intrinsic ordering and separates
    handwriting-digit classes (pendigits) materially better than chance. A utility.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as _sla
from scipy.spatial.distance import pdist, squareform

from ..belief import Belief
from ..node import Node
from ..registry import register


# ============================================================ Koopman / DMD helpers
def delay_embed(x: np.ndarray, d: int) -> np.ndarray:
    """Hankel delay-embedding: stack d consecutive samples into each column.

    Returns a (d, T-d+1) matrix whose column k is [x_k, ..., x_{k+d-1}]^T. This lifts a
    scalar series into a d-dim observable space so DMD's linear operator can capture
    oscillation/decay a 1-D map cannot.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    d = max(1, min(d, n))
    cols = n - d + 1
    return np.stack([x[k:k + d] for k in range(cols)], axis=1)


def dmd_fit(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the DMD linear operator A on time-shifted snapshot matrices X' ≈ A X.

    Given embedded snapshots H (d, m), split into X = H[:, :-1] and X' = H[:, 1:] and
    solve A = X' X^+ via the least-squares pseudo-inverse (lstsq, not inv). Returns
    (A, eigenvalues_of_A). A captures one-step linear dynamics in the embedded space.
    """
    X = snapshots[:, :-1]
    Xp = snapshots[:, 1:]
    # A^T solves X^T A^T = X'^T  → use lstsq on the transposed system (stable, no inv).
    At, *_ = np.linalg.lstsq(X.T, Xp.T, rcond=None)
    A = At.T
    eigs = np.linalg.eigvals(A)
    return A, eigs


def dmd_forecast(snapshots: np.ndarray, A: np.ndarray, steps: int) -> np.ndarray:
    """Forecast the embedded state forward ``steps`` by repeatedly applying A.

    The last embedded column is the current state; A advances it one step. The forecast
    for the scalar series is the LAST entry of each advanced state (the newest sample).
    """
    state = snapshots[:, -1].copy()
    out = np.empty(steps, dtype=float)
    for k in range(steps):
        state = A @ state
        out[k] = state[-1]
    return out


def dmd_eig_summary(eigs: np.ndarray) -> tuple[list, list, list]:
    """Per-mode (magnitude, angular frequency, growth rate) from discrete eigenvalues.

    For a unit-time step, λ = exp((g + iω)·1): magnitude |λ| (|λ|<1 decays, >1 grows),
    ω = angle(λ) (rad/step → frequency ω/2π cycles/step), growth rate g = ln|λ|.
    """
    mags = np.abs(eigs)
    freqs = np.angle(eigs) / (2.0 * np.pi)            # cycles per step
    growth = np.log(np.maximum(mags, 1e-300))         # ln|λ| per step
    return mags.tolist(), freqs.tolist(), growth.tolist()


@register
class KoopmanDMDNode(Node):
    """Koopman operator / Dynamic Mode Decomposition forecaster (DYNAMICAL).

    Delay-embeds column 0, fits the linear Koopman operator on time-shifted snapshots,
    eigen-decomposes it for oscillation frequencies / decay rates, and forecasts the
    next value by linear evolution. Emits a ``forecast`` with the next value plus the
    discrete-time eigenvalue summary (magnitude/phase) and the number of modes.
    """

    layer = "equation"
    node_type = "koopman_dmd"
    requires_y = False
    is_transformer = False
    cost = "low"

    def __init__(self, name=None, embed_dim: int = 12, **params):
        super().__init__(name=name, embed_dim=embed_dim, **params)
        self.embed_dim = int(embed_dim)

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        if n < 6:
            return Belief(type="forecast",
                          payload={"estimate": float(x[-1]), "n_modes": 0,
                                   "eig_magnitude": [], "eig_freq": [],
                                   "note": "too-short-for-dmd"},
                          confidence=0.1)
        d = max(2, min(self.embed_dim, n // 2))
        H = delay_embed(x, d)
        A, eigs = dmd_fit(H)
        next_val = float(dmd_forecast(H, A, 1)[0])
        mags, freqs, growth = dmd_eig_summary(eigs)
        # confidence: bounded, grows with sample size, capped (returns are barely predictable).
        conf = float(np.clip(n / 1000.0, 0.2, 0.7))
        return Belief(
            type="forecast",
            payload={
                "estimate": next_val,
                "n_modes": int(eigs.size),
                "eig_magnitude": mags,
                "eig_freq": freqs,
                "eig_growth": growth,
                "embed_dim": int(d),
            },
            confidence=conf,
        )


# ============================================================ BSTS / Kalman helpers
def kalman_local_linear_trend(y: np.ndarray, obs_var: float, level_var: float,
                              slope_var: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Kalman filter for the local-linear-trend state-space model.

    State sₜ = [levelₜ, slopeₜ]; transition  levelₜ = levelₜ₋₁ + slopeₜ₋₁ + η,
    slopeₜ = slopeₜ₋₁ + ζ; observation yₜ = levelₜ + ε. Q = diag(level_var, slope_var),
    R = obs_var. Returns (filtered_levels, filtered_slopes, last_level, last_slope_var)
    for one-step-ahead forecasting. Uses np.linalg.solve, never inv.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Hmat = np.array([[1.0, 0.0]])
    Q = np.diag([level_var, slope_var])
    s = np.array([y[0], 0.0])                          # initial state
    P = np.eye(2) * (np.var(y) + 1.0)                  # diffuse-ish prior
    levels = np.empty(n); slopes = np.empty(n)
    for t in range(n):
        s = F @ s                                      # predict
        P = F @ P @ F.T + Q
        resid = y[t] - (Hmat @ s)[0]                   # innovation
        S = (Hmat @ P @ Hmat.T)[0, 0] + obs_var
        K = (P @ Hmat.T).ravel() / S                   # Kalman gain (scalar S)
        s = s + K * resid                              # update
        P = (np.eye(2) - np.outer(K, Hmat)) @ P
        levels[t] = s[0]; slopes[t] = s[1]
    return levels, slopes, s, float(P[0, 0] + obs_var)


def estimate_state_variances(y: np.ndarray) -> tuple[float, float, float]:
    """Crude method-of-moments variances for the LLT model from the series scale.

    obs_var from the high-frequency (second-difference) noise; level/slope process
    variances as small fractions of the series variance. Light, optimizer-free defaults
    that let the filter track a trend without an EM loop.
    """
    y = np.asarray(y, dtype=float).ravel()
    dd = np.diff(y, n=2) if y.size > 2 else np.array([0.0])
    obs_var = max(float(np.var(dd)) / 6.0, 1e-9)       # 2nd-diff var ≈ 6σ²_ε for white obs noise
    scale = max(float(np.var(y)), 1e-9)
    level_var = max(scale * 1e-3, 1e-9)
    slope_var = max(scale * 1e-5, 1e-12)
    return obs_var, level_var, slope_var


@register
class BSTSForecastNode(Node):
    """Bayesian Structural Time Series — local-linear-trend Kalman forecaster (FORECAST).

    Runs column 0 through a local-level + slope Kalman filter, extracting a smooth trend
    and forecasting the next value as level + slope. Emits a ``forecast`` with the point
    estimate and its predictive std. Strong for trend extraction on a LEVEL series;
    degrades to ~persistence on noisy returns.
    """

    layer = "equation"
    node_type = "bsts_forecast"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        y = X[:, 0].astype(float)
        n = y.size
        if n < 4:
            return Belief(type="forecast",
                          payload={"estimate": float(y[-1]), "std": 0.0,
                                   "note": "too-short-for-kalman"},
                          confidence=0.1)
        obs_var, level_var, slope_var = estimate_state_variances(y)
        levels, slopes, s, pred_var = kalman_local_linear_trend(y, obs_var, level_var, slope_var)
        estimate = float(s[0] + s[1])                  # next level = level + slope
        std = float(np.sqrt(max(pred_var, 0.0)))
        conf = float(np.clip(n / 1000.0, 0.2, 0.8))
        return Belief(
            type="forecast",
            payload={
                "estimate": estimate,
                "std": std,
                "level": float(s[0]),
                "slope": float(s[1]),
                "filtered_level": levels.tolist(),
            },
            confidence=conf,
        )


# ============================================================ Information Geometry helpers
def fisher_rao_gaussian(mu1: float, s1: float, mu2: float, s2: float) -> float:
    """Fisher-Rao geodesic distance between two univariate Gaussians N(μ,σ).

    Under the Fisher information metric the Gaussian family is the hyperbolic upper
    half-plane; mapping (μ,σ) → (μ/√2, σ) the geodesic distance is
        d = √2 · arccosh(1 + [ (μ1−μ2)² + 2(σ1−σ2)² ] / (4 σ1 σ2) ).
    It is 0 iff the distributions are identical and grows monotonically as μ or σ
    separate. (Costa-Santos-Hancock 2015 closed form.)
    """
    s1 = max(float(s1), 1e-12); s2 = max(float(s2), 1e-12)
    num = (mu1 - mu2) ** 2 + 2.0 * (s1 - s2) ** 2
    arg = 1.0 + num / (4.0 * s1 * s2)
    return float(np.sqrt(2.0) * np.arccosh(max(arg, 1.0)))


def rolling_gaussian_params(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (mean, std) over non-overlapping consecutive windows of length ``window``."""
    x = np.asarray(x, dtype=float).ravel()
    n_win = x.size // window
    mus = np.empty(n_win); sigmas = np.empty(n_win)
    for i in range(n_win):
        seg = x[i * window:(i + 1) * window]
        mus[i] = float(np.mean(seg))
        sigmas[i] = max(float(np.std(seg)), 1e-12)
    return mus, sigmas


@register
class InfoGeomDistanceNode(Node):
    """Information-Geometry Fisher-Rao regime-distance (GEOMETRIC, utility).

    Models each rolling window of column 0 as a Gaussian (μ,σ) and emits the closed-form
    Fisher-Rao geodesic distance between each window and the previous one as a per-window
    ``signal`` series. The distance spikes when the mean or volatility regime shifts.
    """

    layer = "signal"
    node_type = "infogeom_distance"
    requires_y = False
    is_transformer = False
    cost = "low"

    def __init__(self, name=None, window: int = 30, **params):
        super().__init__(name=name, window=window, **params)
        self.window = int(window)

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        w = max(2, min(self.window, max(2, n // 4)))
        mus, sigmas = rolling_gaussian_params(x, w)
        n_win = mus.size
        if n_win < 2:
            return Belief(type="signal",
                          payload={"series": [0.0] * max(n_win, 1),
                                   "max_distance": 0.0, "mean_distance": 0.0,
                                   "note": "too-short-for-windows"},
                          confidence=0.1)
        dist = np.zeros(n_win)
        for i in range(1, n_win):
            dist[i] = fisher_rao_gaussian(mus[i - 1], sigmas[i - 1], mus[i], sigmas[i])
        return Belief(
            type="signal",
            payload={
                "series": dist.tolist(),
                "max_distance": float(np.max(dist)),
                "mean_distance": float(np.mean(dist[1:])),
                "window": int(w),
                "n_windows": int(n_win),
            },
            confidence=float(np.clip(n_win / 30.0, 0.2, 0.9)),
        )


# ============================================================ Diffusion map helpers
def diffusion_embed(points: np.ndarray, n_coords: int = 3,
                    epsilon: float | None = None) -> np.ndarray:
    """Diffusion-map embedding of (N, p) points into the leading diffusion coordinates.

    Gaussian affinity Wᵢⱼ = exp(−‖xᵢ−xⱼ‖²/ε); row-normalize to the Markov matrix
    P = D⁻¹W; its right eigenvectors (skipping the trivial constant) scaled by their
    eigenvalues are the diffusion coordinates. ε defaults to the median squared pairwise
    distance (standard self-tuning heuristic). Uses a symmetric conjugate so the
    eigen-solve is real and stable (np.linalg.eigh), never inv.
    """
    P = np.asarray(points, dtype=float)
    N = P.shape[0]
    sq = squareform(pdist(P, metric="sqeuclidean"))
    if epsilon is None:
        med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
        epsilon = float(med) if med > 0 else 1.0
    W = np.exp(-sq / epsilon)
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-300))
    # Symmetric normalized matrix shares eigenvalues with P = D^-1 W; eigh is stable.
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]                      # descending
    vals = vals[order]; vecs = vecs[:, order]
    psi = d_inv_sqrt[:, None] * vecs                    # right eigenvectors of P
    k = min(n_coords, N - 1)
    # skip column 0 (trivial constant eigenvector λ≈1); scale by eigenvalue.
    coords = psi[:, 1:1 + k] * vals[1:1 + k][None, :]
    return coords


@register
class DiffusionMapNode(Node):
    """Diffusion-map manifold embedding (GEOMETRIC, utility, general-purpose).

    Delay-embeds column 0 (or uses (T,D) rows directly when D>1), builds a Gaussian-kernel
    Markov diffusion operator, eigen-decomposes it, and emits the leading diffusion
    coordinate as a ``signal`` series. Recovers a 1-D manifold's intrinsic ordering and
    (Rule 36) separates handwriting-digit classes on pendigits far better than chance.
    """

    layer = "signal"
    node_type = "diffusion_map"
    requires_y = False
    is_transformer = False
    cost = "med"

    _MAX_N = 1500

    def __init__(self, name=None, embed_dim: int = 5, n_coords: int = 3, **params):
        super().__init__(name=name, embed_dim=embed_dim, n_coords=n_coords, **params)
        self.embed_dim = int(embed_dim)
        self.n_coords = int(n_coords)

    def _build_points(self, X: np.ndarray) -> np.ndarray:
        """Multivariate (D>1) rows are points directly; a scalar series is delay-embedded."""
        if X.shape[1] > 1:
            return X
        x = X[:, 0]
        d = max(1, min(self.embed_dim, max(1, x.size // 3)))
        return delay_embed(x, d).T                      # (n_points, d)

    def _predict(self, X: np.ndarray) -> Belief:
        pts = self._build_points(X)
        if pts.shape[0] > self._MAX_N:
            pts = pts[-self._MAX_N:]
        N = pts.shape[0]
        if N < 3:
            return Belief(type="signal",
                          payload={"series": [0.0] * max(N, 1),
                                   "n_coords": 0, "note": "too-few-points"},
                          confidence=0.1)
        coords = diffusion_embed(pts, n_coords=self.n_coords)
        leading = coords[:, 0]
        return Belief(
            type="signal",
            payload={
                "series": leading.tolist(),
                "coords": coords.tolist(),
                "n_coords": int(coords.shape[1]),
                "n_points": int(N),
            },
            confidence=float(np.clip(N / 500.0, 0.2, 0.9)),
        )
