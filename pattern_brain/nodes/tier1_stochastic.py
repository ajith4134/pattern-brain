"""TIER-1 STOCHASTIC-PROCESS / PHYSICS nodes (CONCEPT_EQUATION_BANK.md).

Light-stack only (numpy/scipy/sklearn), each behind the generic Node interface (Rule 23)
and emitting an existing v0.1 interlingua belief type. Domain-agnostic: a generic (T, D)
sequence in, a Belief out — no candle/orderbook knowledge inside the node.

Built here (oracle-test-first, per CLAUDE.md):
  * ``jump_diffusion`` — Merton jump-diffusion: separate a diffusion N(μ,σ²) component from
    compound-Poisson jumps via a local-vol-standardized threshold; estimate jump intensity λ,
    jump mean/std and the diffusion vol. A TAIL/GAP detector — judged by recovery correctness.
  * ``heston_vol`` — Heston stochastic vol: variance follows the CIR SDE
    dv = κ(θ−v)dt + ξ√v dW. Method-of-moments on a realized-variance series recovers κ,θ,ξ and
    forecasts the next variance. Judged vs the persistence baseline (RV_{t+1}=RV_t).
  * ``sv_particle`` — Stochastic Volatility via a bootstrap particle filter: latent log-vol
    AR(1) h_t = μ + φ(h_{t-1}−μ) + ση, obs y_t = exp(h_t/2)ε_t. Filters the latent vol.
  * ``percolation_risk`` — percolation / network-contagion systemic risk: a threshold
    correlation graph across assets; the giant-connected-component fraction + the percolation
    transition threshold are the systemic-risk indicators.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from ..belief import Belief
from ..node import Node
from ..registry import register


# ============================================================ 1. Merton jump-diffusion
def _rolling_mad_std(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling robust scale (1.4826 * MAD) — the local diffusion volatility.

    A robust local scale is used so that the jumps themselves (the things we want to flag)
    do not inflate the threshold that is supposed to catch them. Returns a length-n array,
    floored positive so the standardized score is always finite.
    """
    n = x.size
    half = max(w // 2, 1)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        out[i] = 1.4826 * mad
    glob = 1.4826 * np.median(np.abs(x - np.median(x)))
    floor = max(glob, 1e-12) * 0.25
    return np.maximum(out, floor)


def detect_jumps(x: np.ndarray, threshold: float = 4.0, window: int = 50) -> np.ndarray:
    """Boolean jump mask: observations whose local-vol-standardized magnitude exceeds
    ``threshold`` MADs. A Gaussian diffusion almost never exceeds 4σ, so a clean diffusion
    yields ~no flags while compound-Poisson jumps (heavy outliers) are caught."""
    x = np.asarray(x, dtype=float).ravel()
    scale = _rolling_mad_std(x, window)
    med = np.median(x)
    z = np.abs(x - med) / scale
    return z > threshold


def fit_jump_diffusion(x: np.ndarray, threshold: float = 4.0, window: int = 50) -> dict:
    """Separate diffusion from jumps, then estimate the Merton parameters by moments.

    λ̂ = (#jumps)/n (per-step jump probability ≈ intensity for a unit-time step); the jump
    mean/std are the sample moments of the flagged observations; the diffusion vol is the
    robust scale of the UN-flagged observations (jumps removed).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    mask = detect_jumps(x, threshold, window)
    n_jumps = int(mask.sum())
    diffusion = x[~mask]
    diff_vol = float(1.4826 * np.median(np.abs(diffusion - np.median(diffusion)))) if diffusion.size > 2 else float(np.std(x))
    diff_mu = float(np.median(diffusion)) if diffusion.size else 0.0
    if n_jumps > 0:
        jvals = x[mask]
        jump_mean = float(np.mean(jvals))
        jump_std = float(np.std(jvals)) if n_jumps > 1 else float(abs(jvals[0]))
    else:
        jump_mean, jump_std = 0.0, 0.0
    return {"mask": mask, "lambda": n_jumps / max(n, 1), "n_jumps": n_jumps,
            "jump_mean": jump_mean, "jump_std": jump_std,
            "diffusion_vol": max(diff_vol, 1e-12), "diffusion_mu": diff_mu}


@register
class JumpDiffusionNode(Node):
    """Merton jump-diffusion separator (stochastic-process / tail tool). Splits a return
    series into a Gaussian diffusion and compound-Poisson jumps via a robust local-vol
    threshold. Emits a ``signal`` whose per-point ``series`` is the jump indicator (1=jump);
    payload carries λ, jump mean/std, diffusion vol and the jump count. A gap/tail detector,
    judged by recovery correctness, not point-forecast PnL."""

    layer = "signal"
    node_type = "jump_diffusion"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        window = int(min(max(20, n // 10), 100))
        res = fit_jump_diffusion(x, threshold=3.5, window=window)
        series = res["mask"].astype(float)
        conf = float(np.clip(n / 500.0, 0.2, 0.9))
        return Belief(
            type="signal",
            payload={
                "series": series.tolist(),            # per-point jump indicator, length n
                "lambda": float(res["lambda"]),
                "jump_mean": float(res["jump_mean"]),
                "jump_std": float(res["jump_std"]),
                "diffusion_vol": float(res["diffusion_vol"]),
                "n_jumps": int(res["n_jumps"]),
            },
            confidence=conf,
        )


# ============================================================ 2. Heston / CIR variance
def fit_cir_mom(rv: np.ndarray, dt: float = 1.0) -> dict:
    """Method-of-moments estimate of the CIR variance SDE dv = κ(θ−v)dt + ξ√v dW.

    θ̂ = mean(RV) (the long-run variance level). κ̂ from the AR(1) autocorrelation of RV:
    the exact CIR discretization is E[v_{t+1}|v_t] = θ + e^{−κdt}(v_t − θ), so the lag-1
    autocorrelation φ = e^{−κdt} ⇒ κ̂ = −ln(φ)/dt. ξ̂ from the conditional variance of the
    AR(1) residual, which for CIR scales with v: Var = ξ²·v·(1−e^{−κdt})/κ·… → ξ̂ matched
    on the residual variance divided by the mean level.
    """
    rv = np.clip(np.asarray(rv, dtype=float).ravel(), 0.0, None)
    theta = float(np.mean(rv))
    r0, r1 = rv[:-1], rv[1:]
    # AR(1) slope of de-meaned RV = phi = exp(-kappa*dt)
    denom = float(np.sum((r0 - theta) ** 2))
    phi = float(np.sum((r0 - theta) * (r1 - theta)) / denom) if denom > 1e-30 else 0.0
    phi = float(np.clip(phi, 1e-6, 0.999999))
    kappa = -np.log(phi) / dt
    # residual of the AR(1) fit; CIR conditional variance ∝ v → scale ξ off residual var / mean v
    resid = (r1 - theta) - phi * (r0 - theta)
    resid_var = float(np.var(resid))
    mean_v = max(theta, 1e-12)
    # discrete CIR: Var[v_{t+1}|v_t] = v_t * ξ²/κ * e^{-κdt}(1-e^{-κdt}) + θ ξ²/(2κ)(1-e^{-κdt})²
    factor = (1.0 - phi ** 2) / (2.0 * kappa) if kappa > 1e-9 else dt
    xi = float(np.sqrt(max(resid_var / (mean_v * max(factor, 1e-12)), 0.0)))
    return {"kappa": float(kappa), "theta": theta, "xi": xi, "phi": phi}


@register
class HestonVolNode(Node):
    """Heston stochastic-volatility estimator (CIR variance). Takes a realized-variance
    series (column 0; the harness supplies RV = returns²), recovers κ (mean-reversion speed),
    θ (long-run variance) and ξ (vol-of-vol) by method-of-moments, and forecasts the next
    variance via the exact CIR conditional mean θ + φ(v_t−θ). Emits a ``forecast``."""

    layer = "equation"
    node_type = "heston_vol"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        rv = np.clip(X[:, 0].astype(float), 0.0, None)
        n = rv.size
        if n < 10:
            est = {"kappa": 0.0, "theta": float(np.mean(rv)) if n else 0.0, "xi": 0.0, "phi": 0.0}
        else:
            est = fit_cir_mom(rv)
        v_last = float(rv[-1]) if n else 0.0
        next_var = est["theta"] + est["phi"] * (v_last - est["theta"])
        next_var = float(max(next_var, 0.0))
        conf = float(np.clip(n / 500.0, 0.2, 0.9))
        return Belief(
            type="forecast",
            payload={
                "estimate": next_var,
                "next_variance": next_var,
                "vol_forecast": float(np.sqrt(next_var)),
                "kappa": float(est["kappa"]),
                "theta": float(est["theta"]),
                "xi": float(est["xi"]),
            },
            confidence=conf,
        )


# ============================================================ 3. SV bootstrap particle filter
def _estimate_sv_params(y: np.ndarray) -> dict:
    """Moment estimates of the log-vol AR(1) from observed returns y.

    log(y²) = h + log(ε²); with ε~N(0,1), E[log ε²]≈−1.2704 and Var≈π²/2. So
    ŝ = log(y²) + 1.2704 is a NOISY observation of h. The observation noise is iid, so it
    only inflates the lag-0 variance — the autocovariances at lags k≥1 are exactly those of
    the AR(1) state: γ_k(s) = φ^k·Var(h). Estimating φ from the lag-1 SAMPLE autocorrelation
    of ŝ therefore suffers a severe ATTENUATION bias (the huge obs-noise variance crushes it
    toward 0). The fix: recover φ from the SLOPE of log γ_k vs k over several lags (k≥1),
    where the obs noise cancels; then Var(h)=γ_1/φ and ση²=Var(h)(1−φ²).
    """
    y = np.asarray(y, dtype=float).ravel()
    s = np.log(y ** 2 + 1e-12) + 1.2704           # de-bias the observation-noise mean
    mu = float(np.mean(s))
    sd = s - mu
    n = sd.size
    K = min(10, max(2, n // 50))
    gk = np.array([float(np.mean(sd[:n - k] * sd[k:])) for k in range(1, K + 1)])
    valid = gk > 0
    if valid.sum() >= 2:                          # log-linear fit of γ_k = γ·φ^k → slope = ln φ
        ks = np.arange(1, K + 1)[valid]
        log_phi = float(np.polyfit(ks, np.log(gk[valid]), 1)[0])
        phi = float(np.clip(np.exp(log_phi), -0.99, 0.999))
    else:
        phi = 0.0
    var_h = max(gk[0] / phi, 1e-6) if phi > 1e-6 else max(np.var(sd) - (np.pi ** 2) / 2.0, 1e-6)
    sigma_eta = float(np.sqrt(max(var_h * (1.0 - phi ** 2), 1e-4)))
    return {"mu": mu, "phi": phi, "sigma_eta": float(np.clip(sigma_eta, 1e-3, 3.0))}


def bootstrap_particle_filter(y: np.ndarray, mu: float, phi: float, sigma_eta: float,
                              n_particles: int = 400, seed: int = 0) -> np.ndarray:
    """Bootstrap (SIR) particle filter for the latent log-vol; returns the filtered E[exp(h_t/2)].

    Propagate particles through the AR(1) state, weight by the Gaussian observation density
    y_t ~ N(0, exp(h_t)), normalize, resample (systematic). The filtered volatility estimate
    at each t is the weighted mean of exp(h/2) over the particle cloud.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    h = rng.normal(mu, sigma_eta / np.sqrt(max(1 - phi ** 2, 1e-3)), n_particles)
    filt = np.empty(n)
    h_lo, h_hi = mu - 20.0, mu + 20.0             # keep exp(h) finite (overflow guard)
    for t in range(n):
        h = mu + phi * (h - mu) + sigma_eta * rng.normal(size=n_particles)
        h = np.clip(h, h_lo, h_hi)
        var = np.exp(h)
        logw = -0.5 * np.log(2 * np.pi * var) - 0.5 * y[t] ** 2 / var
        logw -= logw.max()
        w = np.exp(logw)
        wsum = w.sum()
        if not np.isfinite(wsum) or wsum <= 0:
            w = np.ones(n_particles) / n_particles
        else:
            w /= wsum
        filt[t] = float(np.sum(w * np.exp(h / 2.0)))
        # systematic resampling to fight weight degeneracy
        positions = (rng.random() + np.arange(n_particles)) / n_particles
        idx = np.searchsorted(np.cumsum(w), positions)
        idx = np.clip(idx, 0, n_particles - 1)
        h = h[idx]
    return filt


@register
class SVParticleNode(Node):
    """Stochastic-volatility bootstrap particle filter. Treats column 0 as a return series,
    infers the latent log-vol AR(1) (μ,φ,ση) by moments, then particle-filters the latent
    volatility path. Emits a ``forecast`` whose ``series`` is the filtered volatility (a
    de-noised vol estimate); payload carries the recovered AR(1) parameters."""

    layer = "probability"
    node_type = "sv_particle"
    requires_y = False
    is_transformer = False
    cost = "med"

    _MAX_N = 2000

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n_full = x.size
        pad = 0
        if n_full > self._MAX_N:                       # bound the filter cost
            pad = n_full - self._MAX_N
            x = x[-self._MAX_N:]
        x = x - np.mean(x)                             # center; the filter models y_t ~ N(0,·)
        if x.size < 20 or np.std(x) < 1e-12:
            filt = np.full(x.size, float(np.std(x)) or 1e-6)
            params = {"mu": 0.0, "phi": 0.0, "sigma_eta": 0.0}
        else:
            params = _estimate_sv_params(x)
            filt = bootstrap_particle_filter(x, params["mu"], params["phi"],
                                             params["sigma_eta"], n_particles=400, seed=0)
        series = np.concatenate([np.full(pad, filt[0] if filt.size else 0.0), filt])
        conf = float(np.clip(n_full / 500.0, 0.2, 0.9))
        return Belief(
            type="forecast",
            payload={
                "series": series.tolist(),             # filtered volatility, length n_full
                "estimate": float(filt[-1]) if filt.size else 0.0,
                "vol_forecast": float(filt[-1]) if filt.size else 0.0,
                "phi": float(params["phi"]),
                "mu": float(params["mu"]),
                "sigma_eta": float(params["sigma_eta"]),
            },
            confidence=conf,
        )


# ============================================================ 4. percolation / network risk
def _giant_component_fraction(adj: np.ndarray) -> float:
    """Fraction of nodes in the largest connected component of an undirected graph (union-find)."""
    d = adj.shape[0]
    parent = list(range(d))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(d):
        for j in range(i + 1, d):
            if adj[i, j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    roots = [find(i) for i in range(d)]
    counts = np.bincount(roots, minlength=d)
    return float(counts.max() / d) if d else 0.0


def percolation_sweep(corr: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, float]:
    """Sweep the |correlation| edge threshold; return giant-component fraction at each
    threshold and the percolation-transition threshold (the largest threshold at which a
    giant component — fraction > 0.5 — still spans the network)."""
    abs_c = np.abs(corr).copy()
    np.fill_diagonal(abs_c, 0.0)
    fracs = np.array([_giant_component_fraction(abs_c >= t) for t in thresholds])
    above = np.where(fracs > 0.5)[0]
    perc_thr = float(thresholds[above[-1]]) if above.size else float(thresholds[0])
    return fracs, perc_thr


@register
class PercolationRiskNode(Node):
    """Percolation / network-contagion systemic-risk indicator. Input is a (T, D) cross-section
    of returns (D assets in columns). Builds a threshold |correlation| graph, sweeps the
    threshold to locate the percolation transition, and reports the giant-connected-component
    fraction and mean degree as systemic-risk measures. Emits a ``signal`` whose ``series`` is
    a rolling systemic-risk index (rises in high-correlation / crisis windows)."""

    layer = "signal"
    node_type = "percolation_risk"
    requires_y = False
    is_transformer = False
    cost = "med"

    _EDGE_THR = 0.5          # fixed reference threshold for the rolling/degree readouts
    _WINDOW = 60

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        if D < 2:                                      # need a cross-section for a network
            return Belief(type="signal",
                          payload={"series": [0.0] * T, "giant_component_frac": 0.0,
                                   "percolation_threshold": 0.0, "mean_degree": 0.0,
                                   "note": "needs D>=2 assets for a network"},
                          confidence=0.1)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        thresholds = np.linspace(0.05, 0.95, 19)
        fracs, perc_thr = percolation_sweep(corr, thresholds)
        adj = (np.abs(corr) >= self._EDGE_THR)
        np.fill_diagonal(adj, False)
        giant = _giant_component_fraction(adj)
        mean_degree = float(adj.sum(axis=1).mean())
        series = self._rolling_systemic_risk(X)
        conf = float(np.clip(T / 500.0, 0.2, 0.9))
        return Belief(
            type="signal",
            payload={
                "series": series.tolist(),             # rolling systemic-risk index, length T
                "giant_component_frac": float(giant),
                "percolation_threshold": float(perc_thr),
                "mean_degree": mean_degree,
                "order_parameter_curve": fracs.tolist(),
                "threshold_grid": thresholds.tolist(),
            },
            confidence=conf,
        )

    def _rolling_systemic_risk(self, X: np.ndarray) -> np.ndarray:
        """Per-time systemic-risk index = giant-component fraction in a trailing window's
        correlation graph at the fixed reference threshold (rises when assets co-move)."""
        T, D = X.shape
        w = min(self._WINDOW, max(T // 4, 10))
        out = np.zeros(T)
        for t in range(T):
            lo = max(0, t - w + 1)
            seg = X[lo:t + 1]
            if seg.shape[0] < 5:
                continue
            c = np.nan_to_num(np.corrcoef(seg, rowvar=False), nan=0.0)
            adj = (np.abs(c) >= self._EDGE_THR)
            np.fill_diagonal(adj, False)
            out[t] = _giant_component_fraction(adj)
        return out
