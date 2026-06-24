"""Tier-3 stochastic-process foundations (Rule 30 invention; PLAN.md §11 family).

Three canonical SDE-engine nodes — the honest baselines and density evolvers that
the more exotic models in the bank must beat. Light-stack (numpy only), each behind
the generic Node interface (Rule 23) and emitting EXISTING interlingua belief types
(Block 46) so the bank stays conformant.

* ``gbm_baseline`` (layer ``sequence``) — **Geometric / Bachelier Brownian Motion**:
  the canonical "random-walk-with-drift" forecaster. Estimate per-step drift mu and
  vol sigma from the window's log-returns; the one-step level forecast is the
  closed-form lognormal mean ``E[x_{t+1}|x_t] = x_t * exp(mu + sigma^2/2)``. On a
  series that crosses/▒touches zero (no logs) it falls back to arithmetic Bachelier
  BM ``x_t + mean(dx)``. Emits ``forecast``. This is THE canonical baseline — expect
  it to roughly tie persistence (we report that honestly).

* ``langevin_sampler`` (layer ``sequence``) — **Langevin / Ornstein-Uhlenbeck**
  dynamics ``dX = -gamma (X - mu) dt + sigma dW``. Fits (gamma, mu, sigma) from the
  AR(1) regression ``X_{t+1} = a + b X_t`` (b = e^{-gamma dt}, mu = a/(1-b)); the
  one-step conditional mean is ``a + b X_t``. NOTE (honest overlap): this is the
  SDE-engine framing of the existing ``ou_mean_reversion`` node — the discrete
  Langevin conditional mean is EXACTLY the AR(1)/OU form (the oracle test pins this
  to <1e-8). Emits ``forecast``. Its design horizon is mean-reverting LEVELS
  (spreads), not raw returns (Rule 34).

* ``fokker_planck`` (layer ``probability``) — **Fokker-Planck forward equation**:
  evolves the probability DENSITY of the next value one step forward from the fitted
  drift + diffusion. For the OU/Bachelier generator the forward solution is the exact
  Gaussian transition density ``N(m, v)`` with ``m = mu + (x_t-mu) e^{-gamma dt}`` and
  ``v = sigma^2/(2 gamma)(1 - e^{-2 gamma dt})`` (the diffusive limit ``v = sigma^2 dt``
  when gamma -> 0). Discretized on a grid that integrates to 1. Emits ``density``
  (judged by correctness — integrates to 1, mean/var match the SDE — not by PnL).

Domain-agnostic by construction (Rule 23): a generic (T, D) sequence in, a Belief
out; nothing here references candles or order books.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

_EPS = 1e-12


# ===================================================================== GBM
def _log_returns(x: np.ndarray):
    """Per-step log-returns of a strictly-positive series, or ``None`` if any value
    is non-positive (logs undefined -> caller uses the Bachelier fallback)."""
    if np.any(x <= 0.0):
        return None
    return np.diff(np.log(x))


@register
class GBMBaselineNode(Node):
    """Geometric Brownian Motion baseline (Bachelier 1900 / Black-Scholes drift).

    Estimates drift mu and vol sigma from the window's log-returns and forecasts the
    next LEVEL as the closed-form lognormal mean. The honest random-walk-with-drift
    baseline every fancier forecaster must beat. Falls back to arithmetic (Bachelier)
    Brownian motion on series that aren't strictly positive.
    """

    layer = "sequence"
    node_type = "gbm_baseline"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        x = x[np.isfinite(x)]
        last = float(x[-1]) if x.size else 0.0
        if x.size < 3:
            return Belief("forecast", {"next_vector": [last], "estimate": last,
                                       "std": 0.0, "model": "GBM(persistence)"},
                          0.1, self.name)

        r = _log_returns(x)
        if r is not None and r.size >= 2:
            mu = float(np.mean(r))            # per-step log-drift
            sigma = float(np.std(r, ddof=1))  # per-step log-vol
            # lognormal mean of the next level: E[x_{t+1}|x_t] = x_t * exp(mu + sigma^2/2)
            pred = last * float(np.exp(mu + 0.5 * sigma ** 2))
            # 1-sigma band on the level (lognormal sd): x_t*exp(mu+s^2/2)*sqrt(exp(s^2)-1)
            band = pred * float(np.sqrt(max(np.exp(sigma ** 2) - 1.0, 0.0)))
            model = "GBM(lognormal)"
        else:
            # arithmetic Bachelier BM: drift = mean increment, band = increment sd
            dx = np.diff(x)
            mu = float(np.mean(dx))
            sigma = float(np.std(dx, ddof=1)) if dx.size >= 2 else 0.0
            pred = last + mu
            band = sigma
            model = "Bachelier(arithmetic)"

        if not np.isfinite(pred):
            return Belief("forecast", {"next_vector": [last], "estimate": last,
                                       "std": 0.0, "model": "GBM(persistence)"},
                          0.1, self.name)
        # confidence: drift magnitude relative to its own noise band (small for a
        # near-zero-drift random walk -> honestly low, since GBM ~ persistence).
        conf = float(max(0.05, min(0.9, abs(pred - last) / (band + _EPS))))
        return Belief("forecast",
                      {"next_vector": [float(pred)], "estimate": float(pred),
                       "std": float(band), "mu": mu, "sigma": sigma, "model": model},
                      conf, self.name)


# ================================================================ Langevin/OU
def _ar1_fit(x: np.ndarray):
    """OLS AR(1): x_{t+1} = a + b x_t. Returns (a, b, resid) via solve (not invert)."""
    x0, x1 = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(x0), x0])
    # normal equations solved with np.linalg.lstsq (QR-based, stable) — Rule 10
    coef, *_ = np.linalg.lstsq(A, x1, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = x1 - (a + b * x0)
    return a, b, resid


def _ou_params_from_ar1(a: float, b: float, resid: np.ndarray, dt: float = 1.0):
    """Map an AR(1) fit to OU/Langevin (gamma, mu, sigma).

    OU: dX = -gamma(X-mu)dt + sigma dW.  Discrete: X_{t+1} = mu + (X_t-mu)e^{-gamma dt} + eps.
    So b = e^{-gamma dt}, mu = a/(1-b); the innovation sd gives sigma via
    var(eps) = sigma^2/(2 gamma)(1-e^{-2 gamma dt}).
    """
    b_stable = float(min(max(b, -0.999999), 0.999999))
    mu = a / (1.0 - b_stable) if abs(1.0 - b_stable) > 1e-9 else float("nan")
    resid_var = float(np.var(resid)) + _EPS
    if 0.0 < b_stable < 1.0:
        gamma = -np.log(b_stable) / dt
        denom = (1.0 - b_stable ** 2)
        sigma = float(np.sqrt(max(2.0 * gamma * resid_var / denom, 0.0))) if denom > 1e-12 else 0.0
        half_life = float(np.log(2.0) / gamma) if gamma > 1e-12 else float("inf")
    else:
        # non-mean-reverting (unit-root / explosive): no OU reversion; report the
        # diffusive limit (gamma -> 0) honestly.
        gamma = 0.0
        sigma = float(np.sqrt(resid_var / dt))
        half_life = float("inf")
    return float(gamma), float(mu), float(sigma), float(half_life)


@register
class LangevinSamplerNode(Node):
    """Langevin / Ornstein-Uhlenbeck SDE engine (IDEA tier-3).

    dX = -gamma (X - mu) dt + sigma dW. Fits (gamma, mu, sigma) by OLS on the AR(1)
    form and emits the one-step conditional-mean forecast plus reversion diagnostics
    (gamma, mu, half-life, standardized deviation z_dev).

    HONEST OVERLAP: the discrete Langevin conditional mean E[X_{t+1}|X_t] = a + b X_t
    is identical to AR(1) and to the existing ``ou_mean_reversion`` node — this is the
    SDE-engine framing of the same math (the oracle test pins the equality to <1e-8).
    Design horizon = mean-reverting LEVELS (spreads), not raw returns (Rule 34).
    """

    layer = "sequence"
    node_type = "langevin_sampler"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        x = x[np.isfinite(x)]
        last = float(x[-1]) if x.size else 0.0
        if x.size < 8:
            return Belief("forecast", {"next_vector": [last], "estimate": last,
                                       "model": "Langevin(persistence)"}, 0.1, self.name)

        a, b, resid = _ar1_fit(x)
        pred = a + b * last
        if not np.isfinite(pred):
            return Belief("forecast", {"next_vector": [last], "estimate": last,
                                       "model": "Langevin(persistence)"}, 0.1, self.name)

        gamma, mu, sigma, half_life = _ou_params_from_ar1(a, b, resid, dt=1.0)
        z_dev = float((last - mu) / (sigma + _EPS)) if np.isfinite(mu) else 0.0

        # confidence from in-sample R^2 of the AR(1) fit (bounded)
        x1 = x[1:]
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((x1 - x1.mean()) ** 2)) + _EPS
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        conf = float(max(0.05, min(0.95, r2)))

        return Belief("forecast",
                      {"next_vector": [float(pred)], "estimate": float(pred),
                       "gamma": gamma, "mu": mu if np.isfinite(mu) else 0.0,
                       "sigma": sigma, "half_life": half_life, "z_dev": z_dev,
                       "model": "Langevin/OU(AR1)"},
                      conf, self.name)


# ============================================================== Fokker-Planck
def _ou_transition_moments(x: np.ndarray, dt: float = 1.0):
    """Fit the OU/Bachelier generator and return the forward (Fokker-Planck)
    transition moments (mean m, variance v) from the LAST observed value, plus the
    fitted (gamma, mu, sigma) and the AR(1) residual SSE/n for a BIC."""
    a, b, resid = _ar1_fit(x)
    gamma, mu, sigma, _hl = _ou_params_from_ar1(a, b, resid, dt=dt)
    last = float(x[-1])
    b_eff = float(np.exp(-gamma * dt))
    if gamma > 1e-9:
        m = mu + (last - mu) * b_eff
        v = sigma ** 2 / (2.0 * gamma) * (1.0 - b_eff ** 2)
    else:
        # diffusive limit (driftless/unit-root): m = a + b*last, v = sigma^2 dt
        m = a + b * last
        v = sigma ** 2 * dt
    v = float(max(v, _EPS))
    n = max(1, len(resid))
    sse = float(np.sum(resid ** 2))
    return float(m), v, gamma, mu, sigma, sse, n


@register
class FokkerPlanckNode(Node):
    """Fokker-Planck forward equation — evolve the next-value DENSITY one step.

    For the OU/Bachelier generator fitted from the data, the forward solution of
    dp/dt = -d/dx[A(x) p] + 1/2 d^2/dx^2[B p] is the exact Gaussian transition
    density N(m, v) (m, v are the closed-form OU transition moments). The density is
    discretized on a +-4-sigma grid that integrates to 1 (Simpson-faithful trapezoid)
    and emitted as the bank's ``density`` belief. Judged by correctness, not PnL.
    """

    layer = "probability"
    node_type = "fokker_planck"

    _GRID = 401          # odd -> symmetric, mode-centered grid

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            last = float(x[-1]) if x.size else 0.0
            return self._degenerate(last)

        m, v, gamma, mu, sigma, sse, n = _ou_transition_moments(x, dt=1.0)
        sd = float(np.sqrt(v))
        if not (np.isfinite(m) and sd > 0):
            return self._degenerate(float(x[-1]))

        # Gaussian transition density on a +-4 sigma grid (captures > 0.99994 mass).
        grid = np.linspace(m - 4.0 * sd, m + 4.0 * sd, self._GRID)
        pdf = np.exp(-0.5 * ((grid - m) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        mass = float(trap(pdf, grid))
        pdf = pdf / (mass + _EPS)                 # renormalize over the finite grid -> integrates to 1

        # discrete bin masses (probability per cell) -> labels + max_proba for the
        # 'density' interlingua contract; the mode bin carries the peak mass.
        widths = np.diff(grid)
        cell_mass = 0.5 * (pdf[:-1] + pdf[1:]) * widths     # trapezoid mass per cell
        cell_mass = cell_mass / (cell_mass.sum() + _EPS)
        mode_bin = int(np.argmax(cell_mass))
        max_proba = float(cell_mass[mode_bin])

        # BIC of the OU(AR1) generator fit (k=3 free params) — model-fit scalar for
        # the density contract (Gaussian innovations: -2 logL + k ln n).
        resid_var = sse / n + _EPS
        loglik = -0.5 * n * (np.log(2.0 * np.pi * resid_var) + 1.0)
        bic = float(-2.0 * loglik + 3.0 * np.log(max(n, 2)))

        # confidence: peakedness of the transition density relative to the data range
        rng = float(np.ptp(x)) + _EPS
        conf = float(max(0.05, min(0.95, 1.0 / (1.0 + sd / rng))))

        return Belief("density",
                      {"labels": [mode_bin], "max_proba": max_proba, "bic": bic,
                       "mean": m, "variance": v, "std": sd,
                       "gamma": gamma, "mu": mu if np.isfinite(mu) else 0.0,
                       "sigma": sigma, "next_vector": [m],
                       "grid": grid.tolist(), "pdf": pdf.tolist(),
                       "model": "FokkerPlanck(OU-transition)"},
                      conf, self.name)

    def _degenerate(self, last: float) -> Belief:
        """Too little data: a near-delta density centered on the last value."""
        sd = 1e-3
        grid = np.linspace(last - 4.0 * sd, last + 4.0 * sd, self._GRID)
        pdf = np.exp(-0.5 * ((grid - last) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        pdf = pdf / (float(trap(pdf, grid)) + _EPS)
        return Belief("density",
                      {"labels": [self._GRID // 2], "max_proba": 1.0, "bic": 0.0,
                       "mean": last, "variance": sd ** 2, "std": sd,
                       "gamma": 0.0, "mu": last, "sigma": 0.0, "next_vector": [last],
                       "grid": grid.tolist(), "pdf": pdf.tolist(),
                       "model": "FokkerPlanck(degenerate)"},
                      0.1, self.name)
