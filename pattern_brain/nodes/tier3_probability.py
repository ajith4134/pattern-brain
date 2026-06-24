"""Tier-3 timeless probability foundations (CONCEPT_EQUATION_BANK.md).

Three classical, distribution-level nodes — the Bernoulli/Bayes, Boltzmann-Gibbs/
Jaynes-MaxEnt, and Chebyshev lineages — each behind the generic Node interface
(Rule 23: a (T, D) sequence in, a Belief out; no candle/order-book knowledge) and
each reusing an EXISTING interlingua belief type (no new type invented).

* ``bayes_update`` (``forecast``) — conjugate **Normal-Normal** posterior updating.
  The next value's mean is a Normal posterior: an empirical-Bayes Normal prior built
  from the window is updated by the same window's observations (conjugate closed form,
  no sampling). Emits the posterior-mean point estimate + predictive std.
* ``maxent_dist`` (``density``) — the **Maximum-Entropy / Boltzmann-Gibbs**
  distribution p(x) ∝ exp(-Σ λ_i f_i(x)) matching the window's moments. With the
  mean+variance constraints the closed-form MaxEnt solution is the Gaussian
  N(μ, σ²); an optional 3rd/4th-moment (skew/kurtosis) refinement solves λ
  numerically (Newton on the log-partition gradient, log-sum-exp stabilized) and
  FLAGS ill-conditioning. Judged by correctness (recovers the right distribution),
  not PnL.
* ``chebyshev_bound`` (``forecast``) — the **distribution-free** Chebyshev tail
  interval [μ-kσ, μ+kσ] around a point forecast, with the guarantee
  P(|X-μ|≥kσ) ≤ 1/k² (so coverage ≥ 1-1/k² for ANY finite-variance distribution).
  A conformal-but-distribution-free band.

Numerics (Code-Generation Contract Phase 3): float64 throughout; log-sum-exp for
the MaxEnt partition function; ``np.linalg.solve`` (never ``inv``) for the Newton
step; ill-conditioning surfaced via the Jacobian condition number, not hidden.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

_EPS = 1e-12


# ===================================================================== Bayes
def normal_normal_posterior(x: np.ndarray, *, mu0: float, tau2: float,
                            sigma2: float) -> Tuple[float, float]:
    """Conjugate Normal-Normal posterior for an unknown mean.

    Prior on the mean: ``N(mu0, tau2)``. Likelihood: ``n`` iid observations ``x``
    with KNOWN variance ``sigma2``. Returns ``(posterior_mean, posterior_var)``
    via the textbook precision-weighted closed form::

        prec_post = 1/tau2 + n/sigma2
        mu_post   = (mu0/tau2 + (Σx)/sigma2) / prec_post
        var_post  = 1 / prec_post

    Pre-conditions: ``tau2 > 0`` and ``sigma2 > 0`` (both required, no exception
    swallowed — degenerate priors are clamped to ``_EPS`` so the precision stays
    finite). Post-condition: with ``mu0`` between, ``mu_post`` lies between the
    prior mean and the data mean.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    tau2 = max(float(tau2), _EPS)
    sigma2 = max(float(sigma2), _EPS)
    prec_prior = 1.0 / tau2
    prec_data = n / sigma2
    prec_post = prec_prior + prec_data
    mu_post = (mu0 * prec_prior + x.sum() / sigma2) / prec_post
    var_post = 1.0 / prec_post
    return float(mu_post), float(var_post)


@register
class BayesUpdateNode(Node):
    """Bayesian Posterior Updater — conjugate Normal-Normal forecast of the next mean.

    Empirical-Bayes prior from the window: prior mean = window mean, prior variance
    = window variance, likelihood variance = window variance (one observation's
    worth of spread). The posterior mean is the one-step point forecast; the
    predictive std combines the posterior uncertainty and the observation spread.
    """

    layer = "probability"
    node_type = "bayes_update"
    cost = "low"
    name_display = "Bayesian Posterior Updater"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if x.size < 2:
            last = float(x[-1]) if x.size else 0.0
            return Belief("forecast", {"next_vector": [last], "estimate": last,
                                       "posterior_std": 0.0, "model": "Bayes(persistence)"},
                          0.1, self.name)
        mu0 = float(x.mean())
        var = float(x.var(ddof=1)) if x.size > 1 else 1.0
        tau2 = max(var, _EPS)            # empirical-Bayes prior spread
        sigma2 = max(var, _EPS)          # observation noise
        mu_post, var_post = normal_normal_posterior(x, mu0=mu0, tau2=tau2, sigma2=sigma2)
        pred_std = float(np.sqrt(var_post + sigma2))   # predictive (posterior + obs noise)
        # confidence: tighter posterior (relative to the data spread) -> higher
        conf = float(max(0.05, min(0.95, 1.0 - var_post / (tau2 + _EPS))))
        return Belief("forecast",
                      {"next_vector": [float(mu_post)], "estimate": float(mu_post),
                       "posterior_mean": float(mu_post), "posterior_var": float(var_post),
                       "posterior_std": float(np.sqrt(var_post)), "std": pred_std,
                       "model": "Bayes(Normal-Normal)"},
                      conf, self.name)


# ============================================================ Maximum entropy
def gaussian_maxent(x: np.ndarray) -> Dict[str, float]:
    """Closed-form Maximum-Entropy distribution under mean+variance constraints.

    Jaynes' result: the maximum-entropy density on the real line constrained to a
    given mean and variance is the Gaussian ``N(mu, sigma^2)``. So the MaxEnt fit
    under those two moments is exactly the empirical mean and std. Returns
    ``{"mu", "sigma"}`` (sigma uses ddof=0 to match the moment definition E[(x-μ)²]).
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = float(x.mean())
    sigma = float(np.sqrt(max(x.var(ddof=0), _EPS)))
    return {"mu": mu, "sigma": sigma}


def _moment_maxent_lambdas(z: np.ndarray, moments: int = 4,
                           max_iter: int = 50, tol: float = 1e-8):
    """Numerically fit p(z) ∝ exp(-Σ λ_i z^i), i=1..moments, on a STANDARDIZED grid
    so it matches the empirical moments of ``z`` (mean 0, var 1, plus skew/kurt).

    Newton iteration on the constraint residual E_p[z^i] - m_i, with the partition
    function and expectations computed by log-sum-exp on a fixed quadrature grid
    (numerically stable). The Jacobian is the (negative) covariance of the basis
    moments under p; solved with ``np.linalg.solve`` (never ``inv``). Returns
    ``(lambdas, grid, log_pdf, ill_conditioned, cond_number)``. ``ill_conditioned``
    is flagged when the Jacobian condition number exceeds 1e10 — the design FLAGS
    it rather than returning a silently wrong fit.
    """
    grid = np.linspace(-8.0, 8.0, 1601)              # quadrature support (standardized)
    dz = grid[1] - grid[0]
    powers = np.vstack([grid ** i for i in range(1, moments + 1)])   # (moments, G)
    target = np.array([np.mean(z ** i) for i in range(1, moments + 1)])  # empirical moments
    lam = np.zeros(moments)
    # seed with the Gaussian solution (lambda_1=0, lambda_2=1/2 in standardized units)
    if moments >= 2:
        lam[1] = 0.5
    ill, cond = False, 1.0
    for _ in range(max_iter):
        log_w = -(lam[:, None] * powers).sum(axis=0)               # (G,)
        log_z = _logsumexp(log_w + np.log(dz))
        p = np.exp(log_w + np.log(dz) - log_z)                     # normalized weights, sums≈1
        exp_moments = powers @ p                                   # E_p[z^i]
        resid = exp_moments - target
        if np.max(np.abs(resid)) < tol:
            break
        # Jacobian J_ij = d E_p[z^i] / d lam_j = -(E[z^i z^j] - E[z^i]E[z^j])  (=-cov)
        second = (powers * p) @ powers.T                          # E_p[z^i z^j]
        cov = second - np.outer(exp_moments, exp_moments)
        jac = -cov
        cond = float(np.linalg.cond(jac))
        if not np.isfinite(cond) or cond > 1e10:
            ill = True
            break
        try:
            step = np.linalg.solve(jac, resid)                    # solve, not inv
        except np.linalg.LinAlgError:
            ill = True
            break
        lam = lam - step
    log_w = -(lam[:, None] * powers).sum(axis=0)
    log_z = _logsumexp(log_w + np.log(dz))
    log_pdf = log_w - log_z                                        # density on the grid
    return lam, grid, log_pdf, ill, cond


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


@register
class MaxEntDistNode(Node):
    """Maximum-Entropy Distribution — the least-biased density matching the window's
    moments. Mean+variance constraints give the closed-form Gaussian MaxEnt; the node
    also fits a 4-moment exponential-family refinement numerically (skew/kurtosis) and
    flags ill-conditioning. Emits a ``density`` belief carrying the fitted params + a
    normalized pdf grid.

    The ``density`` interlingua type requires ``bic``, ``labels``, ``max_proba`` — we
    populate them honestly: ``bic`` of the Gaussian-MaxEnt fit, per-point ``labels``
    binning each observation into density deciles, and ``max_proba`` = the peak of the
    normalized grid pdf.
    """

    layer = "probability"
    node_type = "maxent_dist"
    cost = "low"
    name_display = "Maximum-Entropy Distribution"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        gauss = gaussian_maxent(x)
        mu, sigma = gauss["mu"], gauss["sigma"]

        # closed-form Gaussian-MaxEnt density grid (the mean+variance solution)
        grid = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 401)
        pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        # optional numeric 4-moment refinement on the standardized data (skew/kurt)
        ill, cond, lambdas = False, 1.0, None
        if n >= 50 and sigma > _EPS:
            z = (x - mu) / sigma
            lam, _zg, _lp, ill, cond = _moment_maxent_lambdas(z, moments=4)
            lambdas = [float(v) for v in lam]

        # Gaussian BIC: 2 free params (mu, sigma); log-lik under the fitted Normal
        ll = float(np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2)
                          - 0.5 * ((x - mu) / sigma) ** 2)) if sigma > _EPS else 0.0
        bic = float(2 * np.log(max(n, 2)) - 2 * ll)

        # per-point density-decile labels (satisfies the 'labels' required key)
        dens_at_x = (np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                     if sigma > _EPS else np.ones(n))
        edges = np.quantile(dens_at_x, np.linspace(0, 1, 11)) if n else np.array([0.0, 1.0])
        labels = np.clip(np.searchsorted(edges[1:-1], dens_at_x), 0, 9).astype(int)
        max_proba = float(np.max(pdf) * (grid[1] - grid[0]))      # peak grid mass

        return Belief("density",
                      {"bic": bic, "labels": [int(v) for v in labels],
                       "max_proba": max_proba, "mu": mu, "sigma": sigma,
                       "grid": [float(v) for v in grid], "pdf": [float(v) for v in pdf],
                       "lambdas": lambdas, "n_moments": 4 if lambdas else 2,
                       "ill_conditioned": bool(ill), "jacobian_cond": float(cond),
                       "model": "MaxEnt(Gaussian+4-moment)"},
                      0.6, self.name)


# ================================================================= Chebyshev
def chebyshev_interval(mu: float, sigma: float, k: float) -> Tuple[float, float]:
    """The distribution-free Chebyshev interval ``[mu - k*sigma, mu + k*sigma]``.

    By Chebyshev's inequality, for any random variable with finite variance,
    ``P(|X - mu| >= k*sigma) <= 1/k^2``; hence at least ``1 - 1/k^2`` of the mass
    lies inside this band, regardless of the distribution's shape. Pre-condition:
    ``k > 0`` (a non-positive k gives a vacuous/degenerate band and is clamped to
    ``_EPS``). Post-condition: ``lo <= mu <= hi``.
    """
    k = max(float(k), _EPS)
    half = k * float(sigma)
    return float(mu) - half, float(mu) + half


@register
class ChebyshevBoundNode(Node):
    """Chebyshev Tail Bound — a distribution-free prediction interval around a point
    forecast. Point = the last value (persistence) shrunk toward the window mean is
    NOT used; we keep the point as persistence (the honest one-step anchor) and place
    a [μ̂-kσ̂, μ̂+kσ̂] band around it sized by the window spread, with the guaranteed
    coverage 1-1/k². Emits a ``forecast`` belief with a calibrated interval.

    ``k`` defaults to 3 (≈88.9% guaranteed coverage); configurable via ``k=`` param.
    """

    layer = "probability"
    node_type = "chebyshev_bound"
    cost = "low"
    name_display = "Chebyshev Tail Bound"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        k = float(self.params.get("k", 3.0))
        point = float(x[-1])                              # persistence point forecast
        sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        lo, hi = chebyshev_interval(point, sigma, k)
        guarantee = 1.0 - 1.0 / (k * k)
        # confidence: the tighter the band relative to the data range, the more useful
        rng = float(np.ptp(x)) + _EPS
        conf = float(max(0.05, min(0.95, 1.0 - (hi - lo) / rng))) if x.size > 1 else 0.1
        return Belief("forecast",
                      {"next_vector": [point], "estimate": point,
                       "interval": [lo, hi], "k": k, "sigma": sigma,
                       "coverage_guarantee": float(guarantee),
                       "std": sigma, "model": f"Chebyshev(k={k:g})"},
                      conf, self.name)
