"""TIER-1 directly-buildable classics (CONCEPT_EQUATION_BANK.md), built ONE AT A TIME.

Light-stack only (numpy/scipy), each behind the generic Node interface (Rule 23) and
emitting an existing v0.1 interlingua belief type. Domain-agnostic: a generic (T, D)
sequence in, a Belief out — no candle/orderbook knowledge.

Built so far:
  * ``evt_tail_risk`` — Extreme Value Theory tail estimator (Hill index + GPD
    Peaks-Over-Threshold → tail index, VaR, Expected Shortfall). A RISK tool, judged
    by correctness (it recovers known tail indices), not point-forecast PnL.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


# ============================================================ EVT helpers
def hill_tail_index(x: np.ndarray, k_frac: float = 0.05) -> float:
    """Hill estimator of the tail index α from the upper-tail order statistics.

    For data with a Pareto-type tail P(X>x) ~ x^-α, the Hill estimator of the top-k
    log-spacings converges to 1/α, so α̂ = 1/H. Larger α = thinner tail.

    Parameters
    ----------
    x : array of magnitudes (non-negative scale; abs is applied defensively).
    k_frac : fraction of the largest observations used for the tail fit.

    Returns ``inf`` for a degenerate/too-short tail (no estimable heaviness).
    """
    x = np.sort(np.abs(np.asarray(x, dtype=float)))[::-1]   # descending
    n = x.size
    if n < 10:
        return float("inf")
    k = min(max(2, int(k_frac * n)), n - 1)
    threshold = max(x[k], 1e-12)
    top = x[:k]
    hill = float(np.mean(np.log(top / threshold)))
    return 1.0 / hill if hill > 1e-12 else float("inf")


def gpd_pot_fit(excess: np.ndarray) -> tuple[float, float]:
    """Fit a Generalized Pareto Distribution to threshold EXCESSES by method-of-moments.

    GPD(ξ, β) has mean m = β/(1−ξ) and variance v = β²/[(1−ξ)²(1−2ξ)], so
    m²/v = 1−2ξ ⇒ ξ̂ = (1 − m²/v)/2 and β̂ = m̄(1 − ξ̂). MoM is optimizer-free and
    stable for ξ < 1/2 (the finite-variance regime that covers real return tails).

    Returns (xi, beta); beta is floored positive.
    """
    e = np.asarray(excess, dtype=float)
    e = e[e > 0]
    if e.size < 5:
        return 0.0, max(float(np.std(e)) if e.size else 1.0, 1e-9)
    m = float(np.mean(e))
    v = float(np.var(e))
    if v <= 1e-18 or m <= 1e-18:
        return 0.0, max(m, 1e-9)
    xi = 0.5 * (1.0 - m * m / v)
    xi = float(np.clip(xi, -0.5, 0.49))     # keep below the infinite-variance boundary
    beta = max(m * (1.0 - xi), 1e-9)
    return xi, beta


def evt_var_es(x: np.ndarray, p: float = 0.99, q_thr: float = 0.95) -> tuple[float, float]:
    """POT (McNeil-Frey) Value-at-Risk and Expected Shortfall at level p on the UPPER tail.

    Threshold u = the q_thr empirical quantile; fit a GPD to the excesses, then
        VaR_p = u + (β/ξ)[ ((n/Nu)(1−p))^{−ξ} − 1 ],
        ES_p  = VaR_p/(1−ξ) + (β − ξu)/(1−ξ).
    Falls back to the empirical quantile when there are too few exceedances or ξ≈0.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    u = float(np.quantile(x, q_thr))
    excess = x[x > u] - u
    nu = excess.size
    if nu < 5:
        var = float(np.quantile(x, p))
        es = float(np.mean(x[x >= var])) if np.any(x >= var) else var
        return var, max(es, var)
    xi, beta = gpd_pot_fit(excess)
    ratio = (n / nu) * (1.0 - p)
    if abs(xi) < 1e-4:
        var = u + beta * (-np.log(ratio))            # ξ→0 exponential limit
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi) if xi < 1.0 else var
    return float(var), float(max(es, var))


# ============================================================ the node
@register
class EVTTailRiskNode(Node):
    """Extreme Value Theory tail-risk estimator (D1 / TIER-3-adjacent risk tool).

    Characterizes the heavy tail of a series: the Hill tail index α (smaller = heavier),
    the GPD shape ξ, and POT Value-at-Risk / Expected Shortfall at 99%. Emits a `signal`
    whose `series` is a per-point extremeness signal (robust |deviation| in MAD units) and
    whose payload carries the tail summary. Judged by correctness, not PnL.
    """

    layer = "signal"
    node_type = "evt_tail_risk"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        dev = x - np.median(x)
        absdev = np.abs(dev)
        mad = float(np.median(absdev)) or float(np.std(absdev)) or 1.0
        series = absdev / (mad + 1e-12)                 # per-point extremeness, finite, length n

        if n < 30:
            return Belief(
                type="signal",
                payload={"series": series.tolist(), "tail_index": 10.0, "xi": 0.0,
                         "var_99": float(np.max(absdev)) if n else 0.0, "es_99": 0.0,
                         "threshold": 0.0, "n_exceedances": 0, "note": "too-short-for-tail-fit"},
                confidence=float(np.clip((n - 20) / 400.0, 0.0, 0.9)),
            )

        tail_index = hill_tail_index(absdev, k_frac=0.10)
        u = float(np.quantile(absdev, 0.95))
        excess = absdev[absdev > u] - u
        xi, beta = gpd_pot_fit(excess)
        var_99, es_99 = evt_var_es(absdev, p=0.99, q_thr=0.95)
        # confidence grows with sample size and the number of tail exceedances available.
        conf = np.clip((n - 20) / 400.0, 0.0, 0.9) * np.clip(excess.size / 30.0, 0.3, 1.0)

        return Belief(
            type="signal",
            payload={
                "series": series.tolist(),
                "tail_index": float(tail_index if np.isfinite(tail_index) else 10.0),
                "xi": float(xi),
                "beta": float(beta),
                "var_99": float(var_99),
                "es_99": float(es_99),
                "threshold": u,
                "n_exceedances": int(excess.size),
            },
            confidence=float(np.clip(conf, 0.0, 1.0)),
        )
