"""TIER-1 directly-buildable classics (CONCEPT_EQUATION_BANK.md), built ONE AT A TIME.

Light-stack only (numpy/scipy), each behind the generic Node interface (Rule 23) and
emitting an existing v0.1 interlingua belief type. Domain-agnostic: a generic (T, D)
sequence in, a Belief out — no candle/orderbook knowledge.

Built so far:
  * ``evt_tail_risk`` — Extreme Value Theory tail estimator (Hill index + GPD
    Peaks-Over-Threshold → tail index, VaR, Expected Shortfall). A RISK tool, judged
    by correctness (it recovers known tail indices), not point-forecast PnL.
  * ``johansen_coint`` — Engle-Granger cointegration / stat-arb spread (own light-stack
    ADF). Hedge ratio + ADF stationarity test of the residual + half-life; emits a
    z-scored mean-reversion signal. D≥2 ⇒ test the pair; D==1 ⇒ test the level itself.
  * ``har_rv`` — Corsi (2009) Heterogeneous AR realized-volatility forecaster
    (daily/weekly/monthly RV regression). A VOL forecaster — vol is genuinely
    predictable, unlike returns — emits the next-step RV forecast.
  * ``copula_dependence`` — dependency structure separate from the marginals: Kendall's
    τ, Gaussian-copula ρ, and empirical upper/lower TAIL dependence (joint-crash risk).
  * ``bocpd_break`` — Bayesian Online Change-Point Detection (Adams-MacKay 2007,
    Gaussian model + constant hazard) → per-step change-point probability (anomaly).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats

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


# ============================================================ ADF / cointegration helpers
# Dickey-Fuller 5% critical values (constant, no trend): plain ADF ≈ -2.86; the
# Engle-Granger residual test is stricter (the residual is an ESTIMATED combination) so
# it uses MacKinnon's ≈ -3.34 (one regressor) bar instead.
_ADF_CRIT_5 = -2.86
_EG_CRIT_5 = -3.34


def adf_tstat(y: np.ndarray, p: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic for a unit root (with constant, no trend).

    Regress Δyₜ = α + γ yₜ₋₁ + Σᵢ δᵢ Δyₜ₋ᵢ + εₜ and return γ̂/se(γ̂). A unit root
    (random walk) gives γ≈0 → t near 0; mean reversion gives γ<0 → t very negative.
    More negative than the critical value ⇒ reject the unit root ⇒ stationary.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 4 * p + 12:
        p = 0
    dy = np.diff(y)                                   # dy[k] = Δy_{k+1}
    resp = dy[p:]                                     # Δyₜ for t = p+1 .. n-1
    ylag = y[p:n - 1]                                 # yₜ₋₁
    cols = [np.ones_like(resp), ylag]
    for i in range(1, p + 1):
        cols.append(dy[p - i:(n - 1) - i])            # Δyₜ₋ᵢ
    Xmat = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Xmat, resp, rcond=None)
    resid = resp - Xmat @ beta
    dof = max(Xmat.shape[0] - Xmat.shape[1], 1)
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(Xmat.T @ Xmat + 1e-12 * np.eye(Xmat.shape[1]))
    se_gamma = float(np.sqrt(max(s2 * xtx_inv[1, 1], 1e-30)))
    return float(beta[1] / se_gamma)


def _residual_half_life(resid: np.ndarray) -> float:
    """Mean-reversion half-life from an AR(1) fit of the residual: −ln2/ln(φ)."""
    r0, r1 = resid[:-1], resid[1:]
    phi = float(np.polyfit(r0, r1, 1)[0])
    if not (0.0 < phi < 1.0):
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))


def engle_granger(X: np.ndarray) -> dict:
    """Engle-Granger two-step cointegration test on a (T, D≥2) array.

    Regress column 0 on the rest (+ const) for the hedge ratio, then ADF-test the
    residual spread for stationarity. ``cointegrated`` iff the residual ADF stat clears
    the (stricter) Engle-Granger 5% bar. For D==1 the single column is treated as the
    candidate spread itself (plain ADF bar).
    """
    X = np.asarray(X, dtype=float)
    y = X[:, 0]
    if X.shape[1] == 1:
        resid = y - np.mean(y)
        stat = adf_tstat(y)
        return {"hedge_ratio": float("nan"), "intercept": float(np.mean(y)),
                "adf_stat": stat, "cointegrated": bool(stat < _ADF_CRIT_5),
                "half_life": _residual_half_life(resid), "spread": resid}
    Z = np.column_stack([np.ones(X.shape[0]), X[:, 1:]])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    stat = adf_tstat(resid)
    return {"hedge_ratio": float(coef[1]), "intercept": float(coef[0]),
            "adf_stat": stat, "cointegrated": bool(stat < _EG_CRIT_5),
            "half_life": _residual_half_life(resid), "spread": resid}


@register
class JohansenCointNode(Node):
    """Cointegration / stat-arb spread (D1). Emits the z-scored equilibrium residual as a
    mean-reversion ``signal`` plus the hedge ratio, ADF stationarity verdict and half-life."""

    layer = "signal"
    node_type = "johansen_coint"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        res = engle_granger(X)
        spread = res["spread"]
        z = (spread - np.mean(spread)) / (np.std(spread) + 1e-12)
        conf = float(np.clip(1.0 / (1.0 + abs(res["adf_stat"]) ** -1) if res["cointegrated"]
                             else 0.2, 0.0, 0.95)) if X.shape[0] >= 30 else 0.2
        return Belief(
            type="signal",
            payload={
                "series": z.tolist(),                 # mean-reversion signal (per-point)
                "hedge_ratio": res["hedge_ratio"],
                "intercept": res["intercept"],
                "adf_stat": res["adf_stat"],
                "cointegrated": res["cointegrated"],
                "half_life": res["half_life"],
            },
            confidence=conf,
        )


# ============================================================ HAR-RV (Corsi 2009)
def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing window-w mean ending at each index (shorter windows at the start)."""
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.empty_like(a, dtype=float)
    for i in range(a.size):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def har_fit_forecast(rv: np.ndarray, train_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the HAR-RV regression on the train split, forecast RV over the test split.

    RVₜ₊₁ = c + β_d·RVₜ + β_w·RV̄ₜ⁽⁵⁾ + β_m·RV̄ₜ⁽²²⁾.  Returns (har_forecast,
    naive_forecast=RVₜ, fitted_coefficients) aligned to the test targets RVₜ₊₁.
    """
    rv = np.asarray(rv, dtype=float).ravel()
    rv = np.clip(rv, 0.0, None)
    d, w, m = rv, _rolling_mean(rv, 5), _rolling_mean(rv, 22)
    feat = np.column_stack([np.ones(rv.size), d, w, m])[:-1]   # features at t
    target = rv[1:]                                            # RV_{t+1}
    n = target.size
    split = max(int(train_frac * n), 30)
    coef, *_ = np.linalg.lstsq(feat[:split], target[:split], rcond=None)
    fc = np.clip(feat[split:] @ coef, 0.0, None)
    naive = d[:-1][split:]                                     # RV_t over the test span
    return fc, naive, coef


@register
class HARRVNode(Node):
    """HAR realized-volatility forecaster (D1 vol). Column 0 is taken as a value series;
    its squared first-difference is the realized-variance proxy. Emits a ``forecast`` of
    the next-step RV (vol is predictable where returns are not)."""

    layer = "equation"
    node_type = "har_rv"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        rv = np.diff(x) ** 2 if x.size > 2 else np.array([0.0])
        rv = np.clip(rv, 0.0, None)
        d, w, m = rv, _rolling_mean(rv, 5), _rolling_mean(rv, 22)
        next_feat = np.array([1.0, d[-1], w[-1], m[-1]])
        if rv.size >= 40:
            fc, _, coef = har_fit_forecast(rv, train_frac=0.7)
            rv_forecast = float(max(np.dot(next_feat, coef), 0.0))
            series = np.concatenate([[rv[0]], np.clip(
                np.column_stack([np.ones(rv.size), d, w, m]) @ coef, 0.0, None)])
            conf = float(np.clip(rv.size / 500.0, 0.2, 0.9))
        else:
            rv_forecast = float(np.mean(rv))
            series = np.full(rv.size + 1, rv_forecast)
            coef = np.zeros(4)
            conf = 0.2
        return Belief(
            type="forecast",
            payload={
                "estimate": rv_forecast,
                "rv_forecast": rv_forecast,
                "vol_forecast": float(np.sqrt(rv_forecast)),
                "series": series.tolist(),
                "coef": coef.tolist(),
            },
            confidence=conf,
        )


# ============================================================ Copula dependence
def empirical_tail_dependence(u: np.ndarray, v: np.ndarray, q: float = 0.95) -> tuple[float, float]:
    """Empirical lower/upper tail-dependence coefficients of two uniform pseudo-samples.

    λ_U ≈ P(V>F⁻¹(q) | U>F⁻¹(q)); λ_L the mirror at the lower quantile 1−q.
    Tail dependence is the probability of JOINT extremes — the part a correlation misses.
    """
    u = np.asarray(u, float); v = np.asarray(v, float)
    uu, vu = np.quantile(u, q), np.quantile(v, q)
    mu = u > uu
    upper = float(np.mean(v[mu] > vu)) if mu.sum() else 0.0
    ul, vl = np.quantile(u, 1 - q), np.quantile(v, 1 - q)
    ml = u < ul
    lower = float(np.mean(v[ml] < vl)) if ml.sum() else 0.0
    return lower, upper


@register
class CopulaDependenceNode(Node):
    """Dependency structure separate from the marginals (D1, [I] risk). Reports Kendall's
    τ, Spearman ρ, the Gaussian-copula correlation ρ=sin(πτ/2) and the empirical upper/
    lower tail dependence (joint-crash co-movement). Emits a co-extremeness ``signal``."""

    layer = "signal"
    node_type = "copula_dependence"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        if X.shape[1] < 2:
            return Belief(type="signal",
                          payload={"series": [0.0] * T, "kendall_tau": 0.0,
                                   "note": "needs D>=2 for dependence"},
                          confidence=0.1)
        a, b = X[:, 0], X[:, 1]
        u = _stats.rankdata(a) / (T + 1)
        v = _stats.rankdata(b) / (T + 1)
        tau = float(_stats.kendalltau(a, b).statistic)
        rho_s = float(_stats.spearmanr(a, b).statistic)
        rho_gauss = float(np.sin(np.pi * tau / 2.0))
        q = 0.90 if T < 300 else 0.95
        lower, upper = empirical_tail_dependence(u, v, q=q)
        series = (4.0 * (u - 0.5) * (v - 0.5))         # +1 both-extreme-same-side, length T
        return Belief(
            type="signal",
            payload={
                "series": series.tolist(),
                "kendall_tau": tau,
                "spearman_rho": rho_s,
                "rho_gaussian_copula": rho_gauss,
                "tail_dep_upper": upper,
                "tail_dep_lower": lower,
            },
            confidence=float(np.clip(T / 500.0, 0.2, 0.9)),
        )


# ============================================================ BOCPD (Adams-MacKay 2007)
def bocpd_gaussian(x: np.ndarray, hazard: float = 1.0 / 100.0,
                   mu0: float | None = None, sigma2: float | None = None,
                   kappa0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian Online Change-Point Detection with a Gaussian (known-variance,
    Normal-prior-on-mean) observation model and a constant hazard.

    Returns (cp_score, run_length_posterior). NOTE: P(run length=0) is identically the
    hazard after normalisation (a known BOCPD property), so it is useless as a detector.
    The change-point SIGNAL is the COLLAPSE of the run-length posterior: cp_scoreₜ is the
    fractional drop in the expected run length E[rₜ] (∈[0,1], ≈0 while a regime persists,
    →1 the step a break is absorbed). O(n²); the caller bounds n.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if sigma2 is None:
        d = np.diff(x)
        sigma2 = float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2) if d.size else 1.0
        sigma2 = max(sigma2, 1e-9)
    if mu0 is None:
        mu0 = float(np.mean(x)) if n else 0.0
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    exp_runlen = np.zeros(n)
    mus = [mu0]; kappas = [kappa0]                     # posterior mean/pseudo-count per run length
    for t in range(n):
        mu_arr = np.array(mus); kap = np.array(kappas)
        pred_var = sigma2 * (1.0 + 1.0 / kap)
        pred = _stats.norm.pdf(x[t], mu_arr, np.sqrt(pred_var))
        prior = R[t, :t + 1]
        R[t + 1, 1:t + 2] = prior * pred * (1.0 - hazard)   # growth (run length r -> r+1)
        R[t + 1, 0] = float(np.sum(prior * pred * hazard))  # change (reset to 0)
        R[t + 1, :t + 2] /= R[t + 1, :t + 2].sum() + 1e-300
        exp_runlen[t] = float(np.dot(np.arange(t + 2), R[t + 1, :t + 2]))
        new_mus = [mu0]; new_kappas = [kappa0]         # run length 0 resets to the prior
        for k, mval in zip(kappas, mus):
            new_kappas.append(k + 1.0)
            new_mus.append((k * mval + x[t]) / (k + 1.0))
        mus, kappas = new_mus, new_kappas
    cp_score = np.zeros(n)
    drop = exp_runlen[:-1] - exp_runlen[1:]            # positive when E[r] collapses
    cp_score[1:] = np.clip(drop, 0.0, None) / (exp_runlen[:-1] + 1.0)
    return cp_score, R


@register
class BOCPDBreakNode(Node):
    """Bayesian online change-point detector (D1 regime). Emits an ``anomaly`` belief whose
    per-step ``scores`` are the change-point probabilities; ``flags`` mark probable breaks.
    Input longer than 1200 points is tail-truncated (the O(n²) recursion is bounded)."""

    layer = "signal"
    node_type = "bocpd_break"
    requires_y = False
    is_transformer = False
    cost = "med"

    _MAX_N = 1200

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n_full = x.size
        pad = 0
        if n_full > self._MAX_N:                       # bound the O(n^2) recursion
            pad = n_full - self._MAX_N
            x = x[-self._MAX_N:]
        cp, _ = bocpd_gaussian(x, hazard=1.0 / 100.0)
        thr = max(0.2, float(np.quantile(cp, 0.95))) if cp.size else 0.5
        flags_tail = (cp > thr)
        scores = np.concatenate([np.zeros(pad), cp]).tolist()
        flags = ([False] * pad) + flags_tail.tolist()
        n_anom = int(np.sum(flags_tail))
        return Belief(
            type="anomaly",
            payload={
                "scores": scores,
                "flags": flags,
                "n_anomalies": n_anom,
                "fraction": float(n_anom / max(n_full, 1)),
                "threshold": thr,
            },
            confidence=float(np.clip(n_full / 500.0, 0.2, 0.9)),
        )
