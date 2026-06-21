"""Phase 7a — classical econometrics nodes (PLAN.md §11, Block 57).

The classical time-series + volatility family the owner listed (AR/MA/ARMA/ARIMA/
SARIMA/VAR/GARCH/EGARCH/Prophet), implemented LIGHT-STACK (numpy/scipy only — no
statsmodels/fbprophet, per §0b), each behind the generic Node interface (Rule 23)
and emitting the existing `forecast` belief type (Block 46). Volatility models put
the conditional variance in the payload (like `kalman_filter`'s `variance` key).

* **Mean models:** `arma_forecast` (Hannan-Rissanen ARMA), `arima_forecast`
  (differenced ARMA), `sarima_forecast` (seasonal differencing + AR),
  `var_forecast` (vector autoregression; AR(1) fallback on 1-D).
* **Volatility models:** `garch_forecast` (GARCH(1,1) MLE), `egarch_forecast`
  (asymmetric/leverage), `ewma_volatility` (RiskMetrics λ=0.94).
* **Decomposition:** `holt_winters_forecast` (triple exponential smoothing),
  `prophet_like_forecast` (piecewise trend + Fourier seasonality via least squares).

Domain-agnostic: a generic (T, D) sequence in, a `forecast` Belief out.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


# ----------------------------------------------------------------- helpers
def _ar_coeffs(x: np.ndarray, p: int):
    n = len(x) - p
    if n <= p + 1:
        return None
    Xl = np.column_stack([x[p - 1 - k: p - 1 - k + n] for k in range(p)] + [np.ones(n)])
    beta, *_ = np.linalg.lstsq(Xl, x[p:], rcond=None)
    resid = x[p:] - Xl @ beta
    return beta, resid


def _hannan_rissanen(x: np.ndarray, p: int = 2, q: int = 2):
    """Two-stage ARMA(p,q): long-AR residuals, then regress on lagged values+resids.
    Returns the one-step forecast."""
    n = len(x)
    if n < p + q + 6:
        return float(x[-1]), 0.1
    long = min(10, n // 3)
    fit = _ar_coeffs(x, long)
    if fit is None:
        return float(x[-1]), 0.1
    _, resid = fit
    e = np.concatenate([np.zeros(n - len(resid)), resid])   # align residuals to x
    m = max(p, q)
    rows, y = [], []
    for t in range(m, n):
        rows.append([x[t - 1 - k] for k in range(p)] + [e[t - 1 - k] for k in range(q)] + [1.0])
        y.append(x[t])
    A = np.asarray(rows)
    beta, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    feat = [x[-1 - k] for k in range(p)] + [e[-1 - k] for k in range(q)] + [1.0]
    pred = float(np.asarray(feat) @ beta)
    nrmse = float(np.std(np.asarray(y) - A @ beta) / (np.std(y) + 1e-9))
    return pred, max(0.05, min(0.9, 1.0 - nrmse))


@register
class ARMANode(Node):
    """ARMA(p,q) via the Hannan-Rissanen two-stage estimator."""
    layer = "sequence"
    node_type = "arma_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        pred, conf = _hannan_rissanen(x, 2, 2)
        return Belief("forecast", {"next_vector": [pred], "model": "ARMA(2,2)"}, conf, self.name)


@register
class ARIMANode(Node):
    """ARIMA(p,1,q): difference once, ARMA-forecast the change, integrate back."""
    layer = "sequence"
    node_type = "arima_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if len(x) < 12:
            return Belief("forecast", {"next_vector": [float(x[-1])]}, 0.1, self.name)
        d = np.diff(x)
        dpred, conf = _hannan_rissanen(d, 2, 2)
        pred = float(x[-1] + dpred)
        return Belief("forecast", {"next_vector": [pred], "model": "ARIMA(2,1,2)"}, conf, self.name)


def _dominant_period(x: np.ndarray, max_p: int = 60) -> int:
    n = len(x)
    if n < 16:
        return 0
    z = np.diff(x)                     # detrend first: a trend/random-walk otherwise
    z = z - z.mean()                   # dominates low-lag autocorrelation and masks the season
    nz = len(z)
    ac = np.correlate(z, z, "full")[nz - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = 2
    seg = ac[lo: min(max_p, nz // 2)]
    if seg.size == 0:
        return 0
    k = int(np.argmax(seg)) + lo
    return k if ac[k] > 0.2 else 0


@register
class SARIMANode(Node):
    """Seasonal ARIMA (light): seasonal-difference at the detected dominant period,
    AR-forecast, integrate back. Falls back to ARIMA when no seasonality is found."""
    layer = "sequence"
    node_type = "sarima_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        m = _dominant_period(x)
        if m < 2 or len(x) < 2 * m + 6:
            d = np.diff(x)
            dpred, conf = _hannan_rissanen(d, 2, 1)
            return Belief("forecast", {"next_vector": [float(x[-1] + dpred)],
                                       "model": "ARIMA(2,1,1) [no season]"}, conf, self.name)
        sd = x[m:] - x[:-m]                       # seasonal difference
        dpred, conf = _hannan_rissanen(sd, 2, 1)
        pred = float(x[-m] + dpred)               # integrate seasonal diff
        return Belief("forecast", {"next_vector": [pred], "season": m,
                                   "model": f"SARIMA s={m}"}, conf, self.name)


@register
class VARNode(Node):
    """Vector Autoregression VAR(1): jointly regress all D channels on their last
    values; forecast the whole vector (channel 0 first). AR(1) when D==1."""
    layer = "sequence"
    node_type = "var_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        if T < D + 3:
            return Belief("forecast", {"next_vector": [float(X[-1, 0])]}, 0.1, self.name)
        Xp, Xn = X[:-1], X[1:]
        A = np.column_stack([Xp, np.ones(T - 1)])
        B, *_ = np.linalg.lstsq(A, Xn, rcond=None)        # (D+1, D)
        nxt = np.concatenate([X[-1], [1.0]]) @ B
        resid = Xn - A @ B
        nrmse = float(np.mean(np.std(resid, 0)) / (np.mean(np.std(Xn, 0)) + 1e-9))
        return Belief("forecast", {"next_vector": [float(v) for v in nxt],
                                   "model": f"VAR(1) D={D}"},
                      max(0.05, min(0.9, 1.0 - nrmse)), self.name)


# ----------------------------------------------------------- volatility
def _garch11_mle(r: np.ndarray):
    from scipy.optimize import minimize
    v0 = np.var(r) + 1e-9

    def negloglik(theta):
        w, a, b = theta
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        s2 = np.empty(len(r)); s2[0] = v0
        for t in range(1, len(r)):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        s2 = np.maximum(s2, 1e-12)
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    res = minimize(negloglik, [v0 * 0.1, 0.08, 0.88], method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-6})
    w, a, b = res.x
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        w, a, b = v0 * 0.1, 0.08, 0.88
    s2 = v0
    for t in range(1, len(r)):
        s2 = w + a * r[t - 1] ** 2 + b * s2
    return w + a * r[-1] ** 2 + b * s2, (float(w), float(a), float(b))


@register
class GARCHNode(Node):
    """GARCH(1,1) conditional-variance forecast (MLE) — the volatility workhorse.
    The level forecast is unchanged; the payload carries next-step variance/vol."""
    layer = "sequence"
    node_type = "garch_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        r = np.diff(x)
        if len(r) < 20:
            return Belief("forecast", {"next_vector": [float(x[-1])],
                                       "variance": float(np.var(r) + 1e-9)}, 0.2, self.name)
        r = r - r.mean()
        try:
            var_next, (w, a, b) = _garch11_mle(r)
        except Exception:
            var_next, (w, a, b) = float(np.var(r)), (0.0, 0.0, 0.0)
        var_next = float(max(var_next, 1e-12))
        return Belief("forecast", {"next_vector": [float(x[-1])], "variance": var_next,
                                   "volatility": float(np.sqrt(var_next)),
                                   "params": {"omega": w, "alpha": a, "beta": b},
                                   "persistence": a + b, "model": "GARCH(1,1)"},
                      max(0.1, min(0.9, a + b)), self.name)


@register
class EGARCHNode(Node):
    """EGARCH(1,1) — log-variance with asymmetry (leverage effect: down-moves raise
    vol more than up-moves). Light MLE."""
    layer = "sequence"
    node_type = "egarch_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        r = np.diff(x)
        if len(r) < 24:
            return Belief("forecast", {"next_vector": [float(x[-1])],
                                       "variance": float(np.var(r) + 1e-9)}, 0.2, self.name)
        r = r - r.mean()
        from scipy.optimize import minimize
        c = np.sqrt(2 / np.pi)

        def nll(theta):
            w, a, g, b = theta
            if abs(b) >= 0.999:
                return 1e10
            ln = np.log(np.var(r) + 1e-9)
            tot = 0.0
            for t in range(len(r)):
                s2 = np.exp(ln)
                z = r[t] / (np.sqrt(s2) + 1e-9)
                tot += 0.5 * (ln + r[t] ** 2 / (s2 + 1e-12))
                ln = w + b * ln + a * (abs(z) - c) + g * z
            return tot

        try:
            res = minimize(nll, [0.0, 0.1, -0.05, 0.9], method="Nelder-Mead",
                           options={"maxiter": 300})
            w, a, g, b = res.x
        except Exception:
            w, a, g, b = 0.0, 0.1, -0.05, 0.9
        ln = np.log(np.var(r) + 1e-9)
        for t in range(len(r)):
            z = r[t] / (np.sqrt(np.exp(ln)) + 1e-9)
            ln = w + b * ln + a * (abs(z) - c) + g * z
        var_next = float(min(max(np.exp(ln), 1e-12), 1e6))
        return Belief("forecast", {"next_vector": [float(x[-1])], "variance": var_next,
                                   "volatility": float(np.sqrt(var_next)),
                                   "leverage": float(g), "model": "EGARCH(1,1)"},
                      0.5, self.name)


@register
class EWMAVolatilityNode(Node):
    """RiskMetrics EWMA volatility (λ=0.94) — parameter-free, robust conditional
    variance forecast."""
    layer = "sequence"
    node_type = "ewma_volatility"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        r = np.diff(x)
        if len(r) < 3:
            return Belief("forecast", {"next_vector": [float(x[-1])],
                                       "variance": float(np.var(r) + 1e-9)}, 0.2, self.name)
        lam, s2 = 0.94, float(np.var(r[: max(3, len(r) // 5)]) + 1e-9)
        for t in range(1, len(r)):
            s2 = lam * s2 + (1 - lam) * r[t - 1] ** 2
        s2 = lam * s2 + (1 - lam) * r[-1] ** 2
        return Belief("forecast", {"next_vector": [float(x[-1])], "variance": float(s2),
                                   "volatility": float(np.sqrt(s2)), "lambda": lam,
                                   "model": "EWMA/RiskMetrics"}, 0.5, self.name)


# ----------------------------------------------------------- decomposition
@register
class HoltWintersNode(Node):
    """Holt-Winters triple exponential smoothing (additive: level + trend + season)."""
    layer = "sequence"
    node_type = "holt_winters_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        m = _dominant_period(x)
        if m < 2 or len(x) < 2 * m + 4:
            # fall back to double (Holt linear)
            a, b = 0.5, 0.3
            lvl, tr = x[0], x[1] - x[0]
            for t in range(1, len(x)):
                pl = lvl
                lvl = a * x[t] + (1 - a) * (lvl + tr)
                tr = b * (lvl - pl) + (1 - b) * tr
            return Belief("forecast", {"next_vector": [float(lvl + tr)],
                                       "model": "Holt linear [no season]"}, 0.4, self.name)
        a, b, g = 0.3, 0.1, 0.3
        lvl = x[:m].mean()
        tr = (x[m:2 * m].mean() - x[:m].mean()) / m
        season = list(x[:m] - lvl)
        for t in range(len(x)):
            s = season[t % m]
            pl = lvl
            lvl = a * (x[t] - s) + (1 - a) * (lvl + tr)
            tr = b * (lvl - pl) + (1 - b) * tr
            season[t % m] = g * (x[t] - lvl) + (1 - g) * s
        pred = float(lvl + tr + season[len(x) % m])
        return Belief("forecast", {"next_vector": [pred], "season": m,
                                   "model": "Holt-Winters"}, 0.45, self.name)


@register
class ProphetLikeNode(Node):
    """Prophet-style additive decomposition: piecewise-linear trend + Fourier
    seasonality fit by least squares; forecast one step ahead. No fbprophet dep."""
    layer = "sequence"
    node_type = "prophet_like_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = len(x)
        if n < 12:
            return Belief("forecast", {"next_vector": [float(x[-1])]}, 0.1, self.name)
        t = np.arange(n)
        cols = [np.ones(n), t]                          # intercept + linear trend
        # one knot at the midpoint -> piecewise-linear trend
        knot = n // 2
        cols.append(np.maximum(0, t - knot))
        m = _dominant_period(x)
        if m >= 2:
            for k in (1, 2, 3):
                cols.append(np.sin(2 * np.pi * k * t / m))
                cols.append(np.cos(2 * np.pi * k * t / m))
        A = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(A, x, rcond=None)
        tf = n
        frow = [1.0, tf, max(0, tf - knot)]
        if m >= 2:
            for k in (1, 2, 3):
                frow += [np.sin(2 * np.pi * k * tf / m), np.cos(2 * np.pi * k * tf / m)]
        pred = float(np.asarray(frow) @ beta)
        nrmse = float(np.std(x - A @ beta) / (np.std(x) + 1e-9))
        return Belief("forecast", {"next_vector": [pred], "season": m,
                                   "model": "Prophet-like"},
                      max(0.05, min(0.9, 1.0 - nrmse)), self.name)
