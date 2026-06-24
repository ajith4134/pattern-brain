"""Tier-3 kinematics & pricing foundations (kinematics / optimal-control / pricing).

Three classical-physics / quantitative-finance foundations, each wrapped behind the
generic Node interface (Rule 23 — a (T, D) sequence in, a Belief out; nothing inside
knows about candles or order books). The closed-form math is exposed as standalone,
oracle-testable helpers (``kinematic_next_step``, ``almgren_holdings``, ``bs_price``,
``bs_greeks``) so correctness is provable independently of the Node wrapper.

* ``momentum_kinematics`` (layer ``sequence``) — Newton/Leibniz constant-acceleration
  step. Treats the series as a 1-D position: velocity v = 1st difference, acceleration
  a = 2nd difference, then the kinematic extrapolation x_{t+1} = x_t + v + 0.5*a.
  Emits a ``forecast``. Honestly a momentum/inertia forecaster — judged vs persistence.
* ``almgren_exec`` (layer ``decision``) — Almgren-Chriss optimal execution (the
  Pontryagin-maximum-principle / discrete-Euler-Lagrange solution of the
  impact+risk cost functional). The optimal liquidation holdings follow the sinh
  schedule x_k = X*sinh(kappa(N-k))/sinh(kappa N); the node outputs the next-step
  optimal trade fraction. Emits a ``decision``. Judged by design-job correctness
  (matches the closed form; reduces to linear/TWAP as risk-aversion -> 0).
* ``bs_pricing`` (layer ``signal``) — Black-Scholes-Merton pricing utility. The
  closed-form European option price + Greeks (a PRICING IDENTITY, not a market
  forecaster). As a Node it derives a generic, finite "fair-value vol" signal series
  (rolling realized volatility of the window). Validated by put-call parity, Greek
  signs, and monotonicity in vol — a correctness-checked UTILITY, not a predictor.

Numerical policy (CLAUDE.md Phase 3): float64 throughout; stable closed forms;
explicit limits at the degenerate boundaries (kappa->0, T->0, sigma->0).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import norm

from ..belief import Belief
from ..node import Node
from ..registry import register


# =====================================================================
# 1. Kinematics — constant-acceleration one-step extrapolation
# =====================================================================
def kinematic_next_step(x: np.ndarray) -> float:
    """One-step constant-acceleration forecast of a 1-D position series.

    Treat the series as position sampled at unit time step. Newton/Leibniz kinematics:
        velocity      v = x_t - x_{t-1}                  (1st difference)
        acceleration  a = x_t - 2 x_{t-1} + x_{t-2}      (2nd difference)
        x_{t+1} ~= x_t + v + 0.5 * a + 0.5 * a = x_t + v + a.

    The kinematic half-step x_t + v + 0.5 a is the *instantaneous* extrapolation, but
    the BACKWARD velocity v = x_t - x_{t-1} already lags the true velocity at t by
    half an acceleration step (v_true(t) = v + 0.5 a); adding the remaining 0.5 a gives
    the discrete constant-acceleration step x_t + v + a == 3 x_t - 3 x_{t-1} + x_{t-2}.
    On a true constant-acceleration series x_k = x0 + v0 k + 0.5 a0 k^2 this recovers
    x_{t+1} EXACTLY (it is the exact forward extrapolation of a quadratic from 3 points).

    Pre-conditions: ``x`` is a finite 1-D array of length >= 1.
    Post-condition: a finite float. Degenerate fallbacks:
        len == 1 -> x_0 (persistence); len == 2 -> x_1 (persistence, no acceleration
        estimable). The 2-point case deliberately does NOT extrapolate velocity,
        keeping the node a pure persistence baseline until acceleration is defined.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        return float(x[-1])
    velocity = x[-1] - x[-2]
    acceleration = x[-1] - 2.0 * x[-2] + x[-3]
    return float(x[-1] + velocity + acceleration)


@register
class MomentumKinematicsNode(Node):
    """Momentum/Inertia Forecaster — constant-acceleration kinematic step
    (Newton/Leibniz). Treats price as position; extrapolates one step from the
    last velocity and acceleration. Emits a ``forecast`` belief."""
    layer = "sequence"
    node_type = "momentum_kinematics"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        pred = kinematic_next_step(x)
        if not math.isfinite(pred):
            pred = float(x[-1])
        # confidence: how steady the acceleration is over the window (a smooth,
        # constant-acceleration regime is where this forecaster is trustworthy).
        conf = 0.3
        if x.size >= 4:
            accel = np.diff(x, n=2)
            denom = float(np.mean(np.abs(accel))) + 1e-9
            steadiness = 1.0 - float(np.std(accel)) / (denom + float(np.std(accel)) + 1e-9)
            conf = float(max(0.05, min(0.9, steadiness)))
        velocity = float(x[-1] - x[-2]) if x.size >= 2 else 0.0
        acceleration = float(x[-1] - 2 * x[-2] + x[-3]) if x.size >= 3 else 0.0
        return Belief("forecast",
                      {"next_vector": [pred], "velocity": velocity,
                       "acceleration": acceleration, "model": "constant-acceleration kinematics"},
                      conf, self.name)


# =====================================================================
# 2. Almgren-Chriss optimal execution — sinh liquidation schedule
# =====================================================================
def almgren_kappa(risk_aversion: float, sigma: float, eta: float, tau: float) -> float:
    """The Almgren-Chriss urgency parameter kappa (per unit time).

    kappa = (1/tau) * arccosh(1 + 0.5 * lambda * sigma^2 * tau^2 / eta), where
    lambda is risk-aversion, sigma the volatility, eta the temporary-impact
    coefficient, tau the step length. lambda -> 0 (risk-neutral) gives kappa -> 0,
    i.e. the linear/TWAP schedule. All inputs must be >= 0; eta, tau > 0.
    """
    arg = 1.0 + 0.5 * risk_aversion * sigma * sigma * tau * tau / eta
    arg = max(arg, 1.0)                       # arccosh domain is [1, inf)
    return float(math.acosh(arg) / tau)


def almgren_holdings(X: float, N: int, kappa: float, tau: float = 1.0) -> np.ndarray:
    """Optimal liquidation HOLDINGS trajectory x_0..x_N (Almgren-Chriss closed form).

        x_k = X * sinh(kappa * (N - k) * tau) / sinh(kappa * N * tau),   k = 0..N

    Pre-conditions: X is the initial inventory (any sign; scale-free), N >= 1 integer,
    kappa >= 0, tau > 0. Post-conditions: length N+1; x_0 == X; x_N == 0; the
    trajectory is monotone toward 0. As kappa -> 0 it reduces to the linear (TWAP)
    schedule x_k = X*(1 - k/N) (handled by the explicit limit to avoid 0/0).
    """
    if N < 1:
        raise ValueError("N must be >= 1 (at least one execution step)")
    k = np.arange(N + 1, dtype=float)
    kN = kappa * N * tau
    if kN < 1e-6:                              # kappa -> 0: exact linear/TWAP limit
        return X * (1.0 - k / N)
    return X * np.sinh(kappa * (N - k) * tau) / np.sinh(kN)


def almgren_next_fraction(X: float, N: int, kappa: float, tau: float = 1.0) -> float:
    """Fraction of the initial inventory to trade in the FIRST step: (x_0 - x_1)/X.
    Equals 1/N as kappa -> 0 (equal TWAP slices); larger (front-loaded) as urgency rises."""
    h = almgren_holdings(X, N, kappa, tau)
    if X == 0:
        return 0.0
    return float((h[0] - h[1]) / X)


@register
class AlmgrenExecutionNode(Node):
    """Almgren-Chriss Optimal Executor — Pontryagin-optimal liquidation schedule.

    The series' recent volatility sets the urgency kappa; the node returns the
    optimal next-step trade fraction of a unit position to liquidate over a fixed
    horizon. Emits a ``decision`` belief (the execution fraction in the payload).
    This is a control/decision utility — judged by schedule correctness, not by
    forecasting the series."""
    layer = "decision"
    node_type = "almgren_exec"

    def __init__(self, horizon: int = 10, risk_aversion: float = 1e-6,
                 eta: float = 1.0, **kw):
        super().__init__(horizon=horizon, risk_aversion=risk_aversion, eta=eta, **kw)
        self.horizon = int(horizon)
        self.risk_aversion = float(risk_aversion)
        self.eta = float(eta)

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        # generic, scale-free volatility estimate from the window's one-step changes
        sigma = float(np.std(np.diff(x))) if x.size > 2 else 0.0
        kappa = almgren_kappa(self.risk_aversion, sigma, self.eta, tau=1.0)
        N = max(1, self.horizon)
        frac = almgren_next_fraction(1.0, N, kappa, tau=1.0)
        frac = float(min(1.0, max(1.0 / N, frac)))          # bounded to [1/N, 1]
        # discrete decision: 1 = act now (front-loaded / urgent), 0 = hold (patient/TWAP)
        action = int(frac > 1.5 / N)
        # confidence rises with urgency (how front-loaded the optimal schedule is)
        conf = float(min(0.9, max(0.1, frac)))
        return Belief("decision",
                      {"action": action, "execution_fraction": frac,
                       "kappa": float(kappa), "horizon": N,
                       "sigma": sigma, "model": "Almgren-Chriss sinh schedule"},
                      conf, self.name)


# =====================================================================
# 3. Black-Scholes-Merton pricing utility
# =====================================================================
def _bs_d1_d2(S: float, K: float, r: float, T: float, sigma: float):
    """The Black-Scholes d1, d2 (caller guarantees T>0, sigma>0, S>0, K>0)."""
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S: float, K: float, r: float, T: float, sigma: float,
             call: bool = True) -> float:
    """Black-Scholes-Merton European option price (no dividends).

        C = S*N(d1) - K*e^{-rT}*N(d2),   P = K*e^{-rT}*N(-d2) - S*N(-d1)
        d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma*sqrt(T)),  d2 = d1 - sigma*sqrt(T)

    Pre-conditions: S>0, K>0, T>=0, sigma>=0. At the boundary T->0 or sigma->0 the
    price collapses to the discounted intrinsic value (the no-uncertainty limit),
    handled explicitly to avoid division by zero. Post-condition: a finite,
    non-negative price that satisfies put-call parity C - P = S - K e^{-rT}."""
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")
    disc_k = K * math.exp(-r * T)
    if T == 0 or sigma == 0:                              # degenerate: intrinsic value
        intrinsic = (S - disc_k) if call else (disc_k - S)
        return float(max(intrinsic, 0.0))
    d1, d2 = _bs_d1_d2(S, K, r, T, sigma)
    if call:
        return float(S * norm.cdf(d1) - disc_k * norm.cdf(d2))
    return float(disc_k * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S: float, K: float, r: float, T: float, sigma: float,
              call: bool = True) -> Dict[str, float]:
    """Closed-form first-order Greeks (delta, gamma, vega, theta, rho).

    delta in [0,1] for calls / [-1,0] for puts; gamma>0; vega>0 (all by construction
    for T,sigma>0). At T->0 / sigma->0 returns the degenerate limits (a step-function
    delta, zero gamma/vega)."""
    if T <= 0 or sigma <= 0:
        itm = S > K * math.exp(-r * T)
        delta = (1.0 if itm else 0.0) if call else (-1.0 if not itm else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _bs_d1_d2(S, K, r, T, sigma)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = math.sqrt(T)
    disc_k = K * math.exp(-r * T)
    gamma = pdf_d1 / (S * sigma * sqrt_t)
    vega = S * pdf_d1 * sqrt_t
    if call:
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_t)) - r * disc_k * norm.cdf(d2)
        rho = T * disc_k * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_t)) + r * disc_k * norm.cdf(-d2)
        rho = -T * disc_k * norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


def _rolling_realized_vol(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation of one-step changes (a generic 'fair-value vol'
    feature). Returns a (T,) array aligned to x; the first ``window`` points use the
    expanding estimate so the series stays finite and full-length (Rule 23)."""
    d = np.diff(x, prepend=x[0])                      # length T, first = 0
    T = len(x)
    out = np.empty(T, dtype=float)
    for t in range(T):
        lo = max(0, t - window + 1)
        seg = d[lo:t + 1]
        out[t] = float(np.std(seg)) if seg.size > 1 else 0.0
    return out


@register
class BlackScholesPricingNode(Node):
    """Black-Scholes Fair-Value-Vol — a pricing UTILITY (correctness-validated by
    put-call parity / Greek signs / vol-monotonicity), NOT a baseline-beating
    predictor. As a Node it turns the window into a generic finite signal series:
    the rolling realized volatility ("fair-value vol") plus an at-the-money BS
    reference price computed from that vol. Emits a ``signal`` belief."""
    layer = "signal"
    node_type = "bs_pricing"
    is_transformer = True

    def __init__(self, window: int = 20, rate: float = 0.0, ttm: float = 1.0, **kw):
        super().__init__(window=window, rate=rate, ttm=ttm, **kw)
        self.window = int(window)
        self.rate = float(rate)
        self.ttm = float(ttm)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Per-feature rolling realized-vol series — a finite generic (T, D) signal."""
        cols = [_rolling_realized_vol(X[:, j], self.window) for j in range(X.shape[1])]
        return np.column_stack(cols)

    def _predict(self, X: np.ndarray) -> Belief:
        vol_series = self._transform(X)                   # (T, D) fair-value vol
        x = X[:, 0].astype(float)
        spot = float(abs(x[-1])) + 1e-9
        # realized-vol of the window as a per-period sigma; annualize-free (generic)
        sigma = float(vol_series[-1, 0]) / spot           # scale-free vol of returns
        sigma = max(sigma, 1e-6)
        atm_price = bs_price(spot, spot, self.rate, self.ttm, sigma, call=True)
        greeks = bs_greeks(spot, spot, self.rate, self.ttm, sigma, call=True)
        mean_vol = float(np.mean(vol_series))
        return Belief("signal",
                      {"series": vol_series.tolist(),
                       "mean_abs_change": mean_vol,
                       "fair_value_vol": sigma,
                       "atm_call_price": float(atm_price),
                       "delta": greeks["delta"], "vega": greeks["vega"],
                       "model": "Black-Scholes rolling realized-vol"},
                      float(min(0.9, max(0.1, mean_vol / (spot + 1e-9)))), self.name)


__all__ = [
    "kinematic_next_step", "almgren_kappa", "almgren_holdings", "almgren_next_fraction",
    "bs_price", "bs_greeks",
    "MomentumKinematicsNode", "AlmgrenExecutionNode", "BlackScholesPricingNode",
]
