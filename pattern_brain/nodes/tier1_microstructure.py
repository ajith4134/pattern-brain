"""TIER-1 MICROSTRUCTURE nodes (CONCEPT_EQUATION_BANK.md), built oracle-test-first.

Light-stack only (numpy/scipy), each behind the generic Node interface (Rule 23)
and emitting an existing v0.1 interlingua belief type. Domain-agnostic: a generic
(T, D) sequence in, a Belief out — NO candle/orderbook knowledge lives in the node.
The eval harness (tests/test_tier1_microstructure.py) owns the §G mapping from raw
trades (times, price, qty, side) into the (T, D) each node consumes.

Built here:
  * ``ofi`` — Order Flow Imbalance. Rolling net buy/sell pressure OFIₜ = Σ signed
    volume over a window. col0 = signed volume; optional col1 = price (level) used
    only to REPORT corr(OFI, contemporaneous return). Emits a `signal`.
  * ``vpin`` — Volume-Synchronized Probability of Informed Trading (Easley-López de
    Prado-O'Hara 2012). Equal-volume buckets, per-bucket |V_buy−V_sell|/(V_buy+V_sell);
    VPIN = mean over buckets. col0 = signed volume; sign ⇒ buy/sell. Emits a `signal`.
  * ``kyle_impact`` — Kyle's (1985) lambda price impact. Regress ΔP on signed volume
    ΔP = λ·sv + c; λ = illiquidity (price moved per unit signed flow). col0 = signed
    volume, col1 = price level (ΔP taken internally). Emits an `equation`.

All three are domain-agnostic flow signals/estimators judged by CORRECTNESS (they
recover a known synthetic answer and behave sanely on real trades), not by PnL.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


# ============================================================ OFI helpers
def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing window-w sum ending at each index (shorter windows at the start).

    Length-preserving: out[i] = Σ a[max(0, i-w+1) .. i]. window<=1 is a no-op copy.
    """
    a = np.asarray(a, dtype=float)
    if window <= 1:
        return a.copy()
    c = np.cumsum(np.insert(a, 0, 0.0))
    lo = np.clip(np.arange(a.size) - window + 1, 0, None)
    return c[np.arange(a.size) + 1] - c[lo]


@register
class OrderFlowImbalanceNode(Node):
    """Order Flow Imbalance (microstructure). Column 0 is a signed-flow series
    (positive = buy-initiated volume, negative = sell-initiated); the rolling sum
    over ``window`` steps is the net buy/sell pressure OFIₜ. If a second column is
    present it is treated as a price level used ONLY to report the contemporaneous
    correlation corr(OFI, Δlog-price). Emits a `signal` whose ``series`` is the
    per-step OFI."""

    layer = "signal"
    node_type = "ofi"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        sv = X[:, 0].astype(float)
        window = max(1, int(self.params.get("window", 20)))
        ofi = rolling_sum(sv, window)
        net_imbalance = float(np.sum(sv))

        corr = 0.0
        if X.shape[1] >= 2 and X.shape[0] >= 3:
            price = X[:, 1].astype(float)
            ret = np.empty_like(price)
            ret[0] = 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[1:] = np.diff(np.log(np.clip(price, 1e-300, None)))
            ret = np.where(np.isfinite(ret), ret, 0.0)
            if np.std(ofi) > 1e-12 and np.std(ret) > 1e-12:
                corr = float(np.corrcoef(ofi, ret)[0, 1])
                corr = corr if np.isfinite(corr) else 0.0

        conf = float(np.clip(X.shape[0] / 500.0, 0.2, 0.9))
        return Belief(
            type="signal",
            payload={
                "series": ofi.tolist(),
                "net_imbalance": net_imbalance,
                "window": window,
                "corr_with_return": corr,
            },
            confidence=conf,
        )


# ============================================================ VPIN helpers
def equal_volume_buckets(volume: np.ndarray, n_buckets: int) -> np.ndarray:
    """Assign each trade to one of ``n_buckets`` equal-(total)-volume buckets.

    The bucket boundaries are the cumulative-volume quantiles, so each bucket holds
    ~Σvolume/n_buckets units of volume (volume-clock, not trade-count). Returns an
    int bucket index per trade in [0, n_buckets-1].
    """
    cum = np.cumsum(np.abs(volume))
    total = cum[-1]
    if total <= 0:
        return np.zeros(volume.size, dtype=int)
    edges = np.linspace(0.0, total, n_buckets + 1)[1:-1]
    return np.clip(np.searchsorted(edges, cum, side="left"), 0, n_buckets - 1)


@register
class VPINNode(Node):
    """Volume-Synchronized Probability of Informed Trading (Easley-López de Prado-
    O'Hara 2012). Column 0 is signed volume (sign = buy/sell, magnitude = size).
    Trades are bucketed into ``n_buckets`` equal-volume buckets; per bucket
    VPINᵦ = |V_buy − V_sell| / (V_buy + V_sell) ∈ [0, 1]. VPIN = mean over buckets
    (flow toxicity / order-flow imbalance under the volume clock). If a price column
    is present, the per-bucket realized volatility is reported for the toxicity↔vol
    check. Emits a `signal` whose ``series`` is the per-bucket VPIN."""

    layer = "signal"
    node_type = "vpin"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        sv = X[:, 0].astype(float)
        n_buckets = max(1, int(self.params.get("n_buckets", 50)))
        n_buckets = min(n_buckets, max(1, sv.size))
        vol = np.abs(sv)
        buyers = np.clip(sv, 0.0, None)
        sellers = np.clip(-sv, 0.0, None)
        idx = equal_volume_buckets(vol, n_buckets)

        price = X[:, 1].astype(float) if X.shape[1] >= 2 else None
        vpin = np.zeros(n_buckets)
        bucket_rv = np.zeros(n_buckets)
        for b in range(n_buckets):
            m = idx == b
            vb, vs = float(buyers[m].sum()), float(sellers[m].sum())
            denom = vb + vs
            vpin[b] = abs(vb - vs) / denom if denom > 0 else 0.0
            if price is not None and m.sum() >= 2:
                p = price[m]
                bucket_rv[b] = float(np.sqrt(np.sum(np.diff(np.log(
                    np.clip(p, 1e-300, None))) ** 2)))

        mean_vpin = float(np.mean(vpin)) if n_buckets else 0.0
        conf = float(np.clip(n_buckets / 50.0, 0.2, 0.9))
        return Belief(
            type="signal",
            payload={
                "series": vpin.tolist(),
                "mean_vpin": mean_vpin,
                "n_buckets": int(n_buckets),
                "bucket_rv": bucket_rv.tolist(),
            },
            confidence=conf,
        )


# ============================================================ Kyle's lambda
def fit_kyle_lambda(signed_volume: np.ndarray, dprice: np.ndarray) -> tuple[float, float, float]:
    """OLS of ΔP on signed volume with intercept: ΔP = λ·sv + c.

    Solved via ``np.linalg.solve`` on the (well-conditioned, 2×2) normal equations
    rather than ``inv``; returns (lambda, intercept, r2). λ is Kyle's price-impact /
    illiquidity coefficient — price moved per unit of signed order flow.
    """
    sv = np.asarray(signed_volume, dtype=float)
    dp = np.asarray(dprice, dtype=float)
    n = sv.size
    if n < 2 or np.std(sv) < 1e-15:
        return 0.0, float(np.mean(dp)) if n else 0.0, 0.0
    Z = np.column_stack([sv, np.ones(n)])
    gram = Z.T @ Z
    coef = np.linalg.solve(gram + 1e-12 * np.eye(2), Z.T @ dp)
    resid = dp - Z @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((dp - dp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return float(coef[0]), float(coef[1]), float(max(r2, 0.0))


@register
class KyleImpactNode(Node):
    """Kyle's (1985) lambda price-impact estimator (microstructure illiquidity).
    Column 0 is signed volume; column 1 is the price LEVEL, from which ΔP is taken
    internally (ΔPₜ = Pₜ − Pₜ₋₁). Regresses ΔP = λ·sv + c; λ is the illiquidity
    coefficient (large λ = price moves a lot per unit flow = illiquid). If only one
    column is given, that column is taken as ΔP directly. Emits an `equation`
    {coef=[λ, intercept], r2}."""

    layer = "equation"
    node_type = "kyle_impact"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        sv = X[:, 0].astype(float)
        if X.shape[1] >= 2:
            price = X[:, 1].astype(float)
            dp = np.empty_like(price)
            dp[0] = 0.0
            dp[1:] = np.diff(price)
            # align ΔPₜ with the signed volume that caused it (drop the seed step)
            sv_fit, dp_fit = sv[1:], dp[1:]
        else:
            sv_fit, dp_fit = sv, sv * 0.0  # degenerate: no price → nothing to impact
        lam, intercept, r2 = fit_kyle_lambda(sv_fit, dp_fit)
        conf = float(np.clip(X.shape[0] / 500.0, 0.2, 0.9)) * float(np.clip(r2, 0.1, 1.0))
        return Belief(
            type="equation",
            payload={
                "coef": [lam, intercept],
                "r2": r2,
                "lambda": lam,
                "intercept": intercept,
            },
            confidence=float(np.clip(conf, 0.0, 1.0)),
        )
