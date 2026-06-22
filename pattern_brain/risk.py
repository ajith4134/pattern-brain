"""Post-Phase-8 — the RISK / VOLATILITY layer (PLAN.md §14, owner idea 1).

The capstone proved the network's edge is DISTRIBUTIONAL (it forecasts the next
return's *distribution* — its spread/volatility — robustly across symbols and
timeframes, DSR-significant), while DIRECTION is at chance. So the product that
plays to a real, replicated edge is **risk / volatility forecasting**, not
direction-betting. This module turns a predictive distribution (an ensemble of
samples) into the signals a risk system actually uses:

* **predicted volatility** (the distribution's spread) — the quantity the net
  forecasts well;
* **Value-at-Risk** and **Expected Shortfall** (CVaR) at a confidence level;
* a **vol-targeted position size** (size down when forecast vol is high) — the
  classic way to convert a vol forecast into a sizing decision;
* a **vol-forecast skill** evaluator: does the forecast vol track realized vol
  better than a naive rolling-historical-vol baseline? (Correlation + QLIKE.)

Domain-agnostic (Rule 23): pure functions on a generic return-distribution ensemble.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def risk_signals(samples: Sequence[float], *, target_vol: float = 0.01,
                 var_level: float = 0.05, max_leverage: float = 3.0,
                 capital: float = 1.0) -> Dict[str, float]:
    """Risk signals from an ensemble of next-period RETURN samples.

    - ``pred_vol``  : forecast volatility = std of the predictive distribution.
    - ``var``       : Value-at-Risk at ``var_level`` (a positive loss magnitude).
    - ``es``        : Expected Shortfall / CVaR (mean loss in the worst ``var_level`` tail).
    - ``position_fraction`` : vol-targeted sizing = clip(target_vol / pred_vol) — smaller
      when forecast vol is high (the core of risk-parity / vol-targeting).
    - ``capital_at_risk``   : ``capital * var``.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"pred_vol": 0.0, "mean": 0.0, "var": 0.0, "es": 0.0, "prob_up": 0.5,
                "position_fraction": 0.0, "capital_at_risk": 0.0}
    pred_vol = float(np.std(s))
    mean = float(np.mean(s))
    q = float(np.quantile(s, var_level))                # left-tail return (usually < 0)
    var = float(max(0.0, -q))                           # loss magnitude
    tail = s[s <= q]
    es = float(max(0.0, -np.mean(tail))) if tail.size else var
    pos = float(np.clip(target_vol / (pred_vol + 1e-12), 0.0, max_leverage))
    return {"pred_vol": pred_vol, "mean": mean, "var": var, "es": es,
            "prob_up": float(np.mean(s > 0)), "position_fraction": pos,
            "capital_at_risk": float(capital) * var}


def realized_vol(returns: Sequence[float], window: int = 12) -> np.ndarray:
    """Causal rolling realized volatility (std of returns over a trailing window)."""
    r = np.asarray(returns, dtype=float)
    out = np.empty(len(r))
    for t in range(len(r)):
        seg = r[max(0, t - window + 1): t + 1]
        out[t] = float(np.std(seg)) if len(seg) > 1 else abs(float(r[t]))
    return out


def evaluate_vol_forecast(pred_vol: Sequence[float], realized: Sequence[float],
                          baseline_vol: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Does the forecast volatility track the realized |move| better than a naive
    baseline? Returns Pearson correlation with |realized|, QLIKE loss (a proper
    volatility loss), and a skill score vs the baseline (lower QLIKE = better)."""
    pv = np.asarray(pred_vol, dtype=float)
    rl = np.abs(np.asarray(realized, dtype=float))
    n = min(len(pv), len(rl))
    pv, rl = pv[:n], rl[:n]
    corr = float(np.corrcoef(pv, rl)[0, 1]) if n > 2 and np.std(pv) > 0 and np.std(rl) > 0 else 0.0
    # QLIKE: log(sigma^2) + realized^2 / sigma^2  (robust volatility loss)
    var_f = pv ** 2 + 1e-12
    qlike = float(np.mean(np.log(var_f) + (rl ** 2) / var_f))
    out = {"n": int(n), "corr_with_realized": corr, "qlike": qlike}
    if baseline_vol is not None:
        bv = np.asarray(baseline_vol, dtype=float)[:n]
        var_b = bv ** 2 + 1e-12
        qlike_b = float(np.mean(np.log(var_b) + (rl ** 2) / var_b))
        out["baseline_qlike"] = qlike_b
        out["qlike_skill"] = float(qlike_b - qlike)     # >0 ⇒ forecast beats baseline
    return out


__all__ = ["risk_signals", "realized_vol", "evaluate_vol_forecast"]
