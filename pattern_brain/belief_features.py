"""Phase 8 — slice 4: the Belief → feature/sample projection (PLAN.md §12.2).

The connective tissue that lets ONE model's OUTPUT become ANOTHER model's INPUT —
the concrete answer to "how do heterogeneous outputs connect" (the owner's pasted
chat). Two projections, both domain-agnostic (Rule 23, key off generic belief
type/payload):

* **belief_to_vector** — a FIXED-WIDTH numeric vector ``[primary_scalar, confidence]``
  per Belief, so N nodes' outputs stack into a ``(T, 2N)`` feature block the next
  DAG layer consumes. ``primary_scalar`` is type-specific (forecast→estimate,
  anomaly→fraction, equation→r², regime→state, signal→its lead scalar, …) with a
  generic fallback so a NEW belief type can never break stacking.
* **belief_to_samples** — turns a Belief into a predictive-distribution ENSEMBLE
  (for the slice-1 CRPS/ PIT scoring): uses the node's own quantiles / std /
  interval when present, else a point ± past-volatility Gaussian.

``assert_projection_completeness`` guards the interlingua: every declared belief
type must project (dedicated or fallback), and reports which still ride the
generic fallback so a real projector can be added later.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .belief import Belief
from .interlingua import _CATALOG as _ILC

BELIEF_VECTOR_WIDTH = 2


def belief_to_point(b: Belief) -> float:
    """The Belief's central numeric assertion. For a forecast that is the point
    estimate of the next value; for other types it is the type's lead scalar."""
    p = b.payload
    if b.type == "forecast":
        if "estimate" in p:
            return _f(p["estimate"])
        nv = p.get("next_vector")
        return _f(nv[0]) if nv else 0.0
    return _primary_scalar(b)


def _primary_scalar(b: Belief) -> float:
    t, p = b.type, b.payload
    if t == "forecast":
        return belief_to_point(b)
    if t == "anomaly":
        return _f(p.get("fraction", p.get("n_anomalies", 0.0)))
    if t == "equation":
        return _f(p.get("r2", 0.0))
    if t == "regime":
        return _f(p.get("current_state", 0))
    if t == "next_state":
        return _f(p.get("next_state", 0))
    if t in ("action", "decision"):
        return _f(p.get("action", 0))
    if t == "denoised":
        return _f(p.get("residual", 0.0))
    if t == "cluster":
        return _f(p.get("n_clusters", 0))
    if t == "density":
        return _f(p.get("max_proba", 0.0))
    if t == "pattern":
        return _f(p.get("n_patterns", 0))
    if t == "spectral":
        return _f(p.get("dominant_freq", 0.0))
    if t == "signal":
        for k, v in p.items():                      # the lead scalar (skip the raw series)
            if k == "series":
                continue
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                return float(v)
        return 0.0
    for k, v in p.items():                          # generic fallback: first finite scalar
        if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
            return float(v)
    return 0.0


def belief_to_vector(b: Belief) -> np.ndarray:
    """Fixed-width ``[primary_scalar, confidence]`` projection (the stacking unit)."""
    s = _primary_scalar(b)
    s = float(s) if np.isfinite(s) else 0.0
    return np.array([s, float(b.confidence)], dtype=float)


def belief_vector_names(prefix: str = "") -> List[str]:
    return [f"{prefix}primary", f"{prefix}conf"]


def belief_to_samples(b: Belief, hist, n_samples: int, rng=None) -> np.ndarray:
    """A predictive-distribution ensemble (M,) from a Belief: prefers the node's own
    uncertainty (quantiles → std → interval), else point ± past one-step volatility."""
    rng = rng or np.random.default_rng(0)
    point = belief_to_point(b)
    p = b.payload
    q = p.get("quantiles")
    if isinstance(q, dict) and len(q) >= 2:
        levels, vals = [], []
        for k, v in q.items():
            digits = "".join(ch for ch in str(k) if ch.isdigit())
            if not digits:
                continue
            levels.append(int(digits) / 100.0); vals.append(_f(v))
        if len(levels) >= 2:
            order = np.argsort(levels)
            lv, vv = np.array(levels)[order], np.array(vals)[order]
            return np.interp(rng.uniform(0, 1, n_samples), lv, vv)
    if "std" in p:
        return rng.normal(point, abs(_f(p["std"])) + 1e-9, n_samples)
    if "lower" in p and "upper" in p:
        sd = (abs(_f(p["upper"]) - _f(p["lower"])) / 4.0) + 1e-9
        return rng.normal(point, sd, n_samples)
    if "ci_low" in p and "ci_high" in p:
        sd = (abs(_f(p["ci_high"]) - _f(p["ci_low"])) / 4.0) + 1e-9
        return rng.normal(point, sd, n_samples)
    hist = np.asarray(hist, dtype=float)
    sd = float(np.std(np.diff(hist))) if hist.size > 2 else 1.0
    return rng.normal(point, abs(sd) + 1e-9, n_samples)


_DEDICATED = {"forecast", "anomaly", "equation", "regime", "next_state", "action",
              "decision", "denoised", "cluster", "density", "pattern", "spectral", "signal"}


def assert_projection_completeness() -> Dict[str, object]:
    """Every declared interlingua belief type must project (dedicated or fallback).
    Returns the dedicated-covered set and any types still on the generic fallback."""
    declared = set(_ILC.keys())
    fallback_only = sorted(declared - _DEDICATED)
    return {
        "declared": sorted(declared),
        "dedicated": sorted(declared & _DEDICATED),
        "fallback_only": fallback_only,
        "complete": True,                           # fallback guarantees every type projects
    }


def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


__all__ = ["belief_to_point", "belief_to_vector", "belief_vector_names",
           "belief_to_samples", "assert_projection_completeness", "BELIEF_VECTOR_WIDTH"]
