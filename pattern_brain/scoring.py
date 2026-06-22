"""Phase 8 — slice 1: proper scoring rules for PREDICTIVE DISTRIBUTIONS
(PLAN.md §12, owner-decided target = a calibrated distribution, not a point).

This is the OBJECTIVE the whole Stacked-DAG search optimizes: given, per period, a
predictive distribution (as an ensemble of samples OR a set of quantiles) and the
realized value, score how good the distribution was — sharpness AND calibration,
which a point error (RMSE) cannot see.

Rules implemented (all standard, all proper):
* **CRPS** (Continuous Ranked Probability Score) — the workhorse distributional
  score; generalizes MAE to distributions, minimized only by the true distribution.
  Sample (energy) form and a quantile (pinball-integral) form.
* **Pinball / quantile loss** — per-quantile proper loss.
* **Brier** + **log-loss** — for a binary probability forecast (e.g., P(up)).
* **PIT** (Probability Integral Transform) + a calibration error — are the predicted
  distributions actually calibrated (PIT ~ Uniform(0,1)), or over/under-confident?
* **CRPS skill score** + a per-period skill **outcome series** that feeds the
  existing :class:`~pattern_brain.evaluator.Evaluator` (so the search inherits the
  purged-walk-forward + Deflated-Sharpe + CSCV/PBO anti-overfit gate).

Domain-agnostic (Rule 23): operates on generic forecasts + realized scalars; nothing
here knows about candles, crypto, or order books. numpy-only.
"""
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


# ----------------------------------------------------------------- CRPS (samples)
def _crps_one(samples: np.ndarray, y: float) -> float:
    """Unbiased CRPS estimator from an ensemble of samples (energy form):
    CRPS = E|X - y| - 0.5 E|X - X'|. The pairwise term uses the sorted identity
    Σ_ij|s_i - s_j| = 2 Σ_k (2k - M - 1) s_(k), so it is O(M log M), not O(M²)."""
    s = np.sort(np.asarray(samples, dtype=float))
    M = s.size
    if M == 0:
        return float("nan")
    if M == 1:
        return float(abs(s[0] - y))
    term1 = float(np.mean(np.abs(s - y)))
    k = np.arange(1, M + 1)
    term2 = float(np.sum((2 * k - M - 1) * s) / (M * M))   # = 0.5 * E|X-X'|
    return term1 - term2


def crps_ensemble(samples: ArrayLike, y: ArrayLike) -> Union[float, np.ndarray]:
    """CRPS for an ensemble forecast. ``samples`` is (M,) with scalar ``y``, or
    (T, M) with ``y`` of shape (T,). Returns a float or a (T,) array."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return _crps_one(samples, float(np.asarray(y).ravel()[0]) if np.ndim(y) else float(y))
    y = np.asarray(y, dtype=float).ravel()
    if y.size != samples.shape[0]:
        raise ValueError(f"y length {y.size} != n_periods {samples.shape[0]}")
    return np.array([_crps_one(samples[t], float(y[t])) for t in range(samples.shape[0])])


# --------------------------------------------------------------- quantile / pinball
def pinball_loss(q: ArrayLike, level: float, y: ArrayLike) -> Union[float, np.ndarray]:
    """Pinball (quantile) loss for predicted quantile ``q`` at probability ``level``
    against realized ``y``. Always non-negative; minimized at the true τ-quantile."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y - q
    out = np.where(d >= 0, level * d, (level - 1.0) * d)
    return float(out) if out.ndim == 0 else out


def crps_from_quantiles(levels: ArrayLike, qvals: ArrayLike,
                        y: ArrayLike) -> Union[float, np.ndarray]:
    """CRPS approximated from a set of predictive quantiles:
    CRPS = 2 ∫_0^1 pinball_τ dτ ≈ 2 · mean_τ pinball_τ over the provided levels.
    ``qvals`` is (L,) with scalar ``y``, or (T, L) with ``y`` of shape (T,)."""
    levels = np.asarray(levels, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    if qvals.ndim == 1:
        yy = float(np.asarray(y).ravel()[0]) if np.ndim(y) else float(y)
        pl = np.array([pinball_loss(qvals[i], levels[i], yy) for i in range(levels.size)])
        return float(2.0 * np.mean(pl))
    y = np.asarray(y, dtype=float).ravel()
    out = np.empty(qvals.shape[0])
    for t in range(qvals.shape[0]):
        pl = np.array([pinball_loss(qvals[t, i], levels[i], y[t]) for i in range(levels.size)])
        out[t] = 2.0 * np.mean(pl)
    return out


# --------------------------------------------------------------- binary prob scores
def brier_score(p: ArrayLike, y: ArrayLike) -> float:
    """Mean Brier score for a probability-of-class-1 forecast vs labels in {0,1}."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: ArrayLike, y: ArrayLike, eps: float = 1e-12) -> float:
    """Mean logarithmic (cross-entropy) loss for a binary probability forecast."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


# ------------------------------------------------------------------ PIT calibration
def pit_values_ensemble(samples: ArrayLike, y: ArrayLike) -> np.ndarray:
    """PIT per observation from an ensemble: fraction of samples ≤ realized.
    A calibrated forecast yields PIT ~ Uniform(0,1)."""
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    return np.array([float(np.mean(samples[t] <= y[t])) for t in range(samples.shape[0])])


def pit_values_quantile(levels: ArrayLike, qvals: ArrayLike, y: ArrayLike) -> np.ndarray:
    """PIT per observation from quantiles: interpolate the predictive CDF at y."""
    levels = np.asarray(levels, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    out = np.empty(qvals.shape[0])
    for t in range(qvals.shape[0]):
        order = np.argsort(qvals[t])
        out[t] = float(np.interp(y[t], qvals[t][order], levels[order], left=0.0, right=1.0))
    return out


def pit_calibration_error(pit: ArrayLike) -> Dict[str, float]:
    """How far the PIT sample is from Uniform(0,1). Returns the Kolmogorov-Smirnov
    statistic vs uniform, the mean absolute ECDF deviation, and a 95% calibration
    flag (KS below the asymptotic 1.36/√n critical value)."""
    pit = np.sort(np.clip(np.asarray(pit, dtype=float), 0.0, 1.0))
    n = pit.size
    if n == 0:
        return {"ks": float("nan"), "mae": float("nan"), "calibrated": False, "n": 0}
    ecdf = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(ecdf - pit)))           # uniform CDF at value p is p
    mae = float(np.mean(np.abs(ecdf - pit)))
    return {"ks": ks, "mae": mae, "calibrated": bool(ks < 1.36 / np.sqrt(n)), "n": int(n)}


# ------------------------------------------------------- skill + Evaluator bridge
def crps_skill_score(model_crps: ArrayLike, baseline_crps: ArrayLike) -> float:
    """1 - mean(model)/mean(baseline). >0 means the model beats the baseline; a
    perfect model →1; equal to baseline →0; worse →negative."""
    m = float(np.mean(model_crps))
    b = float(np.mean(baseline_crps))
    return float(1.0 - m / (b + 1e-12))


def crps_outcome_series(model_crps: ArrayLike,
                        baseline_crps: ArrayLike = None) -> np.ndarray:
    """Turn distributional scores into a per-period, higher-is-better OUTCOME SERIES
    for :meth:`Evaluator.evaluate_outcomes` — so the DAG search inherits the
    purged-walk-forward + Deflated-Sharpe + CSCV/PBO anti-overfit gate. With a
    baseline it is the per-period CRPS *skill* (baseline − model, positive when the
    model wins); without one it is simply −CRPS (lower error → higher outcome)."""
    model_crps = np.asarray(model_crps, dtype=float)
    if baseline_crps is None:
        return -model_crps
    baseline_crps = np.asarray(baseline_crps, dtype=float)
    return baseline_crps - model_crps


def score_distribution(samples: ArrayLike, y: ArrayLike,
                       baseline_crps: ArrayLike = None) -> Dict[str, object]:
    """Convenience: full distributional scorecard for an ensemble forecast sequence —
    mean CRPS, PIT calibration, skill vs baseline, and the Evaluator outcome series."""
    crps = crps_ensemble(samples, y)
    crps_arr = np.atleast_1d(crps)
    pit = pit_values_ensemble(samples, y)
    cal = pit_calibration_error(pit)
    out: Dict[str, object] = {
        "mean_crps": float(np.mean(crps_arr)),
        "calibration": cal,
        "outcome_series": crps_outcome_series(crps_arr, baseline_crps).tolist(),
    }
    if baseline_crps is not None:
        out["crps_skill"] = crps_skill_score(crps_arr, baseline_crps)
    return out


__all__ = [
    "crps_ensemble", "crps_from_quantiles", "pinball_loss",
    "brier_score", "log_loss",
    "pit_values_ensemble", "pit_values_quantile", "pit_calibration_error",
    "crps_skill_score", "crps_outcome_series", "score_distribution",
]
