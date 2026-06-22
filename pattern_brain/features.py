"""Phase 8 — slice 3: the FEATURE FACTORY (PLAN.md §12).

Turns the shared ``(T, D)`` representation (from the encoder adapters) into a
*richer* feature matrix the bank can predict from — the chat's "Feature Factory /
Universal Representation" layer, made concrete and, above all, **causal**.

Two hard rules:
* **Rule 23 (generic-vs-domain split):** only GENERIC, domain-agnostic features
  live here (lags, returns, rolling stats, fractional differencing, momentum,
  EWMA, rolling spectral/vol). Domain features (RSI/MACD/VWAP, funding-rate or
  order-book imbalance) are NOT here — they are injected by ``adapters/`` and
  appended, never baked into the core. ``CORE_FEATURES`` is the explicit generic set.
* **Causality (no look-ahead):** every feature at time ``t`` uses only data at
  ``≤ t``. Rolling stats use expanding-then-rolling past windows; warm-up cells are
  causal-forward-filled (never back-filled). A global standardize (which peeks at
  the whole series) is deliberately absent — ``test_features`` proves causality by
  perturbing the future and checking early features are unchanged.

A content-addressed :class:`FeatureStore` caches results (keyed by the input bytes
+ the factory config) under ``data/feature_store/`` (Rule 1, in-folder) so the
Phase-8 search doesn't recompute features thousands of times.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- causal helpers
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Causal lag by k: out[t] = x[t-k]; the first k cells take x[0] (earliest past)."""
    out = np.empty_like(x)
    if k <= 0:
        return x.copy()
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean, expanding for the first w-1 points (min_periods=1)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w + 1)
    counts = idx - lo + 1
    return (c[idx + 1] - c[lo]) / counts


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    m = _roll_mean(x, w)
    m2 = _roll_mean(x * x, w)
    return np.sqrt(np.maximum(0.0, m2 - m * m))


def _roll_apply(x: np.ndarray, w: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    out = np.empty(len(x))
    for t in range(len(x)):
        out[t] = fn(x[max(0, t - w + 1): t + 1])
    return out


def _frac_diff(x: np.ndarray, d: float, thresh: float = 1e-4) -> np.ndarray:
    """Fixed-width fractional differencing (López de Prado) — stationary WITH memory.
    Causal: out[t] = Σ_k w_k · x[t-k] over a fixed backward window."""
    w = [1.0]
    k = 1
    while True:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh or k > len(x):
            break
        w.append(wk); k += 1
    w = np.array(w[::-1])
    out = np.full(len(x), np.nan)
    width = len(w)
    for t in range(width - 1, len(x)):
        out[t] = float(np.dot(w, x[t - width + 1: t + 1]))
    return out


def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty(len(x)); out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def _causal_fill(col: np.ndarray) -> np.ndarray:
    """Forward-fill non-finite (past→present, no look-ahead); leading non-finite → 0."""
    out = col.astype(float).copy()
    last = 0.0
    seen = False
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]; seen = True
        else:
            out[i] = last if seen else 0.0
    out[~np.isfinite(out)] = 0.0
    return out


# ----------------------------------------------------------------- the feature set
# Each generic feature: (name, fn(x_1d) -> column). x is the PRIMARY series (col 0).
def _feature_builders() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "lag1": lambda x: _shift(x, 1),
        "lag2": lambda x: _shift(x, 2),
        "lag3": lambda x: _shift(x, 3),
        "lag5": lambda x: _shift(x, 5),
        "return": lambda x: x - _shift(x, 1),
        "log_return": lambda x: np.log(np.abs(x) + 1e-9) - np.log(np.abs(_shift(x, 1)) + 1e-9),
        "roll_mean_5": lambda x: _roll_mean(x, 5),
        "roll_mean_20": lambda x: _roll_mean(x, 20),
        "roll_std_10": lambda x: _roll_std(x, 10),
        "roll_zscore_20": lambda x: (x - _roll_mean(x, 20)) / (_roll_std(x, 20) + 1e-9),
        "momentum_10": lambda x: x - _shift(x, 10),
        "ewma_03": lambda x: _ewma(x, 0.3),
        "fracdiff_04": lambda x: _frac_diff(x, 0.4),
        "roll_vol_10": lambda x: _roll_std(x - _shift(x, 1), 10),
    }


CORE_FEATURES: List[str] = list(_feature_builders().keys())
# Domain features (RSI/MACD/VWAP/funding/orderbook-imbalance) live in adapters/, not here.


class FeatureFactory:
    """Builds a causal feature matrix from a shared ``(T, D)``: passes the raw D
    columns through and appends the selected GENERIC features computed on the
    primary series (column 0). Output is a finite ``(T, D + n_features)`` that
    satisfies the core ``Node`` contract."""

    def __init__(self, features: Optional[Sequence[str]] = None,
                 keep_raw: bool = True) -> None:
        builders = _feature_builders()
        self.features = list(features) if features is not None else CORE_FEATURES
        unknown = [f for f in self.features if f not in builders]
        if unknown:
            raise ValueError(f"unknown features {unknown}; known: {sorted(builders)}")
        self._builders = builders
        self.keep_raw = keep_raw

    def signature(self) -> str:
        return json.dumps({"features": self.features, "keep_raw": self.keep_raw},
                          sort_keys=True)

    def transform(self, X) -> Tuple[np.ndarray, List[str]]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x0 = X[:, 0]
        cols: List[np.ndarray] = []
        names: List[str] = []
        if self.keep_raw:
            for j in range(X.shape[1]):
                cols.append(_causal_fill(X[:, j])); names.append(f"raw{j}")
        for f in self.features:
            cols.append(_causal_fill(self._builders[f](x0))); names.append(f)
        out = np.column_stack(cols)
        return out, names

    @property
    def feature_names(self) -> List[str]:
        base = [f"raw{j}" for j in range(0)] if self.keep_raw else []
        return base + list(self.features)


# ----------------------------------------------------------------- feature store
class FeatureStore:
    """Content-addressed cache of feature matrices (keyed by input bytes + factory
    config) under ``data/feature_store/`` — so the search recomputes nothing.
    Rule 1: every path is asserted inside the project folder."""

    def __init__(self, root: Optional[str] = None) -> None:
        root = root or os.path.join(_PROJECT_ROOT, "data", "feature_store")
        root = os.path.abspath(root)
        if not (root == _PROJECT_ROOT or root.startswith(_PROJECT_ROOT + os.sep)):
            raise ValueError(f"Rule 1: feature store must be inside {_PROJECT_ROOT}")
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def key(X: np.ndarray, factory: FeatureFactory) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X, dtype=float).tobytes())
        h.update(factory.signature().encode())
        return h.hexdigest()[:16]

    def _path(self, key: str) -> str:
        p = os.path.abspath(os.path.join(self.root, f"{key}.npz"))
        if not p.startswith(self.root + os.sep):
            raise ValueError("Rule 1: refusing path traversal outside the feature store")
        return p

    def get(self, key: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        return z["matrix"], [str(n) for n in z["names"]]

    def put(self, key: str, matrix: np.ndarray, names: List[str]) -> None:
        np.savez(self._path(key), matrix=matrix, names=np.array(names, dtype=object).astype(str))

    def get_or_compute(self, X, factory: FeatureFactory) -> Tuple[np.ndarray, List[str]]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        k = self.key(X, factory)
        hit = self.get(k)
        if hit is not None:
            return hit
        matrix, names = factory.transform(X)
        self.put(k, matrix, names)
        return matrix, names


__all__ = ["FeatureFactory", "FeatureStore", "CORE_FEATURES"]
