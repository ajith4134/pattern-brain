"""Phase 8 — slice 5: the STACKED-DAG (PLAN.md §12).

Assembles bank nodes into a multi-layer graph that predicts AS ONE NETWORK and
emits a final predictive DISTRIBUTION (scored by slice-1 CRPS). This is "the
spheres connect into a network," made concrete and leakage-safe.

Structure:
* **Layer 0 (base):** forecaster nodes, each turned into a full-length CAUSAL
  one-step prediction column via ``predict(M[:t]) → forecast of x[t]`` (uses only
  the past). These columns are the leakage-safe out-of-fold features.
* **Layer k>0 (meta, optional):** bank nodes whose input matrix is augmented with
  the prior layers' prediction columns — so an upstream model's OUTPUT is a
  downstream model's INPUT (the §12.2 ``belief_features`` connection). Only
  multivariate nodes exploit the extra columns; the real stacking power is the
  combiner below.
* **Combiner (the meta-learner):** turns the base/meta prediction columns into the
  final distribution — ``"mixture"`` (causal skill-weighted mixture of each node's
  point ± expanding-residual spread) or ``"stacked"`` (a causal ridge meta-model
  fit on the prediction columns → point + residual-spread Gaussian; textbook
  stacked generalization).

Every step is causal — ``test_network`` proves it by perturbing the future and
checking earlier predictions are byte-identical.

Domain-agnostic (Rule 23): operates on a generic ``(T, D)``; the target is the
primary series (column 0), which every layer preserves at column 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import create
from .belief_features import belief_to_point
from . import scoring as S


@dataclass
class DAGSpec:
    """A stacked-DAG genome: layers of node_types + a combiner. Serializable so the
    Phase-8 search (slice 6+) can mutate/store it."""
    layers: List[List[str]]
    combiner: str = "mixture"            # "mixture" | "stacked"

    def to_dict(self) -> Dict[str, object]:
        return {"layers": [list(l) for l in self.layers], "combiner": self.combiner}

    def signature(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @property
    def n_nodes(self) -> int:
        return sum(len(l) for l in self.layers)


@dataclass
class DAGForecast:
    index: np.ndarray
    point: np.ndarray
    samples: np.ndarray                  # (n_eval, n_samples)
    realized: np.ndarray
    diag: Dict[str, object] = field(default_factory=dict)


def _causal_pred_column(node_type: str, M: np.ndarray, min_train: int) -> Tuple[np.ndarray, np.ndarray]:
    """A node's full-length CAUSAL one-step prediction column over matrix ``M``
    (forecasts column 0). ``point[t]`` uses only ``M[:t]``; ``spread[t]`` is the
    expanding std of this node's own past residuals (causal)."""
    x = M[:, 0]
    T = len(x)
    point = np.empty(T)
    spread = np.empty(T)
    for t in range(T):
        if t < max(min_train, 3):
            point[t] = x[t - 1] if t > 0 else x[0]
            spread[t] = float(np.std(np.diff(x[:t]))) if t > 2 else 1.0
        else:
            point[t] = belief_to_point(create(node_type).predict(M[:t]))
            res = point[min_train:t] - x[min_train:t]
            sd = float(np.std(res)) if (t - min_train) > 2 else float(np.std(np.diff(x[:t])))
            spread[t] = sd
    return point, np.abs(spread) + 1e-9


class StackedDAG:
    """Executes a :class:`DAGSpec` into a final predictive distribution, all causal."""

    def __init__(self, spec: DAGSpec, min_train: int = 40, n_samples: int = 200,
                 seed: int = 0) -> None:
        self.spec = spec
        self.min_train = min_train
        self.n_samples = n_samples
        self.seed = seed

    def run(self, X) -> DAGForecast:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x = X[:, 0]
        T = len(x)
        M = X.copy()                                    # primary series stays at column 0
        all_points: List[np.ndarray] = []
        all_spreads: List[np.ndarray] = []
        last_layer: List[Tuple[np.ndarray, np.ndarray]] = []
        raw_width = M.shape[1]
        for li, layer in enumerate(self.spec.layers):
            cols, this_layer = [], []
            for nt in layer:
                pt, sp = _causal_pred_column(nt, M, self.min_train)
                cols.append(pt)
                this_layer.append((pt, sp))
                all_points.append(pt)
                all_spreads.append(sp)
            if li < len(self.spec.layers) - 1:          # feed outputs forward to the next layer
                M = np.column_stack([M] + cols)
            last_layer = this_layer
        start = min(self.min_train + 10, T - 1)
        idx = np.arange(start, T)
        if self.spec.combiner == "stacked":
            samples = self._stacked(all_points, x, idx)
        else:
            samples = self._mixture(last_layer, x, idx)
        point = samples.mean(1) if samples.size else np.array([])
        diag = {"augmented_width": int(M.shape[1]), "raw_width": int(raw_width),
                "n_eval": int(idx.size), "n_nodes": self.spec.n_nodes}
        return DAGForecast(idx, point, samples, x[idx], diag)

    # ---- combiners (both causal) ----
    def _mixture(self, last_layer, x, idx) -> np.ndarray:
        pts = np.array([p for p, _ in last_layer])      # (k, T)
        sps = np.array([s for _, s in last_layer])
        M, out = self.n_samples, []
        for t in idx:
            maes = []
            for i in range(len(last_layer)):
                r = np.abs(pts[i, self.min_train:t] - x[self.min_train:t])
                maes.append(float(np.mean(r)) if r.size > 2 else 1.0)
            w = 1.0 / (np.array(maes) + 1e-6)
            w = w / w.sum()
            counts = np.random.default_rng(self.seed * 131 + int(t)).multinomial(M, w)
            samp = [np.random.default_rng(self.seed * 17 + int(t) * 7 + i).normal(
                        pts[i, t], sps[i, t], int(c)) for i, c in enumerate(counts) if c > 0]
            out.append(np.concatenate(samp) if samp else np.full(M, x[t - 1]))
        return np.array(out)

    def _stacked(self, all_points, x, idx) -> np.ndarray:
        from sklearn.linear_model import Ridge
        P = np.array(all_points).T                      # (T, n_pred_cols)
        M, out = self.n_samples, []
        for t in idx:
            Xtr, ytr = P[self.min_train:t], x[self.min_train:t]
            if len(ytr) < 5:
                out.append(np.full(M, x[t - 1])); continue
            m = Ridge(alpha=1.0).fit(Xtr, ytr)
            pred = float(m.predict(P[t:t + 1])[0])
            sd = float(np.std(ytr - m.predict(Xtr))) + 1e-9
            out.append(np.random.default_rng(self.seed * 53 + int(t)).normal(pred, sd, M))
        return np.array(out)

    # ---- scoring (vs persistence baseline) ----
    def score(self, X) -> Dict[str, object]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        fc = self.run(X)
        if fc.index.size == 0:
            return {"n": 0, "mean_crps": float("nan"), "crps_skill": float("nan")}
        x = X[:, 0]
        crps = np.atleast_1d(S.crps_ensemble(fc.samples, fc.realized))
        rng = np.random.default_rng(self.seed + 1)
        base = np.array([rng.normal(x[t - 1], abs(float(np.std(np.diff(x[:t])))) + 1e-9,
                                    self.n_samples) for t in fc.index])
        base_crps = np.atleast_1d(S.crps_ensemble(base, fc.realized))
        cal = S.pit_calibration_error(S.pit_values_ensemble(fc.samples, fc.realized))
        return {
            "n": int(fc.index.size),
            "mean_crps": float(np.mean(crps)),
            "baseline_crps": float(np.mean(base_crps)),
            "crps_skill": S.crps_skill_score(crps, base_crps),
            "calib_ks": float(cal["ks"]),
            "calibrated": bool(cal["calibrated"]),
            "outcome_series": S.crps_outcome_series(crps, base_crps).tolist(),
            "combiner": self.spec.combiner,
            "n_nodes": self.spec.n_nodes,
        }


__all__ = ["DAGSpec", "DAGForecast", "StackedDAG"]
