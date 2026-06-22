"""Phase 8 — slice 4: the OUT-OF-FOLD (OOF) prediction harness (PLAN.md §12).

Produces every node's LEAKAGE-SAFE one-step predictions and scores them as
predictive distributions (slice-1 CRPS/PIT). Protocol = anchored walk-forward:
at each evaluation time ``t`` the node sees ONLY ``X[:t]`` and forecasts ``x[t]``
(the value right after its window) — causal by construction (predicting t uses no
data ≥ t; the one-step gap is the natural embargo). This is the per-node forward
evaluation; the search-selection layer (slice 6+) additionally re-scores through
the Evaluator's purged-WF + DSR/CSCV gate.

Each forecast Belief is turned into a predictive-distribution ensemble via
``belief_features.belief_to_samples`` (the node's own quantiles/std/interval, else
point ± past volatility), then scored by CRPS against a persistence baseline.
Results cache content-addressed under ``data/oof_cache/`` (Rule 1, in-folder).

Domain-agnostic (Rule 23): operates on a generic ``(T, D)``; forecasts the primary
series (column 0).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .registry import create
from .belief_features import belief_to_point, belief_to_samples
from . import scoring as S

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class OOFResult:
    node_type: str
    index: np.ndarray          # the evaluated time indices
    point: np.ndarray          # point forecasts
    realized: np.ndarray       # realized x[t]
    samples: np.ndarray        # (n_eval, n_samples) predictive ensembles


class OOFHarness:
    """Anchored walk-forward one-step OOF predictor + distributional scorer."""

    def __init__(self, min_train: int = 40, stride: int = 1, n_samples: int = 200,
                 seed: int = 0, cache_dir: Optional[str] = None) -> None:
        self.min_train = min_train
        self.stride = stride
        self.n_samples = n_samples
        self.seed = seed
        if cache_dir is not None:
            cache_dir = os.path.abspath(cache_dir)
            if not (cache_dir == _PROJECT_ROOT or cache_dir.startswith(_PROJECT_ROOT + os.sep)):
                raise ValueError(f"Rule 1: oof cache must be inside {_PROJECT_ROOT}")
            os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ core
    def node_oof(self, node_type: str, X) -> OOFResult:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cached = self._cache_get(node_type, X)
        if cached is not None:
            return cached
        x = X[:, 0]
        T = len(x)
        rng = np.random.default_rng(self.seed)
        idx, pts, real, samp = [], [], [], []
        for t in range(self.min_train, T, self.stride):
            b = create(node_type).predict(X[:t])            # forecast of x[t], causal
            idx.append(t)
            pts.append(belief_to_point(b))
            real.append(x[t])
            samp.append(belief_to_samples(b, x[:t], self.n_samples, rng))
        res = OOFResult(node_type, np.array(idx), np.array(pts, float),
                        np.array(real, float),
                        np.array(samp, float) if samp else np.zeros((0, self.n_samples)))
        self._cache_put(node_type, X, res)
        return res

    def _baseline_samples(self, x: np.ndarray, index: np.ndarray, rng) -> np.ndarray:
        """Persistence baseline: predict x[t-1] with spread = past one-step vol."""
        out = []
        for t in index:
            sd = float(np.std(np.diff(x[:t]))) if t > 2 else 1.0
            out.append(rng.normal(x[t - 1], abs(sd) + 1e-9, self.n_samples))
        return np.array(out) if len(out) else np.zeros((0, self.n_samples))

    def score_node(self, node_type: str, X) -> Dict[str, object]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        r = self.node_oof(node_type, X)
        if r.index.size == 0:
            return {"node_type": node_type, "n": 0, "mean_crps": float("nan"),
                    "crps_skill": float("nan"), "calib_ks": float("nan"),
                    "point_mae": float("nan")}
        crps = np.atleast_1d(S.crps_ensemble(r.samples, r.realized))
        rng = np.random.default_rng(self.seed + 1)
        base = self._baseline_samples(X[:, 0], r.index, rng)
        base_crps = np.atleast_1d(S.crps_ensemble(base, r.realized))
        pit = S.pit_values_ensemble(r.samples, r.realized)
        cal = S.pit_calibration_error(pit)
        return {
            "node_type": node_type,
            "n": int(r.index.size),
            "mean_crps": float(np.mean(crps)),
            "baseline_crps": float(np.mean(base_crps)),
            "crps_skill": S.crps_skill_score(crps, base_crps),
            "calib_ks": float(cal["ks"]),
            "calibrated": bool(cal["calibrated"]),
            "point_mae": float(np.mean(np.abs(r.point - r.realized))),
        }

    def sweep(self, X, node_types: List[str]) -> List[Dict[str, object]]:
        """Score many nodes; return the per-node table sorted by CRPS skill (best
        first) — the 'which of the nodes actually predict' leaderboard precursor."""
        rows = [self.score_node(nt, X) for nt in node_types]
        rows.sort(key=lambda d: (d["crps_skill"] if np.isfinite(d["crps_skill"]) else -1e9),
                  reverse=True)
        return rows

    # ----------------------------------------------------------------- cache
    def _key(self, node_type: str, X: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(node_type.encode())
        h.update(f"{self.min_train}|{self.stride}|{self.n_samples}|{self.seed}".encode())
        h.update(np.ascontiguousarray(X, dtype=float).tobytes())
        return h.hexdigest()[:16]

    def _cache_get(self, node_type, X) -> Optional[OOFResult]:
        if not self.cache_dir:
            return None
        p = os.path.join(self.cache_dir, f"{self._key(node_type, X)}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        return OOFResult(node_type, z["index"], z["point"], z["realized"], z["samples"])

    def _cache_put(self, node_type, X, res: OOFResult) -> None:
        if not self.cache_dir:
            return
        p = os.path.join(self.cache_dir, f"{self._key(node_type, X)}.npz")
        np.savez(p, index=res.index, point=res.point, realized=res.realized, samples=res.samples)


def forecast_node_types() -> List[str]:
    """Registered nodes that emit a 'forecast' belief — the ones the OOF harness
    scores as one-step forecasters (others contribute as features via
    belief_features, not as direct predictors)."""
    from .registry import all_node_types
    out = []
    x = np.cumsum(np.random.default_rng(0).normal(size=80)).reshape(-1, 1)
    for nt in all_node_types():
        try:
            if create(nt).predict(x).type == "forecast":
                out.append(nt)
        except Exception:
            continue
    return out


__all__ = ["OOFHarness", "OOFResult", "forecast_node_types"]
