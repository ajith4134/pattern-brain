"""W2 — Adaptive Compute Manager (the owner's "Intelligence Switch", PLAN §17.1).

Runs many bounded jobs (e.g. the W1 Hyperband rung evaluations) across this box's
cores under a resource governor, instead of one-at-a-time. The owner's instinct
was right (saturate the hardware per batch) but the literature is clear: never run
RAM to 100% — swap collapses throughput. So the governor caps **CPU ~95%** and
**RAM ~80%** (psutil if available), packs work to a worker pool sized to that
budget, and falls back to serial execution if a pool can't start. No GPU on this
box, so the "GPU scheduler" in the design is intentionally inert here.

Domain-agnostic (Rule 23): it maps an arbitrary picklable worker over payloads.
The Stacked-DAG scoring worker (:func:`score_spec_payload`) lives here so it's
importable by the process pool.

Honest scope: this is single-node parallelism (ProcessPoolExecutor), the right
first step on a 12-core box. Distributed execution (Ray/Dask) is a later option,
not needed yet.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import psutil           # optional — governor degrades to cpu_count without it
    _HAS_PSUTIL = True
except Exception:           # pragma: no cover - psutil simply absent
    psutil = None
    _HAS_PSUTIL = False


# --------------------------------------------------- the Stacked-DAG score worker
def score_spec_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Score ONE Stacked-DAG spec at a step/sample budget — module-level so a
    process pool can pickle it. Records nothing (the parent owns the leaderboard);
    returns the score dict + skill + the spec it scored (to map results back)."""
    import numpy as np
    import pattern_brain as _pb          # ensures the bank is registered in the worker
    from pattern_brain.network import StackedDAG, DAGSpec
    from pattern_brain.search import _skill
    _ = _pb
    spec = DAGSpec.from_dict(payload["spec"])
    X = np.asarray(payload["X"], dtype=float)
    mt = int(payload["min_train"])
    ns = int(payload["n_samples"])
    seed = int(payload.get("seed", 0))
    steps = payload.get("steps")
    if steps is not None:
        X = X[-(mt + max(12, int(steps))):]
    score = StackedDAG(spec, mt, ns, seed).score(X)
    sk = _skill(score)
    return {"spec": payload["spec"], "score": score,
            "skill": float(sk) if sk == sk else -9.0}


class ComputeManager:
    """Resource-governed parallel map (PLAN §17.1 W2)."""

    def __init__(self, max_cpu_frac: float = 0.95, max_ram_frac: float = 0.80,
                 max_workers: Optional[int] = None, per_job_mb: int = 350) -> None:
        self.max_cpu_frac = float(max_cpu_frac)
        self.max_ram_frac = float(max_ram_frac)
        self.per_job_mb = int(per_job_mb)
        self._max_workers = max_workers

    # ------------------------------------------------------------- worker budget
    def n_workers(self, n_items: int = 1) -> int:
        """Workers to use: CPU-budget-bounded, then RAM-budget-bounded, then by the
        number of items. Always ≥1."""
        cpu = os.cpu_count() or 2
        w = max(1, int(cpu * self.max_cpu_frac))
        if self._max_workers:
            w = min(w, int(self._max_workers))
        if _HAS_PSUTIL:                                   # don't provoke swapping
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            ram_cap = max(1, int((avail_mb * self.max_ram_frac) / max(1, self.per_job_mb)))
            w = min(w, ram_cap)
        return max(1, min(w, max(1, int(n_items))))

    # ------------------------------------------------------------------- map
    def parallel_map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """Map ``fn`` over ``items`` across a governed process pool. Falls back to
        serial execution for tiny batches or if the pool can't start (so it always
        produces a result). Order is preserved."""
        items = list(items)
        if not items:
            return []
        workers = self.n_workers(len(items))
        if workers <= 1 or len(items) < 2:
            return [fn(it) for it in items]
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(fn, items))
        except Exception:                                 # pool unavailable → serial
            return [fn(it) for it in items]

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, Any]:
        """Live governor view for the dashboard 'compute' tile."""
        out: Dict[str, Any] = {"cores": os.cpu_count(), "workers": self.n_workers(64),
                               "max_cpu_frac": self.max_cpu_frac,
                               "max_ram_frac": self.max_ram_frac, "psutil": _HAS_PSUTIL,
                               "gpu": False}
        if _HAS_PSUTIL:
            out["cpu_percent"] = psutil.cpu_percent(interval=0.0)
            vm = psutil.virtual_memory()
            out["ram_percent"] = vm.percent
            out["ram_available_mb"] = round(vm.available / (1024 * 1024))
        return out


__all__ = ["ComputeManager", "score_spec_payload"]
