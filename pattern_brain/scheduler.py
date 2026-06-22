"""W1 — budget-aware smart search: Successive Halving + Hyperband (PLAN §17.1).

The owner's own analysis nailed the core problem: brute-forcing every pathway is
hopeless ("9 billion × 5s ≈ 1,426 years"); the win is *not testing bad pathways*.
This is the mechanism that does it (Li et al., Hyperband, arXiv 1603.06560):

  * **Successive Halving** — give EVERY candidate a tiny budget (a short data
    window + few MC samples), keep the top ``1/eta``, PROMOTE survivors to a
    larger budget, repeat. Full budget is spent only on the ~survivors.
  * **Hyperband** — run several Successive-Halving brackets that trade off "many
    cheap candidates" vs "few well-evaluated ones", so you don't have to guess the
    right budget ladder up front.

Budget here = ``(data_fraction, n_samples)`` threaded into
``DAGSearch.score_spec_budgeted`` — both control scoring cost. W3 (``genome.py``)
is used to drop terminal nodes from forecaster candidate pools before any compute.

ASHA (the async, all-cores-busy variant) is a follow-on — it needs the W2 compute
manager's worker pool; on this single CPU box synchronous SH already captures the
"don't waste compute on bad pathways" win.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from . import genome
from .search import DAGSearch, _skill


def _safe_skill(score: Dict[str, Any]) -> float:
    s = _skill(score)
    return s if s == s else -9.0           # NaN → very low so it loses


class SuccessiveHalving:
    """One Successive-Halving bracket over a :class:`DAGSearch`."""

    def __init__(self, search: DAGSearch, eta: int = 3,
                 rungs: Optional[List[Tuple[float, int]]] = None, seed: int = 0) -> None:
        self.s = search
        self.eta = max(2, int(eta))
        # (data_fraction, n_samples) ladder — cheap → expensive
        full = max(40, int(self.s.n_samples))
        self.rungs = rungs or [(0.4, 16), (0.7, 40), (1.0, full)]
        self.rng = random.Random(seed)

    def _sample(self, n: int) -> List:
        """Sample n distinct valid specs; W3-prune terminal nodes out of the
        forecaster base pool (a terminal/decision node can't be a base model)."""
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < n * 25:
            tries += 1
            spec = self.s._random_spec()
            bases = [b for b in spec.layers[0] if genome.role(b) != "terminal"]
            if not bases:
                bases = [self.rng.choice([p for p in self.s.pool
                                          if genome.role(p) != "terminal"] or self.s.pool)]
            spec.layers[0] = bases
            sig = spec.signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(spec)
        return out

    def run(self, n_configs: int = 27) -> Dict[str, Any]:
        cands = self._sample(max(self.eta, n_configs))
        history: List[Dict[str, Any]] = []
        for ri, (frac, ns) in enumerate(self.rungs):
            last = ri == len(self.rungs) - 1
            scored = [(self.s.score_spec_budgeted(spec, frac=frac, n_samples=ns, record=last),
                       spec) for spec in cands]
            scored.sort(key=lambda z: _safe_skill(z[0]), reverse=True)
            if last:
                kept = len(scored)
            else:
                kept = max(1, math.ceil(len(scored) / self.eta))
            history.append({"rung": ri, "frac": frac, "n_samples": ns,
                            "evaluated": len(scored), "kept": kept,
                            "best_skill": round(_safe_skill(scored[0][0]), 4) if scored else None})
            cands = [spec for _, spec in scored[:kept]]
        return {"best_spec": cands[0].to_dict() if cands else None,
                "rungs": history,
                "total_evaluations": sum(h["evaluated"] for h in history)}


class Hyperband:
    """A few Successive-Halving brackets of decreasing width (PLAN §17.1 W1)."""

    def __init__(self, search: DAGSearch, eta: int = 3, seed: int = 0) -> None:
        self.s = search
        self.eta = max(2, int(eta))
        self.seed = seed

    def run(self, max_configs: int = 27, n_brackets: int = 2) -> Dict[str, Any]:
        brackets: List[Dict[str, Any]] = []
        n = max(self.eta, max_configs)
        for b in range(max(1, n_brackets)):
            sh = SuccessiveHalving(self.s, eta=self.eta, seed=self.seed + b)
            brackets.append(sh.run(n_configs=n))
            n = max(self.eta, n // self.eta)        # next bracket: fewer, deeper candidates
        return {"best": self.s.best(),
                "brackets": brackets,
                "total_evaluations": sum(br["total_evaluations"] for br in brackets)}


__all__ = ["SuccessiveHalving", "Hyperband"]
