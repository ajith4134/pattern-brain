"""Phase 8 — slice 9: the CAPSTONE acceptance test (PLAN.md §12.1).

The whole-project verification, made rigorous: SEARCH/SELECT the best Stacked-DAG on
HISTORY only → FREEZE it → FORWARD-TEST the frozen network on a truly unseen holdout
→ score the predictive DISTRIBUTION (CRPS / PIT calibration / directional hit-rate)
vs a persistence baseline, deflated (DSR) by the search budget. ONLY the frozen
forward-test is the project's reported accuracy — the search's own score never is.

This module is DOMAIN-AGNOSTIC (Rule 23): it takes a generic ``(T, D)`` whose column
0 is the prediction target (e.g. next-period return). The crypto-specific glue (ccxt
fetch + a single-symbol wrapper) lives in ``adapters/crypto.py``.

A FAILURE is a valid, honest outcome — markets may be unpredictable at a horizon;
the test's job is to tell the truth, not manufacture a win.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .registry import create
from .belief_features import belief_to_point, belief_to_samples
from .network import DAGSpec, StackedDAG
from .leaderboard import Leaderboard
from .search import DAGSearch
from .evaluator import Evaluator
from .scoring import (crps_ensemble, crps_skill_score, pit_values_ensemble,
                      pit_calibration_error, crps_outcome_series)


def _next_move(spec: DAGSpec, x0: np.ndarray, X: np.ndarray, n_samples: int, seed: int) -> Dict:
    """The frozen network's forecast of the ACTUAL next step (t = T): a mixture of
    the base layer's one-step predictive distributions from the full history."""
    rng = np.random.default_rng(seed + 99)
    bases = spec.layers[0]
    per = max(8, n_samples // max(1, len(bases)))
    samp = []
    for nt in bases:
        b = create(nt).predict(X)
        samp.append(belief_to_samples(b, x0, per, rng))
    s = np.concatenate(samp)
    return {"mean": float(np.mean(s)), "q10": float(np.quantile(s, 0.1)),
            "q50": float(np.quantile(s, 0.5)), "q90": float(np.quantile(s, 0.9)),
            "prob_up": float(np.mean(s > 0))}


def run_capstone(matrix, *, holdout_frac: float = 0.3, strategy: str = "evolutionary",
                 budget: int = 24, base_pool: Optional[List[str]] = None,
                 min_train: int = 60, n_samples: int = 150, seed: int = 0,
                 llm_chat=None, meta: Optional[Dict] = None) -> Dict:
    X = np.asarray(matrix, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = len(X)
    if T < 80:
        raise ValueError(f"need >= 80 rows for a freeze/forward-test, got {T}")
    split = int(round(T * (1.0 - holdout_frac)))
    Xhist = X[:split]                                   # the search NEVER sees the holdout

    # 1+2. SEARCH/SELECT on history only --------------------------------------
    lb = Leaderboard(":memory:")
    s = DAGSearch(Xhist, leaderboard=lb, base_pool=base_pool, min_train=min_train,
                  n_samples=n_samples, dataset_id="capstone", seed=seed)
    if llm_chat is not None:
        from .agent.planner import DAGPlanner
        DAGPlanner(s, llm_chat=llm_chat).plan_and_search(rounds=max(2, budget // 4), n_per_round=4)
    elif strategy == "random":
        s.random_search(budget)
    else:
        pop = max(4, budget // 4)
        s.evolutionary_search(generations=max(2, budget // pop), pop=pop)
    best_row = s.best()
    if best_row is None:
        raise RuntimeError("search produced no DAG")
    best_spec = DAGSpec.from_dict(json.loads(best_row["spec_json"]))

    # 3. FREEZE + 4. FORWARD-TEST on the unseen holdout -----------------------
    fc = StackedDAG(best_spec, min_train, n_samples, seed).run(X)   # causal one-step over all T
    mask = fc.index >= split
    f_idx, f_samp, f_real = fc.index[mask], fc.samples[mask], fc.realized[mask]
    if len(f_idx) < 5:
        raise RuntimeError("holdout too small after warm-up; lower min_train or holdout_frac")
    x0 = X[:, 0]
    crps = np.atleast_1d(crps_ensemble(f_samp, f_real))
    rng = np.random.default_rng(seed + 7)
    base = np.array([rng.normal(x0[t - 1], abs(float(np.std(np.diff(x0[:t])))) + 1e-9, n_samples)
                     for t in f_idx])
    base_crps = np.atleast_1d(crps_ensemble(base, f_real))
    fskill = crps_skill_score(crps, base_crps)
    cal = pit_calibration_error(pit_values_ensemble(f_samp, f_real))

    # 5. directional hit-rate (did it call the move?) + DSR over the holdout ---
    point = f_samp.mean(1)
    net_dir = float(np.mean(np.sign(point) == np.sign(f_real)))
    base_dir = float(np.mean(np.sign(x0[f_idx - 1]) == np.sign(f_real)))
    dsr = None
    if len(crps) >= 8:
        dsr = float(Evaluator(n_splits=4, embargo=0.02).evaluate_outcomes(
            crps_outcome_series(crps, base_crps).tolist(), n_trials=max(1, budget)).dsr)

    # 6. the actual NEXT future move (frozen net) -----------------------------
    nm = _next_move(best_spec, x0, X, n_samples, seed)
    last_close = (meta or {}).get("last_close")
    is_return_target = (meta or {}).get("target", "return") == "return"
    next_price = (float(last_close) * float(np.exp(nm["mean"]))) if (last_close and is_return_target) else None

    verdict = ("PASS" if (fskill > 0 and cal["calibrated"] and dsr is not None and dsr >= 0.95)
               else "WEAK" if fskill > 0 else "FAIL")
    return {
        **(meta or {}),
        "n_total": T, "n_history": split, "n_holdout": int(len(f_idx)),
        "n_nets_searched": lb.count(), "history_skill": float(best_row["crps_skill"]),
        "best_spec": best_spec.to_dict(),
        "forward_crps": float(np.mean(crps)), "forward_baseline_crps": float(np.mean(base_crps)),
        "forward_skill": float(fskill), "forward_calibrated": bool(cal["calibrated"]),
        "forward_calib_ks": float(cal["ks"]), "forward_dsr": dsr,
        "net_directional_hit": net_dir, "baseline_directional_hit": base_dir,
        "next_move": nm, "next_price_est": next_price, "verdict": verdict,
    }


__all__ = ["run_capstone"]
