"""Phase 8 — slice 7: the SEARCH engine over Stacked-DAGs (PLAN.md §12).

Makes "try different network combinations and see which scores highest" real and
*honest*. Three strategies generate `DAGSpec` genomes, each is scored by
`StackedDAG.score` (slice-1 CRPS), the per-period skill series is run through the
`Evaluator` for a **Deflated Sharpe** that is penalized by the SEARCH BUDGET (the
number of configurations tried — correction B), and every trial is written to the
`Leaderboard`:

* **random** — uniform sampling of the spec space (the simple, unbiased start).
* **evolutionary** — elitism + mutate/crossover of the leaderboard's best.
* **optuna** — Bayesian/TPE sampling (optional dep; degrades out cleanly if absent).

The anti-overfit guard is structural: DSR is deflated by ``n_trials = budget``, so
mining more combinations RAISES the bar each must clear — a high score from trying
thousands of nets does not win unless it survives selection-bias deflation.

Domain-agnostic (Rule 23): searches over generic node_types on a generic ``(T, D)``.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .network import DAGSpec, StackedDAG
from .leaderboard import Leaderboard
from .evaluator import Evaluator

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:                                       # pragma: no cover - optional dep
    _HAS_OPTUNA = False

DEFAULT_POOL = ["drift_forecast", "theta_forecast", "naive_mean_forecast",
                "holt_linear_forecast", "svr_forecast", "rf_forecast"]
COMBINERS = ["mixture", "stacked", "gated"]   # "gated" = Phase-9 MoE regime-gate


def optuna_available() -> bool:
    return _HAS_OPTUNA


def _skill(score: Dict[str, object]) -> float:
    s = score.get("crps_skill")
    return float(s) if (s is not None and s == s) else -1e9


class DAGSearch:
    """Searches the Stacked-DAG space; records every trial to a Leaderboard."""

    def __init__(self, X, leaderboard: Optional[Leaderboard] = None,
                 base_pool: Optional[Sequence[str]] = None, max_base: int = 4,
                 min_train: int = 40, n_samples: int = 120,
                 evaluator: Optional[Evaluator] = None, dataset_id: str = "search",
                 seed: int = 0) -> None:
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.lb = leaderboard or Leaderboard(":memory:")
        self.pool = list(base_pool or DEFAULT_POOL)
        self.max_base = max(1, min(max_base, len(self.pool)))
        self.min_train = min_train
        self.n_samples = n_samples
        self.ev = evaluator or Evaluator(n_splits=4, embargo=0.02)
        self.dataset_id = dataset_id
        self.seed = seed
        self.rng = random.Random(seed)
        self._budget = 1
        self._seen: set = set()

    # --------------------------------------------------------- genome operators
    def _random_spec(self) -> DAGSpec:
        k = self.rng.randint(1, self.max_base)
        bases = self.rng.sample(self.pool, k)
        return DAGSpec([bases], self.rng.choice(COMBINERS))

    def _mutate(self, spec: DAGSpec) -> DAGSpec:
        bases = list(spec.layers[0])
        combiner = spec.combiner
        op = self.rng.random()
        avail = [n for n in self.pool if n not in bases]
        if op < 0.3 and len(bases) < self.max_base and avail:
            bases.append(self.rng.choice(avail))
        elif op < 0.6 and len(bases) > 1:
            bases.pop(self.rng.randrange(len(bases)))
        elif op < 0.85 and avail and bases:
            bases[self.rng.randrange(len(bases))] = self.rng.choice(avail)
        else:
            combiner = "stacked" if combiner == "mixture" else "mixture"
        bases = sorted(set(bases)) or [self.rng.choice(self.pool)]
        return DAGSpec([bases], combiner)

    def _crossover(self, a: DAGSpec, b: DAGSpec) -> DAGSpec:
        union = list(set(a.layers[0]) | set(b.layers[0]))
        k = self.rng.randint(1, min(self.max_base, len(union)))
        bases = self.rng.sample(union, min(k, len(union)))
        return DAGSpec([bases], self.rng.choice([a.combiner, b.combiner]))

    # ------------------------------------------------------------- score+record
    def _score_and_record(self, spec: DAGSpec) -> Dict[str, object]:
        score = StackedDAG(spec, self.min_train, self.n_samples, self.seed).score(self.X)
        dsr = None
        series = score.get("outcome_series") or []
        if score.get("n", 0) >= 8 and len(series) >= 8:
            try:
                dsr = float(self.ev.evaluate_outcomes(series, n_trials=max(1, self._budget)).dsr)
            except Exception:
                dsr = None
        score["dsr"] = dsr
        self.lb.record(spec, score, dataset_id=self.dataset_id, dsr=dsr)
        self._seen.add(spec.signature())
        return score

    # -------------------------------------------------------------- strategies
    def random_search(self, n: int = 10) -> Optional[Dict]:
        self._budget = max(self._budget, n)
        for _ in range(n):
            self._score_and_record(self._random_spec())
        return self.best()

    def evolutionary_search(self, generations: int = 4, pop: int = 6) -> Optional[Dict]:
        self._budget = max(self._budget, generations * pop)
        scored = [(self._score_and_record(s), s) for s in (self._random_spec() for _ in range(pop))]
        for _ in range(max(0, generations - 1)):
            ranked = sorted(scored, key=lambda z: _skill(z[0]), reverse=True)
            elites = [s for _, s in ranked[: max(2, pop // 3)]]
            newpop = list(elites)
            while len(newpop) < pop:
                if self.rng.random() < 0.5 and len(elites) >= 2:
                    child = self._crossover(self.rng.choice(elites), self.rng.choice(elites))
                else:
                    child = self._mutate(self.rng.choice(elites))
                newpop.append(child)
            scored = [(self._score_and_record(s), s) for s in newpop]
        return self.best()

    def optuna_search(self, n_trials: int = 20) -> Optional[Dict]:
        if not _HAS_OPTUNA:
            raise RuntimeError("optuna not installed; use random_search/evolutionary_search")
        self._budget = max(self._budget, n_trials)

        def objective(trial) -> float:
            k = trial.suggest_int("n_base", 1, self.max_base)
            bases: List[str] = []
            for i in range(k):
                b = trial.suggest_categorical(f"b{i}", self.pool)
                if b not in bases:
                    bases.append(b)
            combiner = trial.suggest_categorical("combiner", COMBINERS)
            return _skill(self._score_and_record(DAGSpec([bases], combiner)))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)
        return self.best()

    def best(self) -> Optional[Dict]:
        """The honest winner: best DSR-significant DAG (selection-bias deflated by
        the search budget); falls back to best raw skill if no DSR is available."""
        return (self.lb.best(metric="dsr", dataset_id=self.dataset_id)
                or self.lb.best(metric="crps_skill", dataset_id=self.dataset_id))


__all__ = ["DAGSearch", "DEFAULT_POOL", "COMBINERS", "optuna_available"]
