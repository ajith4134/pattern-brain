"""Features 1, 2 & 3 — the evolution engines (build-tracker final phase, slice 2;
PLAN.md §4). Each EVOLVES candidates and promotes them ONLY through the shared
5-layer :class:`~pattern_brain.evaluator.Evaluator` (slice 1) — the hard gate.

Three levels of the same evolutionary loop (Block 25's hierarchy):

* **Feature 1 — :class:`ModelEvolver`** operates *within a fixed algorithm*:
  create/mutate/crossover a model instance's hyperparameters (a forecaster node's
  ``lag``, ``alpha`` … ). Search in parameter space.
* **Feature 3 — :class:`PathwayEvolver`** operates on the *graph topology*: which
  nodes connect to which (a Connector pathway). Mutate = swap/insert/drop a node
  in a layer-legal slot; crossover = splice two parent pathways. The unit of
  reputation is the pathway (Block 17).
* **Feature 2 — :class:`AlgorithmEvolver`** operates one level above Feature 1:
  it invents *new algorithms* as symbolic expression trees (genetic programming
  over a function set — AutoML-Zero / FunSearch-style), not just settings within
  a fixed one.

All three share :class:`Evolver` — the GA loop (init → score-via-Evaluator →
select-via-Evaluator → breed) — and emit an :class:`EvolutionResult` whose
``history`` (candidates born → tested → promoted/killed) is exactly the §8
dashboard "Evolution feed".

Domain-agnostic (PLAN.md §0 / Rule 23): fitness is a generic *directional reward*
outcome series scored by the (domain-agnostic) Evaluator. Nothing here references
candles/order books; the "predict the next value of channel 0" task is a generic
sequence task, and the financial reading lives only in how data is produced.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluator import (
    Evaluator, EvaluationReport, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
)
from .registry import create, by_layer
from .routing import LAYER_ORDER, _LAYER_IX

# ----------------------------------------------------------------- fitness
LAG_CONTEXT = 16  # how many past points a predictor sees per step


def directional_candidate_fn(make_predictor: Callable[[np.ndarray], Callable[[np.ndarray], float]]):
    """Build an Evaluator ``CandidateFn`` from a predictor factory.

    ``make_predictor(train) -> predict(window) -> float`` returns a one-step
    predictor of channel 0 fitted on the train fold. The outcome on each test step
    is a generic **directional reward**: +1 if the predicted move direction
    matches the realized one, −1 if opposite, 0 on a flat call. The Evaluator then
    scores the consistency of that reward stream (Sharpe/PSR/…)."""
    def candidate_fn(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        train = np.asarray(train, dtype=float)
        test = np.asarray(test, dtype=float)
        try:
            predict = make_predictor(train)
        except Exception:
            return np.zeros(test.shape[0])
        series = np.concatenate([train[-LAG_CONTEXT:, 0], test[:, 0]])
        offset = min(LAG_CONTEXT, train.shape[0])
        outs: List[float] = []
        for i in range(test.shape[0]):
            window = series[max(0, i + offset - LAG_CONTEXT): i + offset]
            if window.size < 2:
                outs.append(0.0); continue
            last = window[-1]
            try:
                pred = float(predict(window))
            except Exception:
                outs.append(0.0); continue
            actual = test[i, 0]
            pdir = np.sign(pred - last)
            adir = np.sign(actual - last)
            outs.append(0.0 if pdir == 0 else float(pdir * adir))
        return np.asarray(outs, dtype=float)
    return candidate_fn


# ----------------------------------------------------------------- records
@dataclass
class Individual:
    """One candidate in a population."""
    genome: Any
    generation: int
    parents: Tuple[Any, ...] = ()
    origin: str = "init"            # init | mutate | crossover
    report: Optional[EvaluationReport] = None
    outcomes: Optional[np.ndarray] = None   # walk-forward outcome series (for PBO)

    def fitness(self) -> float:
        """Scalar fitness for breeding selection (UCB if scored, else -inf)."""
        return self.report.ucb if self.report is not None else -1e18

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": _genome_repr(self.genome),
            "generation": self.generation,
            "origin": self.origin,
            "report": self.report.to_dict() if self.report else None,
        }


def _genome_repr(g: Any) -> Any:
    if isinstance(g, tuple) and len(g) == 2 and isinstance(g[1], dict):  # model genome
        return {"node_type": g[0], "params": g[1]}
    if isinstance(g, (list, tuple)) and all(isinstance(x, str) for x in g):  # pathway
        return list(g)
    return str(g)


@dataclass
class EvolutionResult:
    feature: str
    best: Optional[Individual]
    admitted: List[Individual] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    generations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "generations": self.generations,
            "best": self.best.to_dict() if self.best else None,
            "admitted": [i.to_dict() for i in self.admitted],
            "history": self.history,
        }


# ----------------------------------------------------------------- the loop
class Evolver:
    """Shared GA loop. Subclasses implement ``random_genome``, ``mutate``,
    ``crossover`` and ``make_predictor_fn`` (genome -> predictor factory)."""

    feature = "evolver"

    def __init__(self, evaluator: Optional[Evaluator] = None, population: int = 8,
                 generations: int = 4, elite: int = 3, seed: int = 0,
                 psr_threshold: float = 0.55) -> None:
        # the directional reward is bounded/short, so use lenient significance
        # thresholds here (the gate's job is relative selection, not a 0.95 bar).
        self.evaluator = evaluator or Evaluator(n_splits=4, embargo=0.02,
                                                psr_threshold=psr_threshold,
                                                dsr_threshold=psr_threshold)
        self.population = population
        self.generations = generations
        self.elite = elite
        self.rng = random.Random(seed)
        self._nprng = np.random.default_rng(seed)

    # --- to override ---------------------------------------------------
    def random_genome(self) -> Any: raise NotImplementedError
    def mutate(self, g: Any) -> Any: raise NotImplementedError
    def crossover(self, a: Any, b: Any) -> Any: raise NotImplementedError
    def make_predictor_fn(self, g: Any): raise NotImplementedError

    # --- the loop ------------------------------------------------------
    def _score(self, ind: Individual, X: np.ndarray, n_trials: int) -> None:
        cand = directional_candidate_fn(self.make_predictor_fn(ind.genome))
        ind.outcomes = self.evaluator.walk_forward_outcomes(cand, X)
        ind.report = self.evaluator.evaluate_outcomes(ind.outcomes, n_trials=n_trials)

    def run(self, X: np.ndarray) -> EvolutionResult:
        X = np.asarray(X, dtype=float)
        history: List[Dict[str, Any]] = []
        pop = [Individual(self.random_genome(), 0, origin="init")
               for _ in range(self.population)]
        all_scored: List[Individual] = []
        for gen in range(self.generations):
            n_trials = max(1, len(pop))     # deflate Sharpe by #candidates this gen
            for ind in pop:
                if ind.report is None:
                    self._score(ind, X, n_trials)
                    all_scored.append(ind)
                    history.append({"event": "tested", **ind.to_dict()})
            pop.sort(key=lambda i: i.fitness(), reverse=True)
            elites = pop[:self.elite]
            # breed the next generation from the elites
            children: List[Individual] = []
            while len(children) < self.population - len(elites):
                if len(elites) >= 2 and self.rng.random() < 0.5:
                    a, b = self.rng.sample(elites, 2)
                    g = self.crossover(a.genome, b.genome)
                    child = Individual(g, gen + 1, (a.genome, b.genome), "crossover")
                else:
                    parent = self.rng.choice(elites)
                    child = Individual(self.mutate(parent.genome), gen + 1,
                                       (parent.genome,), "mutate")
                children.append(child)
                history.append({"event": "born", **child.to_dict()})
            pop = elites + children
        # final scoring of any unscored survivors
        for ind in pop:
            if ind.report is None:
                self._score(ind, X, max(1, len(pop)))
                all_scored.append(ind)
                history.append({"event": "tested", **ind.to_dict()})

        # admission gate (Evaluator Layers 3-4) over everything scored
        gate = self._admit(all_scored)
        for ind in gate["admitted"]:
            history.append({"event": "promoted", **ind.to_dict()})
        best = max(all_scored, key=lambda i: i.fitness(), default=None)
        return EvolutionResult(self.feature, best, gate["admitted"], history,
                               self.generations)

    def _admit(self, scored: List[Individual]) -> Dict[str, Any]:
        """Population admission with PROPER multiple-testing control: the whole
        point of the gate is to reject the best-of-many when it's a fluke.

        1. Re-deflate every candidate's DSR against the REAL cross-sectional
           variance of the candidate Sharpes and the TOTAL number tried (not the
           per-generation count) — selecting the best of N inflates significance.
        2. Feed every candidate's outcome stream into Layer-3 CSCV/PBO so an
           overfit population is flagged (on a true null PBO ~ 0.5 -> rejected).
        """
        usable = [i for i in scored if i.outcomes is not None and i.outcomes.size >= 3]
        if not usable:
            return {"admitted": [], "selection": {"admitted": []}}
        N = len(usable)
        srs = np.array([i.report.sharpe for i in usable], dtype=float)
        var_sr = float(max(np.var(srs), 1e-6)) if N > 1 else 1.0 / max(2, usable[0].outcomes.size)
        thr = self.evaluator.psr_threshold
        for i in usable:
            i.report.dsr = deflated_sharpe_ratio(i.outcomes, n_trials=N, var_sr=var_sr)
            reasons = []
            if i.outcomes.size < 3:
                reasons.append("too few observations")
            if i.report.psr < thr:
                reasons.append(f"PSR {i.report.psr:.3f} < {thr}")
            if i.report.dsr < self.evaluator.dsr_threshold:
                reasons.append(f"DSR {i.report.dsr:.3f} < {self.evaluator.dsr_threshold} "
                               f"(deflated over {N} trials)")
            i.report.reasons = reasons
            i.report.passed = not reasons
        # align outcome streams to a common length -> CSCV/PBO matrix
        L = min(i.outcomes.size for i in usable)
        perf = np.column_stack([i.outcomes[-L:] for i in usable]) if L >= 8 else None
        sel = self.evaluator.select([i.report for i in usable], perf_matrix=perf)
        admitted = [usable[idx] for idx in sel["admitted"]]
        return {"admitted": admitted, "selection": sel}


# ========================================================== Feature 1: models
# forecaster node types and the numeric hyperparameters Feature 1 may tune.
_F1_SPACE: Dict[str, Dict[str, Tuple[float, float, bool]]] = {
    # node_type: {param: (lo, hi, is_int)}
    "autoregressive": {"order": (1, 10, True)},
    "exp_smoothing": {"alpha": (0.05, 0.95, False)},
    "holt_linear_forecast": {"alpha": (0.05, 0.95, False), "beta": (0.05, 0.95, False)},
    "theta_forecast": {"alpha": (0.05, 0.95, False)},
    "knn_forecast": {"lag": (2, 12, True)},
    "svr_forecast": {"lag": (2, 12, True)},
    "ridge_forecast": {"lag": (2, 12, True)},
}


class ModelEvolver(Evolver):
    """Feature 1 — evolve a model instance's hyperparameters within a fixed
    algorithm family (forecaster nodes). Genome = ``(node_type, params)``."""
    feature = "feature1_model"

    def __init__(self, node_types: Optional[Sequence[str]] = None, **kw):
        super().__init__(**kw)
        self.node_types = list(node_types) if node_types else list(_F1_SPACE)

    def random_genome(self):
        nt = self.rng.choice(self.node_types)
        return (nt, self._rand_params(nt))

    def _rand_params(self, nt):
        params = {}
        for p, (lo, hi, is_int) in _F1_SPACE.get(nt, {}).items():
            v = self._nprng.uniform(lo, hi)
            params[p] = int(round(v)) if is_int else round(float(v), 4)
        return params

    def mutate(self, g):
        nt, params = g
        params = dict(params)
        space = _F1_SPACE.get(nt, {})
        if space:
            p = self.rng.choice(list(space))
            lo, hi, is_int = space[p]
            cur = params.get(p, (lo + hi) / 2)
            nv = float(np.clip(cur * self._nprng.uniform(0.6, 1.6), lo, hi))
            params[p] = int(round(nv)) if is_int else round(nv, 4)
        return (nt, params)

    def crossover(self, a, b):
        # same family -> blend params; different family -> inherit one parent's
        (nta, pa), (ntb, pb) = a, b
        if nta != ntb:
            return (nta, dict(pa))
        child = {}
        for p in set(pa) | set(pb):
            child[p] = pa.get(p, pb.get(p)) if self.rng.random() < 0.5 else pb.get(p, pa.get(p))
        return (nta, child)

    def make_predictor_fn(self, g):
        nt, params = g

        def make_predictor(train):
            def predict(window):
                node = create(nt, **params)
                b = node.predict(window.reshape(-1, 1))
                pl = b.payload
                if "next_vector" in pl and pl["next_vector"]:
                    return float(pl["next_vector"][0])
                if "estimate" in pl:
                    return float(pl["estimate"])
                return float(window[-1])
            return predict
        return make_predictor


# ======================================================== Feature 3: pathways
class PathwayEvolver(Evolver):
    """Feature 3 — evolve the graph topology (a Connector pathway). Genome = a
    list of node_types that respects the Block-18 layer ordering and ends in a
    decision node. Reputation attaches to the pathway (Block 17)."""
    feature = "feature3_pathway"

    # forecaster nodes whose output direction we read as the pathway's prediction
    _HEADS = ["autoregressive", "theta_forecast", "holt_linear_forecast",
              "drift_forecast", "ridge_forecast", "knn_forecast"]
    _PRE = {"signal": ["difference", "moving_average", "ewma", "detrend"],
            "noise": ["savgol_denoise", "pca_denoise", "median_filter"]}

    def random_genome(self):
        path: List[str] = []
        if self.rng.random() < 0.7:
            path.append(self.rng.choice(self._PRE["signal"]))
        if self.rng.random() < 0.4:
            path.append(self.rng.choice(self._PRE["noise"]))
        path.append(self.rng.choice(self._HEADS))
        return path

    def mutate(self, g):
        path = list(g)
        op = self.rng.random()
        if op < 0.5 and path:                       # swap one node for a sibling
            i = self.rng.randrange(len(path))
            layer = create(path[i]).layer
            pool = self._PRE.get(layer, self._HEADS) if i < len(path) - 1 else self._HEADS
            path[i] = self.rng.choice(pool)
        elif op < 0.75 and len(path) < 4:           # insert a preprocessor
            path.insert(0, self.rng.choice(self._PRE["signal"]))
        elif len(path) > 1:                          # drop a non-head node
            del path[self.rng.randrange(len(path) - 1)]
        return self._legalize(path)

    def crossover(self, a, b):
        a, b = list(a), list(b)
        cut_a = self.rng.randint(0, max(0, len(a) - 1))
        cut_b = self.rng.randint(0, max(0, len(b) - 1))
        child = a[:cut_a] + b[cut_b:]
        return self._legalize(child)

    def _legalize(self, path):
        """Keep only known nodes in non-decreasing layer order, ending in a head."""
        seen, out = set(), []
        last_ix = -1
        for nt in path:
            try:
                li = _LAYER_IX.get(create(nt).layer, 99)
            except Exception:
                continue
            if nt in seen or li < last_ix or li >= _LAYER_IX["sequence"]:
                continue
            out.append(nt); seen.add(nt); last_ix = li
        out.append(self.rng.choice(self._HEADS) if not out or out[-1] not in self._HEADS
                   else out[-1])
        if out[-1] not in self._HEADS:
            out.append(self.rng.choice(self._HEADS))
        return out

    def make_predictor_fn(self, g):
        from .connector import Connector
        path = list(g)
        head = path[-1]

        def make_predictor(train):
            def predict(window):
                # run the preprocessors as a bus, then the forecaster head
                try:
                    conn = Connector(path, name="evolved")
                    res = conn.run(window.reshape(-1, 1))
                    for hop in reversed(res.hops):
                        pl = hop.belief.payload
                        if "next_vector" in pl and pl["next_vector"]:
                            return float(pl["next_vector"][0])
                        if "estimate" in pl:
                            return float(pl["estimate"])
                except Exception:
                    pass
                return float(create(head).predict(window.reshape(-1, 1))
                             .payload.get("estimate", window[-1]))
            return predict
        return make_predictor


# ===================================================== Feature 2: algorithms
# Genetic programming over expression trees: invent a NEW forecasting equation.
_F2_BINOPS = ["+", "-", "*", "/"]
_F2_UNOPS = ["neg", "sin", "tanh", "abs"]


class AlgorithmEvolver(Evolver):
    """Feature 2 — invent new algorithms as symbolic expression trees mapping the
    recent window to a one-step prediction (genetic programming). One level above
    Feature 1: it creates new equations, not settings within a fixed one."""
    feature = "feature2_algorithm"

    def __init__(self, max_depth: int = 3, n_lags: int = 4, **kw):
        super().__init__(**kw)
        self.max_depth = max_depth
        self.n_lags = n_lags

    # tree = ("op", child...) | ("lag", k) | ("const", v) | ("mean",) | ("last",)
    def random_genome(self, depth=None):
        depth = self.max_depth if depth is None else depth
        if depth <= 0 or self.rng.random() < 0.3:
            r = self.rng.random()
            if r < 0.5:
                return ("lag", self.rng.randrange(self.n_lags))
            if r < 0.7:
                return ("const", round(self._nprng.uniform(-1, 1), 3))
            return (self.rng.choice(["mean", "last", "slope"]),)
        if self.rng.random() < 0.6:
            op = self.rng.choice(_F2_BINOPS)
            return (op, self.random_genome(depth - 1), self.random_genome(depth - 1))
        return (self.rng.choice(_F2_UNOPS), self.random_genome(depth - 1))

    def _subtrees(self, tree, acc=None, path=()):
        acc = [] if acc is None else acc
        acc.append((path, tree))
        if isinstance(tree, tuple) and tree[0] in _F2_BINOPS + _F2_UNOPS:
            for i, ch in enumerate(tree[1:], start=1):
                self._subtrees(ch, acc, path + (i,))
        return acc

    def _replace(self, tree, path, new):
        if not path:
            return new
        tree = list(tree)
        tree[path[0]] = self._replace(tree[path[0]], path[1:], new)
        return tuple(tree)

    def mutate(self, g):
        spots = self._subtrees(g)
        path, _ = self.rng.choice(spots)
        return self._replace(g, path, self.random_genome(depth=2))

    def crossover(self, a, b):
        pa, _ = self.rng.choice(self._subtrees(a))
        _, sub = self.rng.choice(self._subtrees(b))
        return self._replace(a, pa, sub)

    def _eval(self, tree, window):
        t = tree[0]
        if t == "lag":
            k = tree[1]
            return float(window[-1 - k]) if k < len(window) else float(window[-1])
        if t == "const":
            return float(tree[1])
        if t == "mean":
            return float(np.mean(window))
        if t == "last":
            return float(window[-1])
        if t == "slope":
            return float(window[-1] - window[-2]) if len(window) > 1 else 0.0
        if t == "neg":
            return -self._eval(tree[1], window)
        if t == "sin":
            return math.sin(self._eval(tree[1], window))
        if t == "tanh":
            return math.tanh(self._eval(tree[1], window))
        if t == "abs":
            return abs(self._eval(tree[1], window))
        a = self._eval(tree[1], window); b = self._eval(tree[2], window)
        if t == "+":
            return a + b
        if t == "-":
            return a - b
        if t == "*":
            return a * b
        if t == "/":
            return a / b if abs(b) > 1e-9 else 1.0   # protected division
        return 0.0

    def make_predictor_fn(self, g):
        def make_predictor(train):
            def predict(window):
                try:
                    v = self._eval(g, window)
                    return float(v) if math.isfinite(v) else float(window[-1])
                except Exception:
                    return float(window[-1])
            return predict
        return make_predictor


__all__ = [
    "Evolver", "ModelEvolver", "PathwayEvolver", "AlgorithmEvolver",
    "Individual", "EvolutionResult", "directional_candidate_fn",
]
