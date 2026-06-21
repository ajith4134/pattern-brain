"""The shared 5-layer Evaluator (build-tracker final phase; PLAN.md §4).

This is the HARD GATE Features 1, 2 & 3 all depend on: a candidate (a mutated
model, a discovered algorithm, or an evolved pathway) must pass this before it
may replace an incumbent. The five layers (Blocks 21-22):

* **Layer 0 — data hygiene:** purged + embargoed walk-forward splits
  (López de Prado 2017). Naive k-fold leaks future information through
  overlapping label windows; this produces leakage-safe (train, test) folds.
* **Layer 1 — per-candidate significance:** Sharpe + Probabilistic Sharpe Ratio
  + Deflated Sharpe Ratio + Minimum Track Record Length (Bailey & López de
  Prado) — catches under-sampled / multiple-testing-inflated scores.
* **Layer 2 — online / non-stationary scoring:** Sliding-Window / Discounted UCB
  per candidate — subsumes pathway death (low score) and curiosity (exploration
  bonus) as ONE mechanism (Block 22).
* **Layer 3 — population multi-objective gate:** NSGA-II-style non-dominated
  (Pareto) sorting across several objectives, paired with CSCV/PBO (probability
  of backtest overfitting) so the front itself isn't an overfitting mirage.
* **Layer 4 — anti-gaming anchor:** a fixed set of held-out episodes every
  promoted candidate must stay sane on.

Domain-agnostic (PLAN.md §0 / Rule 23) — resolves the §0 tension explicitly:
the evaluator operates on a generic **outcome series** (a 1-D array of per-period
"higher-is-better" values). "Sharpe" and "drawdown" here are generic statistics
of that series (mean/std; worst peak-to-trough of its cumulative sum), NOT
anything candle/order-book specific. WHICH statistics define the Pareto front is
a pluggable :class:`ObjectiveSpec`. The domain-specific part — turning a
candidate's predictions into the outcome series (returns for finance, negative
error for another domain) — lives in the caller / an adapter, never here.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

# A candidate, evaluated on a (train, test) fold, yields the test outcome series.
CandidateFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ============================================================ Layer 0: hygiene
def purged_walk_forward_splits(
    n_samples: int, n_splits: int = 5, embargo: float = 0.02, min_train: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leakage-safe expanding walk-forward splits (López de Prado, Layer 0).

    Time is split into ``n_splits + 1`` contiguous blocks; fold *i* tests on
    block *i+1* and trains on everything strictly before it, MINUS an ``embargo``
    gap of ``embargo * n_samples`` samples adjacent to the test block (this both
    purges label-overlap and embargoes serial-correlation leakage). Returns a
    list of ``(train_idx, test_idx)``; folds whose train side is shorter than
    ``min_train`` are skipped.
    """
    if n_samples < n_splits + 2:
        raise ValueError(f"need >= {n_splits + 2} samples for {n_splits} splits")
    bounds = np.linspace(0, n_samples, n_splits + 2, dtype=int)
    gap = int(round(embargo * n_samples))
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        test_lo, test_hi = bounds[i], bounds[i + 1]
        train_hi = max(0, test_lo - gap)
        if train_hi < min_train:
            continue
        out.append((np.arange(0, train_hi), np.arange(test_lo, test_hi)))
    return out


# ========================================================== Layer 1: significance
def sharpe(outcomes: Sequence[float], eps: float = 1e-12) -> float:
    """Per-observation Sharpe (mean / std) of a generic outcome series."""
    x = np.asarray(outcomes, dtype=float)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd + eps))


def max_drawdown(outcomes: Sequence[float]) -> float:
    """Worst peak-to-trough drop of the cumulative outcome curve (>= 0)."""
    x = np.asarray(outcomes, dtype=float)
    if x.size == 0:
        return 0.0
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(curve)
    return float(np.max(peak - curve))


def stability(outcomes: Sequence[float]) -> float:
    """R² of the cumulative curve vs a straight line — how steadily it accrues."""
    x = np.asarray(outcomes, dtype=float)
    if x.size < 3:
        return 0.0
    curve = np.cumsum(x)
    t = np.arange(curve.size)
    a, b = np.polyfit(t, curve, 1)
    pred = a * t + b
    ss_res = float(np.sum((curve - pred) ** 2))
    ss_tot = float(np.sum((curve - curve.mean()) ** 2)) + 1e-12
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def probabilistic_sharpe_ratio(outcomes: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark), correcting the sample SR for length, skew and
    kurtosis (Bailey & López de Prado 2012)."""
    x = np.asarray(outcomes, dtype=float)
    n = x.size
    if n < 3:
        return 0.0
    sr = sharpe(x)
    sd = x.std(ddof=1) + 1e-12
    skew = float(np.mean(((x - x.mean()) / sd) ** 3))
    kurt = float(np.mean(((x - x.mean()) / sd) ** 4))  # non-excess (Pearson)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum sample Sharpe under ``n_trials`` independent trials with
    Sharpe variance ``var_sr`` (the deflation benchmark for DSR)."""
    n_trials = max(1, int(n_trials))
    if n_trials == 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    e = math.e
    term = ((1 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * norm.ppf(1.0 - 1.0 / (n_trials * e)))
    return float(math.sqrt(max(0.0, var_sr)) * term)


def deflated_sharpe_ratio(outcomes: Sequence[float], n_trials: int = 1,
                          var_sr: float = 1.0) -> float:
    """DSR = PSR measured against the expected-maximum Sharpe over ``n_trials``
    (so a strategy selected from many isn't fooled by selection bias)."""
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(outcomes, sr_benchmark=sr_star)


def min_track_record_length(outcomes: Sequence[float], sr_benchmark: float = 0.0,
                            target_prob: float = 0.95) -> float:
    """Min number of observations for PSR(benchmark) to reach ``target_prob``
    (NaN/inf if the sample SR doesn't exceed the benchmark)."""
    x = np.asarray(outcomes, dtype=float)
    if x.size < 3:
        return float("inf")
    sr = sharpe(x)
    if sr <= sr_benchmark:
        return float("inf")
    sd = x.std(ddof=1) + 1e-12
    skew = float(np.mean(((x - x.mean()) / sd) ** 3))
    kurt = float(np.mean(((x - x.mean()) / sd) ** 4))
    z = norm.ppf(target_prob)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
                 * (z / (sr - sr_benchmark)) ** 2)


# ====================================================== Layer 2: non-stationary
def discounted_ucb_score(outcomes: Sequence[float], gamma: float = 0.95,
                         c: float = 2.0) -> float:
    """Discounted-UCB value of a candidate's outcome stream: a recency-weighted
    mean plus an exploration bonus (Block 22). Recent outcomes weigh more, so a
    candidate that has decayed scores low (pathway death) while a rarely-used one
    keeps an exploration bonus (curiosity) — one mechanism, not two rules."""
    x = np.asarray(outcomes, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    w = gamma ** np.arange(n)[::-1]
    wsum = float(w.sum())
    mean = float((w * x).sum() / (wsum + 1e-12))
    bonus = math.sqrt(c * math.log(max(2.0, wsum + 1.0)) / (wsum + 1e-12))
    return mean + bonus


def sliding_window_ucb_score(outcomes: Sequence[float], window: int = 50,
                             c: float = 2.0) -> float:
    """Sliding-Window UCB: UCB over only the last ``window`` outcomes."""
    x = np.asarray(outcomes, dtype=float)[-window:]
    n = x.size
    if n == 0:
        return 0.0
    return float(x.mean() + math.sqrt(c * math.log(max(2.0, n)) / n))


# ========================================================= Layer 3: population
def nondominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """NSGA-II fast non-dominated sort (all objectives MAXIMIZED). Returns a list
    of fronts; front 0 is the Pareto-optimal set."""
    P = np.asarray(objectives, dtype=float)
    n = P.shape[0]
    S: List[List[int]] = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    def dominates(a, b):
        return np.all(P[a] >= P[b]) and np.any(P[a] > P[b])

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def pareto_front(objectives: np.ndarray) -> List[int]:
    """Indices of the non-dominated (Pareto-optimal) candidates (maximization)."""
    if len(objectives) == 0:
        return []
    return nondominated_sort(objectives)[0]


def cscv_pbo(perf_matrix: np.ndarray, n_blocks: int = 8) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    ``perf_matrix`` is ``(n_observations, n_strategies)`` of per-period outcomes.
    Rows are split into ``n_blocks`` contiguous blocks; for every way of choosing
    half the blocks as in-sample, the best-IS strategy's OUT-of-sample rank is
    measured. PBO = fraction of splits where the IS-best strategy lands in the
    bottom half OOS (its logit < 0). Lower is better; ~0.5 means the selection is
    no better than chance (overfit)."""
    M = np.asarray(perf_matrix, dtype=float)
    T, N = M.shape
    if N < 2 or T < n_blocks or n_blocks % 2 != 0:
        return float("nan")
    blocks = np.array_split(np.arange(T), n_blocks)
    logits: List[float] = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])
        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        best = int(np.argmax(is_perf))
        # relative OOS rank of the IS-best strategy in (0, 1)
        rank = float((oos_perf <= oos_perf[best]).sum()) / N
        rank = min(max(rank, 1.0 / (N + 1)), 1.0 - 1.0 / (N + 1))
        logits.append(math.log(rank / (1.0 - rank)))
    if not logits:
        return float("nan")
    return float(np.mean(np.asarray(logits) < 0.0))


# ============================================================= Layer 4: anchor
def anchor_check(anchor_outcomes: Sequence[Sequence[float]],
                 min_mean: float = 0.0) -> bool:
    """Every anchor episode must stay sane: mean outcome >= ``min_mean`` on each
    of the fixed held-out episodes (anti-gaming)."""
    for ep in anchor_outcomes:
        x = np.asarray(ep, dtype=float)
        if x.size == 0 or float(x.mean()) < min_mean:
            return False
    return True


# ============================================================ objective spec
@dataclass(frozen=True)
class ObjectiveSpec:
    """Which generic series statistics define the Layer-3 Pareto front. Pluggable
    so a domain can choose its objectives (resolves PLAN.md §0's tension). Each
    entry maps a name to a function ``outcome_series -> float`` (all MAXIMIZED;
    use negatives for costs like drawdown)."""
    metrics: Tuple[Tuple[str, Callable[[np.ndarray], float]], ...]

    def vector(self, outcomes: np.ndarray) -> Dict[str, float]:
        return {name: float(fn(outcomes)) for name, fn in self.metrics}


DEFAULT_OBJECTIVES = ObjectiveSpec((
    ("mean", lambda x: float(np.mean(x)) if len(x) else 0.0),
    ("sharpe", sharpe),
    ("neg_drawdown", lambda x: -max_drawdown(x)),
    ("stability", stability),
))


# ================================================================ the Evaluator
@dataclass
class EvaluationReport:
    """Per-candidate result (Layers 0-2 + objective vector)."""
    n_obs: int
    sharpe: float
    psr: float
    dsr: float
    min_trl: float
    ucb: float
    objectives: Dict[str, float]
    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_obs": self.n_obs, "sharpe": self.sharpe, "psr": self.psr,
            "dsr": self.dsr, "min_trl": (None if math.isinf(self.min_trl) else self.min_trl),
            "ucb": self.ucb, "objectives": self.objectives,
            "passed": self.passed, "reasons": list(self.reasons),
        }
        return d


class Evaluator:
    """Run the 5-layer gauntlet. ``evaluate_candidate`` applies Layers 0-2 to one
    candidate; ``select`` applies Layers 3-4 to a population (the admission gate
    Features 1/2/3 call before promoting anything)."""

    def __init__(self, n_splits: int = 5, embargo: float = 0.02,
                 psr_threshold: float = 0.95, dsr_threshold: float = 0.95,
                 gamma: float = 0.95, objectives: ObjectiveSpec = DEFAULT_OBJECTIVES,
                 pbo_threshold: float = 0.5) -> None:
        self.n_splits = n_splits
        self.embargo = embargo
        self.psr_threshold = psr_threshold
        self.dsr_threshold = dsr_threshold
        self.gamma = gamma
        self.objectives = objectives
        self.pbo_threshold = pbo_threshold

    # ---- Layer 0 + run -> outcome series
    def walk_forward_outcomes(self, candidate: CandidateFn, X: np.ndarray) -> np.ndarray:
        """Run ``candidate(train, test)`` over purged walk-forward folds and
        concatenate the per-fold test outcome series (leakage-safe, Layer 0)."""
        X = np.asarray(X, dtype=float)
        splits = purged_walk_forward_splits(X.shape[0], self.n_splits, self.embargo)
        pieces: List[np.ndarray] = []
        for tr, te in splits:
            out = np.asarray(candidate(X[tr], X[te]), dtype=float).ravel()
            if out.size:
                pieces.append(out)
        return np.concatenate(pieces) if pieces else np.array([])

    # ---- Layers 1 + 2 on an outcome series
    def evaluate_outcomes(self, outcomes: Sequence[float], n_trials: int = 1,
                          var_sr: Optional[float] = None) -> EvaluationReport:
        x = np.asarray(outcomes, dtype=float)
        sr = sharpe(x)
        # If the cross-trial Sharpe variance isn't supplied, estimate the Sharpe
        # ESTIMATOR variance from the sample (Lo 2002: ~(1 + sr^2/2)/n). Using a
        # hardcoded 1.0 would wildly over-deflate short directional-reward streams.
        if var_sr is None:
            n = max(2, x.size)
            var_sr = (1.0 + 0.5 * sr ** 2) / n
        psr = probabilistic_sharpe_ratio(x)
        dsr = deflated_sharpe_ratio(x, n_trials=n_trials, var_sr=var_sr)
        trl = min_track_record_length(x)
        ucb = discounted_ucb_score(x, gamma=self.gamma)
        objs = self.objectives.vector(x)
        reasons: List[str] = []
        if x.size < 3:
            reasons.append("too few observations")
        if psr < self.psr_threshold:
            reasons.append(f"PSR {psr:.3f} < {self.psr_threshold}")
        if dsr < self.dsr_threshold:
            reasons.append(f"DSR {dsr:.3f} < {self.dsr_threshold} (selection-bias deflated)")
        passed = not reasons
        return EvaluationReport(int(x.size), sr, psr, dsr, trl, ucb, objs, passed, reasons)

    def evaluate_candidate(self, candidate: CandidateFn, X: np.ndarray,
                           n_trials: int = 1, var_sr: Optional[float] = None) -> EvaluationReport:
        """Full Layer-0→2 evaluation of one candidate on dataset ``X``."""
        return self.evaluate_outcomes(self.walk_forward_outcomes(candidate, X),
                                      n_trials=n_trials, var_sr=var_sr)

    # ---- Layers 3 + 4 over a population
    def select(self, reports: Sequence[EvaluationReport],
               perf_matrix: Optional[np.ndarray] = None,
               anchors: Optional[Sequence[Sequence[Sequence[float]]]] = None,
               anchor_min_mean: float = 0.0) -> Dict[str, Any]:
        """Admission gate (Layers 3-4) over a population that already cleared
        Layers 1-2. Returns the Pareto front, the backtest-overfit probability,
        the anchor-survivors, and the final admitted indices."""
        names = [n for n, _ in self.objectives.metrics]
        obj = np.array([[r.objectives[n] for n in names] for r in reports], dtype=float) \
            if reports else np.empty((0, len(names)))
        front = pareto_front(obj)
        pbo = cscv_pbo(perf_matrix) if perf_matrix is not None else float("nan")
        anchor_ok = set(range(len(reports)))
        if anchors is not None:
            anchor_ok = {i for i in range(len(reports))
                         if i < len(anchors) and anchor_check(anchors[i], anchor_min_mean)}
        overfit = (not math.isnan(pbo)) and pbo > self.pbo_threshold
        admitted = [] if overfit else [i for i in front
                                       if reports[i].passed and i in anchor_ok]
        return {
            "pareto_front": list(front),
            "pbo": (None if math.isnan(pbo) else pbo),
            "overfit_flagged": bool(overfit),
            "anchor_survivors": sorted(anchor_ok),
            "admitted": sorted(admitted),
            "objective_names": names,
        }
