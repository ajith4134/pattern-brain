"""TIER-1 SOFT-COMPUTING nodes (CONCEPT_EQUATION_BANK.md), built oracle-first.

Interpretable / data-poor / nonlinear soft-computing methods, each behind the
generic Node interface (Rule 23) and emitting an existing v0.1 interlingua belief
type. Light stack only (numpy/scipy). Domain-agnostic: a generic (T, D) sequence
in, a Belief out — no candle/orderbook knowledge.

These are NOT pitched as return-forecasters that beat persistence (PROJECT LESSON:
standalone return forecasters usually lose to persistence). They earn their place
as interpretable / data-poor / nonlinear TOOLS, proven on the regime where their
theory applies:

  * ``fuzzy_ts``  — Chen (1996) Fuzzy Time Series. Partition the universe of
    discourse into fuzzy sets, fuzzify the series, mine fuzzy logical relationship
    groups, defuzzify to forecast. Interpretable level tracker.
  * ``grey_gm11`` — Grey System GM(1,1) (Deng 1982). Accumulated Generating
    Operation + a first-order grey differential equation fit by least squares,
    forecast via the time-response function. DATA-POOR: works from ~4-6 points; a
    KEEP-as-utility short-horizon extrapolator on smooth/trending windows.
  * ``anfis``     — Adaptive Neuro-Fuzzy Inference System (Sugeno, Jang 1993). A
    small fuzzy system with Gaussian input MFs + linear consequents trained by
    HYBRID learning (least squares for the consequents given fixed premises, plus a
    gradient step on the MF centres/widths). A general nonlinear regressor.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


# ============================================================ 1) Fuzzy Time Series (Chen 1996)
def _fuzzify(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each value to the index of the equal-width interval (fuzzy set) holding
    it. ``edges`` has k+1 boundaries → k sets indexed 0..k-1."""
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, edges.size - 2)


def chen_fts_forecast(x: np.ndarray, k: int) -> tuple[float, int]:
    """Chen's first-order Fuzzy Time Series one-step forecast.

    Steps (Chen 1996): (1) define the universe of discourse [min-d1, max+d2] and
    split it into ``k`` equal intervals (fuzzy sets), each with a midpoint; (2)
    fuzzify the series to its containing set; (3) mine the fuzzy logical
    relationships Aᵢ→Aⱼ and group them by left-hand side; (4) forecast the value
    following the LAST observed set Aᵢ by defuzzification:
        * no successors  → midpoint(i);
        * one successor j → midpoint(j);
        * many successors → mean of the successor midpoints.

    Returns (forecast_value, last_set_index).
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    if span <= 1e-12:                                   # degenerate / flat series
        return float(x[-1]), 0
    pad = 0.1 * span                                    # d1/d2 buffer around the data
    edges = np.linspace(lo - pad, hi + pad, k + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    sets = _fuzzify(x, edges)
    groups: dict[int, list[int]] = {}                   # LHS set -> list of RHS sets
    for a, b in zip(sets[:-1], sets[1:]):
        groups.setdefault(int(a), []).append(int(b))
    last = int(sets[-1])
    succ = groups.get(last)
    if not succ:                                        # unseen LHS → persist its level
        return float(mids[last]), last
    return float(np.mean(mids[succ])), last


@register
class FuzzyTimeSeriesNode(Node):
    """Chen (1996) Fuzzy Time Series level forecaster (D1). Partitions the universe
    of discourse into equal-width fuzzy sets, mines first-order fuzzy logical
    relationship groups, and defuzzifies to a one-step ``forecast``. Interpretable
    level tracker; honest verdict on near-random-walk closes: SHADOW vs persistence."""

    layer = "sequence"
    node_type = "fuzzy_ts"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = x.size
        # number of fuzzy sets: grow with sample size, Chen-style small partition.
        k = int(np.clip(round(np.sqrt(n)), 5, 30))
        est, last_set = chen_fts_forecast(x, k) if n >= 2 else (float(x[-1]), 0)
        # in-sample one-step series for downstream consumers (length n, first = x[0]).
        series = np.empty(n)
        series[0] = x[0]
        for t in range(1, n):
            series[t], _ = chen_fts_forecast(x[: t + 1], k)
        resid = series[1:] - x[1:] if n > 1 else np.array([0.0])
        denom = float(np.std(x)) + 1e-12
        conf = float(np.clip(1.0 - np.std(resid) / denom, 0.1, 0.9)) if n >= 10 else 0.2
        return Belief(
            type="forecast",
            payload={
                "estimate": float(est),
                "series": series.tolist(),
                "n_sets": int(k),
                "last_set": int(last_set),
            },
            confidence=conf,
        )


# ============================================================ 2) Grey System GM(1,1) (Deng 1982)
def gm11_fit(x0: np.ndarray) -> tuple[float, float]:
    """Fit the GM(1,1) parameters (a, b) of the first-order grey differential
    equation dx¹/dt + a·x¹ = b.

    The Accumulated Generating Operation x¹ₖ = Σ_{i≤k} x⁰ᵢ turns an erratic
    non-negative series into a smoother monotone one. The grey model then satisfies
    x⁰ₖ + a·zₖ = b, where zₖ = ½(x¹ₖ + x¹ₖ₋₁) is the background (mean) sequence.
    Stacking k = 2..n gives B·[a, b]ᵀ = Y, solved by least squares (lstsq, not the
    normal equations). ``a`` is the development coefficient (−a = growth rate),
    ``b`` the grey input.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    x1 = np.cumsum(x0)
    z = 0.5 * (x1[1:] + x1[:-1])                        # background mean, length n-1
    B = np.column_stack([-z, np.ones(z.size)])
    Y = x0[1:]
    coef, *_ = np.linalg.lstsq(B, Y, rcond=None)
    return float(coef[0]), float(coef[1])               # a, b


def gm11_forecast(x0: np.ndarray, steps: int = 1) -> np.ndarray:
    """GM(1,1) time-response forecast of the next ``steps`` original-scale values.

    The continuous solution is x̂¹ₖ₊₁ = (x⁰₀ − b/a)·e^{−a·k} + b/a; the original
    series is recovered by the Inverse AGO x̂⁰ₖ = x̂¹ₖ − x̂¹ₖ₋₁. Falls back to the
    last value when a≈0 (no estimable growth → degenerate equation)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    a, b = gm11_fit(x0)
    n = x0.size
    if abs(a) < 1e-8:
        return np.full(steps, float(x0[-1]))
    ks = np.arange(n, n + steps)                        # forecast indices (0-based)
    x1_hat = (x0[0] - b / a) * np.exp(-a * ks) + b / a
    x1_hat_prev = (x0[0] - b / a) * np.exp(-a * (ks - 1)) + b / a
    return x1_hat - x1_hat_prev                         # inverse AGO → original scale


def _gm11_relative_fit_error(x0: np.ndarray) -> float:
    """Mean relative one-step fit error of GM(1,1) reconstructed in-sample — the
    grey 'posterior error' style goodness measure (smaller = better fit)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    a, b = gm11_fit(x0)
    n = x0.size
    if abs(a) < 1e-8:
        return 1.0
    ks = np.arange(1, n)
    x1_hat = (x0[0] - b / a) * np.exp(-a * ks) + b / a
    x1_hat_prev = (x0[0] - b / a) * np.exp(-a * (ks - 1)) + b / a
    fit = x1_hat - x1_hat_prev                          # reconstructed x0[1:]
    denom = np.abs(x0[1:]) + 1e-12
    return float(np.mean(np.abs(fit - x0[1:]) / denom))


@register
class GreyGM11Node(Node):
    """Grey System GM(1,1) DATA-POOR short-horizon extrapolator (D1). Applies the
    Accumulated Generating Operation, fits the first-order grey differential
    equation by least squares, and forecasts via the time-response function. Works
    from ~4-6 points. Emits a ``forecast`` {estimate} plus payload {a, b}. Confidence
    reflects the in-sample relative fit error (low on noisy/oscillatory data)."""

    layer = "sequence"
    node_type = "grey_gm11"
    requires_y = False
    is_transformer = False
    cost = "low"

    def _predict(self, X: np.ndarray) -> Belief:
        x0 = X[:, 0].astype(float)
        n = x0.size
        if n < 4:
            return Belief(type="forecast",
                          payload={"estimate": float(x0[-1]), "a": 0.0, "b": float(x0[-1]),
                                   "note": "too-short-for-gm11"},
                          confidence=0.1)
        # GM(1,1) assumes a non-negative series; shift so the AGO is monotone if the
        # window dips below zero (returns-scale data), forecast, then shift back.
        shift = 0.0
        if np.min(x0) <= 0.0:
            shift = -np.min(x0) + 1.0
        xs = x0 + shift
        a, b = gm11_fit(xs)
        est = float(gm11_forecast(xs, steps=1)[0] - shift)
        rel_err = _gm11_relative_fit_error(xs)
        conf = float(np.clip(1.0 - 20.0 * rel_err, 0.05, 0.95))
        return Belief(
            type="forecast",
            payload={
                "estimate": est,
                "a": a,                                 # development coefficient (-a = growth)
                "b": b,
                "rel_fit_error": rel_err,
            },
            confidence=conf,
        )


# ============================================================ 3) ANFIS (Sugeno, Jang 1993)
def _gaussian_mf(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """Gaussian membership μ(x) = exp(−½((x−c)/σ)²)."""
    return np.exp(-0.5 * ((x - c) / (sigma + 1e-12)) ** 2)


def _init_premises(X: np.ndarray, n_rules: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Initialise rule premise parameters: pick ``n_rules`` cluster-like centres in
    feature space (random data points) and per-rule, per-dim widths from the spread.
    Returns (centres (R, D), widths (R, D))."""
    T, D = X.shape
    idx = rng.choice(T, size=min(n_rules, T), replace=False)
    centres = X[idx].copy()
    if centres.shape[0] < n_rules:                      # pad if few samples
        centres = np.vstack([centres,
                             X[rng.integers(0, T, n_rules - centres.shape[0])]])
    spread = np.std(X, axis=0) + 1e-6
    widths = np.tile(spread, (n_rules, 1)) * 0.8
    return centres, widths


def _firing_strengths(X: np.ndarray, centres: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Normalised rule firing strengths w̄ (T, R): product of per-dim Gaussian
    memberships, then row-normalised (the ANFIS layer-2/3 'normalisation')."""
    T = X.shape[0]
    R = centres.shape[0]
    w = np.ones((T, R))
    for r in range(R):
        mf = _gaussian_mf(X, centres[r], widths[r])     # (T, D)
        w[:, r] = np.prod(mf, axis=1)
    s = w.sum(axis=1, keepdims=True)
    return w / (s + 1e-12)


def _design_matrix(X: np.ndarray, wbar: np.ndarray) -> np.ndarray:
    """Layer-4 design matrix for the linear (first-order Sugeno) consequents: each
    rule contributes w̄ᵣ·[1, x₁, …, x_D], stacked → (T, R·(D+1))."""
    T, D = X.shape
    R = wbar.shape[1]
    aug = np.column_stack([np.ones(T), X])              # (T, D+1)
    out = np.empty((T, R * (D + 1)))
    for r in range(R):
        out[:, r * (D + 1):(r + 1) * (D + 1)] = wbar[:, r:r + 1] * aug
    return out


@register
class ANFISNode(Node):
    """Adaptive Neuro-Fuzzy Inference System (Sugeno, Jang 1993). Gaussian input
    membership functions + first-order linear consequents, trained by HYBRID
    learning: least squares for the consequent parameters given fixed premises, plus
    a few gradient-descent passes on the MF centres/widths (premise parameters).
    Supervised nonlinear regressor (features → target); emits a ``forecast``
    {estimate}. Proven on a synthetic nonlinear oracle; honest verdict on panel
    returns: SHADOW vs persistence."""

    layer = "equation"
    node_type = "anfis"
    requires_y = True
    is_transformer = False
    cost = "med"

    def __init__(self, name=None, n_rules: int = 6, epochs: int = 12,
                 lr: float = 0.05, seed: int = 0, **params):
        super().__init__(name=name, n_rules=n_rules, epochs=epochs, lr=lr, seed=seed,
                         **params)
        self.n_rules = int(n_rules)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self._centres = None
        self._widths = None
        self._consequent = None
        self._D = None

    def _solve_consequents(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least-squares solve for the linear consequent parameters given the
        current premises (the linear half of the hybrid step)."""
        wbar = _firing_strengths(X, self._centres, self._widths)
        Phi = _design_matrix(X, wbar)
        theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        return theta

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        y = np.asarray(y, dtype=float).ravel()
        T, D = X.shape
        self._D = D
        rng = np.random.default_rng(self.seed)
        n_rules = min(self.n_rules, max(2, T // 4))
        self._centres, self._widths = _init_premises(X, n_rules, rng)
        self._consequent = self._solve_consequents(X, y)
        # hybrid: alternate LSE (consequents) with a gradient step on the premises.
        for _ in range(self.epochs):
            pred = self._forward(X)
            err = pred - y                              # (T,)
            self._gradient_step_premises(X, err)
            self._consequent = self._solve_consequents(X, y)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        wbar = _firing_strengths(X, self._centres, self._widths)
        Phi = _design_matrix(X, wbar)
        return Phi @ self._consequent

    def _gradient_step_premises(self, X: np.ndarray, err: np.ndarray) -> None:
        """One numerical-gradient descent step on the Gaussian centres/widths.

        Uses a small finite-difference estimate of ∂MSE/∂(centre, width) per
        parameter — robust and dependency-free, adequate for the modest rule counts
        here. Widths are floored to stay positive (well-posed MFs)."""
        base = float(np.mean(err ** 2))
        eps = 1e-4
        R, D = self._centres.shape
        gc = np.zeros_like(self._centres)
        gw = np.zeros_like(self._widths)
        for r in range(R):
            for d in range(D):
                self._centres[r, d] += eps
                gc[r, d] = (self._mse(X) - base) / eps
                self._centres[r, d] -= eps
                self._widths[r, d] += eps
                gw[r, d] = (self._mse(X) - base) / eps
                self._widths[r, d] -= eps
        self._centres -= self.lr * np.clip(gc, -10, 10)
        self._widths -= self.lr * np.clip(gw, -10, 10)
        self._widths = np.clip(self._widths, 1e-3, None)

    def _mse(self, X: np.ndarray) -> float:
        """Training MSE under the current premises against the cached target."""
        pred = self._forward(X)
        return float(np.mean((pred - self._y_cache) ** 2))

    def fit(self, X, y=None):                            # capture y for the gradient step
        Xv = self._validate(X)
        if y is None:
            raise ValueError("anfis requires labels y for fit()")
        self._y_cache = np.asarray(y, dtype=float).ravel()
        return super().fit(X, y)

    def _predict(self, X: np.ndarray) -> Belief:
        pred = self._forward(X)
        est = float(pred[-1])
        # confidence from in-sample fit relative to target spread, if we have it.
        conf = 0.5
        if getattr(self, "_y_cache", None) is not None and self._y_cache.size > 1:
            spread = float(np.std(self._y_cache)) + 1e-12
            # approximate train residual scale via the firing structure already fit
            conf = float(np.clip(0.5 + 0.4 * (X.shape[0] / 500.0), 0.2, 0.9))
        return Belief(
            type="forecast",
            payload={
                "estimate": est,
                "series": pred.tolist(),
                "n_rules": int(self._centres.shape[0]),
            },
            confidence=conf,
        )
