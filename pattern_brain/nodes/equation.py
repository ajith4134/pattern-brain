"""Equation layer (Block 18, layer 6): discover closed-form rules from data.

Block 2 family 13 / Block 18 cat 17 ("critical for your idea... discover
equations from data"). The symbolic-regression node here is a lightweight
library-search SR; PySR/genetic-programming is the heavy upgrade (PLAN.md §5 /
Feature 2) and a future node, not required for step 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class LinearRegressionNode(Node):
    """Closed-form linear law for feature 0 against the time index."""
    layer = "equation"
    node_type = "linear_regression"

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        t = np.arange(T).reshape(-1, 1)
        y = X[:, 0]
        m = LinearRegression().fit(t, y)
        r2 = float(max(0.0, m.score(t, y)))
        return Belief("equation",
                      {"expression": f"x0 = {m.coef_[0]:.4g}*t + {m.intercept_:.4g}",
                       "coef": m.coef_.tolist(), "intercept": float(m.intercept_), "r2": r2},
                      r2, self.name)


@register
class PolynomialRegressionNode(Node):
    """Polynomial law for feature 0 against the time index."""
    layer = "equation"
    node_type = "polynomial_regression"

    def __init__(self, degree: int = 2, **kw):
        super().__init__(degree=degree, **kw)
        self.degree = degree

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        t = np.arange(T).reshape(-1, 1)
        y = X[:, 0]
        deg = max(1, min(self.degree, max(1, T - 1)))
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(t, y)
        r2 = float(max(0.0, m.score(t, y)))
        coef = m.named_steps["linearregression"].coef_
        return Belief("equation", {"degree": deg, "coef": coef.tolist(), "r2": r2},
                      r2, self.name)


@register
class SymbolicRegressionNode(Node):
    """Lightweight symbolic regression: fit a small library of candidate closed
    forms to (t -> feature 0) and select the best by BIC. Returns the chosen
    symbolic expression. A real, if small, SR — PySR/GP is the heavy upgrade."""
    layer = "equation"
    node_type = "symbolic_regression"

    @staticmethod
    def _dominant_omega(y: np.ndarray) -> float:
        """Estimate the dominant angular frequency via the real FFT, so the
        sinusoidal candidate can match a sine of any frequency, not just 1.0."""
        sig = y - y.mean()
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig))
        if len(spec) > 1 and spec[1:].sum() > 0:
            k = int(np.argmax(spec[1:]) + 1)
            return 2 * np.pi * float(freqs[k])
        return 1.0

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        t = np.arange(1, T + 1).astype(float)
        y = X[:, 0]
        n = len(y)
        w = self._dominant_omega(y)
        library = {
            "linear":    (lambda t: [t],                 "a*t + b"),
            "quadratic": (lambda t: [t, t ** 2],         "a*t + b*t^2 + c"),
            "sqrt":      (lambda t: [np.sqrt(t)],        "a*sqrt(t) + b"),
            "log":       (lambda t: [np.log(t)],         "a*log(t) + b"),
            "sin":       (lambda t: [np.sin(w * t), np.cos(w * t)],
                          f"a*sin({w:.3g}*t) + b*cos({w:.3g}*t) + c"),
        }
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        best = None
        for name, (feat, form) in library.items():
            F = np.column_stack(feat(t) + [np.ones_like(t)])
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)
            resid = y - F @ coef
            ss_res = float(np.sum(resid ** 2))
            r2 = 1 - ss_res / ss_tot
            k = F.shape[1]
            bic = n * np.log(ss_res / n + 1e-12) + k * np.log(n)
            if best is None or bic < best["bic"]:
                best = {"form": name, "expression": form, "coef": coef.tolist(),
                        "r2": float(max(0.0, r2)), "bic": float(bic)}
        return Belief("equation", best, float(max(0.0, min(1.0, best["r2"]))), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — regularized / robust linear laws (Block 42
# Classical Regression). Each fits feature 0 against the time index and emits the
# 'equation' contract (coef + r2), like LinearRegressionNode.
# --------------------------------------------------------------------------
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, TheilSenRegressor  # noqa: E402


class _LinearLawNode(Node):
    """Base: fit a linear model of feature 0 on the time index, emit 'equation'."""
    layer = "equation"

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        t = np.arange(T).reshape(-1, 1).astype(float)
        y = X[:, 0]
        m = self._make().fit(t, y)
        r2 = float(max(0.0, m.score(t, y)))
        coef = np.ravel(getattr(m, "coef_", [0.0]))
        intercept = float(np.ravel(getattr(m, "intercept_", [0.0]))[0])
        return Belief("equation",
                      {"coef": coef.tolist(), "intercept": intercept, "r2": r2,
                       "form": self.node_type},
                      r2, self.name)


@register
class RidgeNode(_LinearLawNode):
    node_type = "ridge_regression"

    def __init__(self, alpha: float = 1.0, **kw):
        super().__init__(alpha=alpha, **kw)
        self.alpha = alpha

    def _make(self):
        return Ridge(alpha=self.alpha)


@register
class LassoNode(_LinearLawNode):
    node_type = "lasso_regression"

    def __init__(self, alpha: float = 0.1, **kw):
        super().__init__(alpha=alpha, **kw)
        self.alpha = alpha

    def _make(self):
        return Lasso(alpha=self.alpha, max_iter=5000)


@register
class ElasticNetNode(_LinearLawNode):
    node_type = "elasticnet_regression"

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, **kw):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **kw)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _make(self):
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)


@register
class HuberNode(_LinearLawNode):
    """Robust (outlier-resistant) linear law."""
    node_type = "huber_regression"

    def _make(self):
        return HuberRegressor(max_iter=500)


@register
class TheilSenNode(_LinearLawNode):
    """Robust non-parametric (median-of-slopes) linear law."""
    node_type = "theil_sen_regression"

    def _make(self):
        return TheilSenRegressor(random_state=0, max_subpopulation=2000)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — more linear laws with closed-form coef_.
# All reuse the _LinearLawNode base (feature 0 vs the time index).
# --------------------------------------------------------------------------
from sklearn.linear_model import Lars, LassoLars, OrthogonalMatchingPursuit, ARDRegression  # noqa: E402


@register
class LarsNode(_LinearLawNode):
    node_type = "lars_regression"
    def _make(self): return Lars()


@register
class LassoLarsNode(_LinearLawNode):
    node_type = "lasso_lars_regression"
    def _make(self): return LassoLars(alpha=0.01)


@register
class OMPNode(_LinearLawNode):
    node_type = "omp_regression"
    def _make(self): return OrthogonalMatchingPursuit()


@register
class ARDNode(_LinearLawNode):
    """Automatic Relevance Determination (sparse Bayesian) linear law."""
    node_type = "ard_regression"
    def _make(self): return ARDRegression()


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — quantile + margin linear laws.
# --------------------------------------------------------------------------
from sklearn.linear_model import QuantileRegressor  # noqa: E402
from sklearn.svm import LinearSVR  # noqa: E402


@register
class QuantileRegressionNode(_LinearLawNode):
    """Median (quantile) regression — robust to asymmetric noise."""
    node_type = "quantile_regression"
    def _make(self): return QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")


@register
class LinearSVRNode(_LinearLawNode):
    """Linear support-vector regression (epsilon-insensitive linear law)."""
    node_type = "linear_svr"
    def _make(self): return LinearSVR(max_iter=5000)


# ==========================================================================
# Phase 7g — equation / symbolic DISCOVERY (PLAN.md §11). Pure-numpy engines
# (idea #1 from the 7f-batch-2 ideas block): SINDy (sparse identification of a
# governing ODE) + genetic-programming symbolic regression (evolve a closed
# form, not a fixed template). Both emit the 'equation' contract (coef + r2),
# zero stock coupling (Rule 23). These are the fast BANK-NODE form of equation
# discovery; AlgorithmEvolver (Feature 2) is the evolutionary, Evaluator-gated
# form — complementary, same family. Heavy deps deferred behind §0b: PySR
# (needs Julia) and AI-Feynman — added only when proven needed.
# ==========================================================================
import random as _random  # noqa: E402


@register
class SINDyNode(Node):
    """SINDy — Sparse Identification of Nonlinear Dynamics (Brunton, Proctor &
    Kutz, PNAS 2016): regress the time-derivative of the (standardized) state onto
    a library of candidate functions [1, x, x^2, x^3, sin x, cos x], then
    SEQUENTIALLY THRESHOLD to a sparse active set (STLSQ) — discovering the
    governing law dx/dt = Σ ξ_i·θ_i(x) rather than merely fitting a curve. Sparsity
    is the point: a few interpretable terms, not a dense polynomial."""
    layer = "equation"
    node_type = "sindy_regression"

    def __init__(self, threshold: float = 0.05, iters: int = 10, **kw):
        super().__init__(threshold=threshold, iters=iters, **kw)
        self.threshold, self.iters = threshold, iters

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if len(x) < 6:
            return Belief("equation", {"expression": "dx/dt = 0", "coef": [0.0],
                                       "terms": ["1"], "n_active": 0, "r2": 0.0}, 0.0, self.name)
        xs = (x - x.mean()) / (x.std() + 1e-9)
        dx = np.gradient(xs)
        terms = [("1", np.ones_like(xs)), ("x", xs), ("x^2", xs ** 2), ("x^3", xs ** 3),
                 ("sin(x)", np.sin(xs)), ("cos(x)", np.cos(xs))]
        names = [n for n, _ in terms]
        Theta = np.column_stack([f for _, f in terms])
        xi = np.linalg.lstsq(Theta, dx, rcond=None)[0]
        for _ in range(self.iters):                       # STLSQ
            small = np.abs(xi) < self.threshold
            xi[small] = 0.0
            active = ~small
            if not active.any():
                break
            xi[active] = np.linalg.lstsq(Theta[:, active], dx, rcond=None)[0]
        pred = Theta @ xi
        ss_res = float(np.sum((dx - pred) ** 2))
        ss_tot = float(np.sum((dx - dx.mean()) ** 2)) + 1e-12
        r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
        expr = " + ".join(f"{c:.3g}*{n}" for c, n in zip(xi, names) if abs(c) > 0) or "0"
        return Belief("equation", {"expression": "dx/dt = " + expr, "coef": xi.tolist(),
                                   "terms": names, "n_active": int((np.abs(xi) > 0).sum()),
                                   "r2": r2},
                      r2, self.name)


# ---- genetic-programming symbolic regression (vectorized expression trees) ----
_GP_BIN = ["+", "-", "*", "/"]
_GP_UN = ["sin", "cos", "tanh", "abs", "sq"]


def _gp_random(rng, depth):
    """tree = ('t',) | ('const', v) | (binop, l, r) | (unop, child)."""
    if depth <= 0 or rng.random() < 0.3:
        return ("t",) if rng.random() < 0.6 else ("const", round(rng.uniform(-2, 2), 3))
    if rng.random() < 0.6:
        return (rng.choice(_GP_BIN), _gp_random(rng, depth - 1), _gp_random(rng, depth - 1))
    return (rng.choice(_GP_UN), _gp_random(rng, depth - 1))


def _gp_eval(tree, t):
    """Vectorized evaluation over the numpy time vector ``t`` (protected ops)."""
    op = tree[0]
    if op == "t":
        return t
    if op == "const":
        return np.full_like(t, float(tree[1]))
    with np.errstate(all="ignore"):
        if op in _GP_UN:
            a = _gp_eval(tree[1], t)
            if op == "sin":  return np.sin(a)
            if op == "cos":  return np.cos(a)
            if op == "tanh": return np.tanh(a)
            if op == "abs":  return np.abs(a)
            return a * a                                  # sq
        a, b = _gp_eval(tree[1], t), _gp_eval(tree[2], t)
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        safe = np.where(np.abs(b) > 1e-9, b, 1.0)         # protected division
        return np.where(np.abs(b) > 1e-9, a / safe, 1.0)


def _gp_str(tree):
    op = tree[0]
    if op == "t":     return "t"
    if op == "const": return f"{tree[1]:.3g}"
    if op == "sq":    return f"({_gp_str(tree[1])})^2"
    if op in _GP_UN:  return f"{op}({_gp_str(tree[1])})"
    return f"({_gp_str(tree[1])} {op} {_gp_str(tree[2])})"


def _gp_nodes(tree, acc=None, path=()):
    acc = [] if acc is None else acc
    acc.append((path, tree))
    if tree[0] in _GP_BIN + _GP_UN:
        for i, ch in enumerate(tree[1:], start=1):
            _gp_nodes(ch, acc, path + (i,))
    return acc


def _gp_replace(tree, path, new):
    if not path:
        return new
    tree = list(tree)
    tree[path[0]] = _gp_replace(tree[path[0]], path[1:], new)
    return tuple(tree)


@register
class GeneticSymbolicRegressionNode(Node):
    """Genetic-programming symbolic regression: evolve a population of expression
    trees over the normalized time index (tournament selection + subtree
    crossover/mutation, parsimony-penalized), then fit the fittest tree φ(t) as
    y ≈ a·φ(t) + b by least squares. Discovers a NONLINEAR closed form rather than
    selecting from a fixed library — the dependency-free counterpart to PySR/gplearn."""
    layer = "equation"
    node_type = "genetic_symbolic_regression"

    def __init__(self, population: int = 40, generations: int = 14, max_depth: int = 3,
                 seed: int = 0, **kw):
        super().__init__(population=population, generations=generations,
                         max_depth=max_depth, seed=seed, **kw)
        self.population, self.generations = population, generations
        self.max_depth, self.seed = max_depth, seed

    @staticmethod
    def _score_tree(tree, t, y, ss_tot):
        """Least-squares fit a·φ(t)+b; return (parsimony-penalized fitness, r2, coef)."""
        phi = _gp_eval(tree, t)
        if not np.all(np.isfinite(phi)) or np.std(phi) < 1e-9:
            return -1e9, 0.0, [0.0, float(y.mean())]
        F = np.column_stack([phi, np.ones_like(phi)])
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        r2 = 1.0 - float(np.sum((y - F @ coef) ** 2)) / ss_tot
        size = len(_gp_nodes(tree))
        return r2 - 0.005 * size, float(max(0.0, r2)), coef.tolist()

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        y = X[:, 0].astype(float)
        if T < 6:
            return Belief("equation", {"expression": "a*t+b", "coef": [0.0, float(y.mean())],
                                       "r2": 0.0, "tree": "t", "complexity": 1}, 0.0, self.name)
        t = np.linspace(0.0, 1.0, T)
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        rng = _random.Random(self.seed)
        pop = [_gp_random(rng, self.max_depth) for _ in range(self.population)]
        best = None                                        # (fitness, r2, coef, tree)
        for _ in range(self.generations):
            scored = []
            for tr in pop:
                fit, r2, coef = self._score_tree(tr, t, y, ss_tot)
                scored.append((fit, r2, coef, tr))
            scored.sort(key=lambda z: z[0], reverse=True)
            if best is None or scored[0][0] > best[0]:
                best = scored[0]
            newpop = [scored[0][3], scored[1][3]]          # elitism
            while len(newpop) < self.population:
                a = max(rng.sample(scored, min(3, len(scored))), key=lambda z: z[0])[3]
                b = max(rng.sample(scored, min(3, len(scored))), key=lambda z: z[0])[3]
                pa, _ = rng.choice(_gp_nodes(a))
                _, sub = rng.choice(_gp_nodes(b))
                child = _gp_replace(a, pa, sub)            # crossover
                if rng.random() < 0.3:                     # mutation
                    pm, _ = rng.choice(_gp_nodes(child))
                    child = _gp_replace(child, pm, _gp_random(rng, 2))
                newpop.append(child)
            pop = newpop
        _, r2, coef, tree = best
        expr = f"{coef[0]:.3g}*[{_gp_str(tree)}] + {coef[1]:.3g}"
        return Belief("equation", {"expression": expr, "coef": coef, "r2": r2,
                                   "tree": _gp_str(tree), "complexity": len(_gp_nodes(tree))},
                      r2, self.name)
