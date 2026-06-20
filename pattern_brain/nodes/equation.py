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
