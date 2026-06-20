"""Decision layer (Block 18, layer 7): turn structure into a discrete decision.

Decisions are emitted in generic terms ("buy"/"sell"/"hold" as abstract
action labels on the underlying signal); they carry no order-execution or
stock-specific semantics — that mapping belongs to a downstream adapter.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class ThresholdPolicyNode(Node):
    """Z-score threshold policy on the cross-feature mean's last value."""
    layer = "decision"
    node_type = "threshold_policy"

    def __init__(self, k: float = 1.0, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        s = X.mean(axis=1)
        mu, sd = s.mean(), s.std() + 1e-9
        z = float((s[-1] - mu) / sd)
        action = "buy" if z > self.k else ("sell" if z < -self.k else "hold")
        return Belief("decision", {"action": action, "z": z},
                      float(np.tanh(abs(z))), self.name)


@register
class SignVoteNode(Node):
    """Majority-sign vote over recent first-differences."""
    layer = "decision"
    node_type = "sign_vote"

    def _predict(self, X: np.ndarray) -> Belief:
        d = np.diff(X.mean(axis=1))
        if len(d) == 0:
            return Belief("decision", {"action": "hold", "vote": 0.0}, 0.0, self.name)
        vote = float(np.sign(d).mean())
        action = "buy" if vote > 0 else ("sell" if vote < 0 else "hold")
        return Belief("decision", {"action": action, "vote": vote}, abs(vote), self.name)


@register
class LogisticRegressionNode(Node):
    """Supervised decision: logistic regression. Requires labels y at fit()."""
    layer = "decision"
    node_type = "logistic_regression"
    requires_y = True

    def _fit(self, X, y=None):
        y = np.asarray(y).ravel()
        self._classes = np.unique(y)
        if len(self._classes) < 2:
            self._m = None  # degenerate: single-class target
        else:
            self._m = LogisticRegression(max_iter=300).fit(X, y)

    def _predict(self, X: np.ndarray) -> Belief:
        if self._m is None:
            return Belief("decision",
                          {"predictions": [int(self._classes[0])] * X.shape[0],
                           "note": "single-class target"}, 0.0, self.name)
        proba = self._m.predict_proba(X)
        pred = self._m.predict(X)
        return Belief("decision",
                      {"predictions": pred.tolist(),
                       "max_proba": proba.max(axis=1).tolist()},
                      float(proba.max(axis=1).mean()), self.name)
