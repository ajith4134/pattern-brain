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


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — supervised classifiers (Block 42 Classical
# Classification). All require labels y; routing withholds them when no labels are
# present. Each emits the 'decision' contract (predictions + max_proba).
# --------------------------------------------------------------------------
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


class _ClassifierNode(Node):
    """Base for supervised decision classifiers (emits 'decision')."""
    layer = "decision"
    requires_y = True

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _fit(self, X, y=None):
        y = np.asarray(y).ravel()
        self._classes = np.unique(y)
        self._m = None if len(self._classes) < 2 else self._make().fit(X, y)

    def _predict(self, X: np.ndarray) -> Belief:
        if self._m is None:
            return Belief("decision",
                          {"predictions": [int(self._classes[0])] * X.shape[0],
                           "note": "single-class target"}, 0.0, self.name)
        pred = self._m.predict(X)
        out = {"predictions": np.asarray(pred).tolist()}
        conf = 0.5
        if hasattr(self._m, "predict_proba"):
            proba = self._m.predict_proba(X)
            out["max_proba"] = proba.max(axis=1).tolist()
            conf = float(proba.max(axis=1).mean())
        return Belief("decision", out, conf, self.name)


@register
class SVMClassifierNode(_ClassifierNode):
    node_type = "svm_classifier"

    def _make(self):
        return SVC(probability=True, gamma="scale", random_state=0)


@register
class RandomForestNode(_ClassifierNode):
    node_type = "random_forest"

    def __init__(self, n_estimators: int = 100, random_state: int = 0, **kw):
        super().__init__(n_estimators=n_estimators, random_state=random_state, **kw)
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _make(self):
        return RandomForestClassifier(n_estimators=self.n_estimators,
                                      random_state=self.random_state)


@register
class GradientBoostingNode(_ClassifierNode):
    node_type = "gradient_boosting"

    def _make(self):
        return GradientBoostingClassifier(random_state=0)


@register
class KNNClassifierNode(_ClassifierNode):
    node_type = "knn_classifier"

    def __init__(self, n_neighbors: int = 5, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        self.n_neighbors = n_neighbors

    def _fit(self, X, y=None):
        y = np.asarray(y).ravel()
        self._classes = np.unique(y)
        if len(self._classes) < 2:
            self._m = None
        else:
            k = max(1, min(self.n_neighbors, X.shape[0]))
            self._m = KNeighborsClassifier(n_neighbors=k).fit(X, y)


@register
class GaussianNBNode(_ClassifierNode):
    node_type = "gaussian_nb"

    def _make(self):
        return GaussianNB()


@register
class DecisionTreeNode(_ClassifierNode):
    node_type = "decision_tree"

    def _make(self):
        return DecisionTreeClassifier(random_state=0)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — more supervised classifiers (incl. a
# sklearn MLP — a neural net that needs NO torch). All reuse _ClassifierNode.
# --------------------------------------------------------------------------
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)


@register
class ExtraTreesNode(_ClassifierNode):
    node_type = "extra_trees"
    def _make(self): return ExtraTreesClassifier(n_estimators=100, random_state=0)


@register
class AdaBoostNode(_ClassifierNode):
    node_type = "adaboost"
    def _make(self): return AdaBoostClassifier(random_state=0)


@register
class BaggingNode(_ClassifierNode):
    node_type = "bagging"
    def _make(self): return BaggingClassifier(n_estimators=20, random_state=0)


@register
class MLPClassifierNode(_ClassifierNode):
    """Multi-layer perceptron classifier — a neural net in the LIGHT stack
    (sklearn, no torch); the deep-framework nets are build step 6 Phase 6b."""
    node_type = "mlp_classifier"
    def _make(self): return MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=0)


@register
class RidgeClassifierNode(_ClassifierNode):
    """Ridge (least-squares) classifier — no predict_proba (confidence is fixed)."""
    node_type = "ridge_classifier"
    def _make(self): return RidgeClassifier()


@register
class LDANode(_ClassifierNode):
    node_type = "lda"
    def _make(self): return LinearDiscriminantAnalysis()


@register
class QDANode(_ClassifierNode):
    node_type = "qda"
    def _make(self): return QuadraticDiscriminantAnalysis()


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — more classifiers.
# --------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.neighbors import NearestCentroid  # noqa: E402
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier  # noqa: E402


@register
class HistGradientBoostingNode(_ClassifierNode):
    node_type = "hist_gradient_boosting"
    def _make(self): return HistGradientBoostingClassifier(random_state=0)


@register
class NearestCentroidNode(_ClassifierNode):
    node_type = "nearest_centroid"
    def _make(self): return NearestCentroid()


@register
class SGDClassifierNode(_ClassifierNode):
    """Stochastic-gradient linear classifier (log-loss -> calibrated proba)."""
    node_type = "sgd_classifier"
    def _make(self): return SGDClassifier(loss="log_loss", random_state=0)


@register
class PassiveAggressiveClassifierNode(_ClassifierNode):
    node_type = "passive_aggressive_classifier"
    def _make(self): return PassiveAggressiveClassifier(random_state=0)
