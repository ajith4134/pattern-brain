"""The common node interface (DISCUSSION_NOTES.md Block 35 / PLAN.md §6).

Every model in the bank — regardless of what is inside it — exposes the same
small surface, modelled on scikit-learn's Estimator API (`fit`/`predict`/
`transform`) plus a single belief-emitting `process()` entry. This is the literal
mechanical realization of PLAN.md §0's principle: "the data bends to the system."

Language policy (PLAN.md §0b): this base class is the contract a node hides
behind. A future C/C++ node bound via pybind11/Cython is a drop-in here — the
Connector calls it identically. The interface is what makes the project polyglot
without touching the Connector, the belief format, or the graph.

Domain-agnostic by construction (Rule 23): a node consumes a sequence of
D-dimensional vectors, shape (T, D), and emits Belief objects. Nothing here
knows about candles, order books, or any stock-specific shape — stock data is
converted into (T, D) inputs by a separate adapter (build-tracker step 3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from .belief import Belief


class Node(ABC):
    """Abstract base for every model-bank node.

    Subclasses set the class attributes (`layer`, `node_type`, ...) and implement
    `_predict`. They may optionally override `_fit` (stateful/learned nodes) and
    `_transform` (signal/noise nodes that output a cleaned/derived sequence).
    """

    # --- identity / metadata (Model Genome schema, Block 18) ---
    layer: str = "abstract"          # functional layer (Block 18's 8-layer ordering)
    node_type: str = "abstract"      # unique registry key
    requires_y: bool = False         # True for supervised nodes
    is_transformer: bool = False     # True if the node also outputs a (T, D') sequence

    def __init__(self, name: Optional[str] = None, **params: Any) -> None:
        self.name = name or self.node_type
        self.params = params
        self._fitted = False

    # ------------------------------------------------------------------ hooks
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Override for learned/stateful nodes. Default: nothing to learn."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> Belief:
        """Produce a Belief from a validated (T, D) array. Required."""

    def _transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.node_type} is not a transformer node")

    # ----------------------------------------------------------- public API
    def fit(self, X: Any, y: Optional[Any] = None) -> "Node":
        X = self._validate(X)
        if self.requires_y and y is None:
            raise ValueError(f"{self.node_type} requires labels y for fit()")
        if y is not None:
            y = np.asarray(y).ravel()
        self._fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: Any) -> Belief:
        X = self._validate(X)
        if not self._fitted:
            if self.requires_y:
                raise RuntimeError(f"{self.node_type}.predict() called before fit(X, y)")
            self.fit(X)  # transductive auto-fit for unsupervised nodes
        belief = self._predict(X)
        belief.source = self.name
        return belief

    def transform(self, X: Any) -> np.ndarray:
        if not self.is_transformer:
            raise NotImplementedError(f"{self.node_type} is not a transformer node")
        X = self._validate(X)
        if not self._fitted and not self.requires_y:
            self.fit(X)
        out = self._transform(X)
        return self._validate(out)

    def process(self, X: Any, y: Optional[Any] = None) -> Belief:
        """Universal one-call entry used by the Connector Intelligence: fit if
        needed, then emit a Belief."""
        if not self._fitted:
            self.fit(X, y)
        return self.predict(X)

    def metadata(self) -> Dict[str, Any]:
        """The node's Model Genome Database record (Block 18), generic."""
        return {
            "node_type": self.node_type,
            "layer": self.layer,
            "name": self.name,
            "requires_y": self.requires_y,
            "is_transformer": self.is_transformer,
            "params": dict(self.params),
            "fitted": self._fitted,
        }

    # ------------------------------------------------------------ validation
    @staticmethod
    def _validate(X: Any) -> np.ndarray:
        """Coerce input to a finite (T, D) float array. 1-D input becomes (T, 1)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"expected a (T, D) sequence, got shape {X.shape}")
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError(f"empty sequence: shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("input contains non-finite values (NaN/inf)")
        return X

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} layer={self.layer} type={self.node_type} fitted={self._fitted}>"
