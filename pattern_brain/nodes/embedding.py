"""Embedding layer — VISION P1 / Stage-1 self-supervised (IDEA-084).

Wraps :class:`pattern_brain.embedding.SelfSupervisedEncoder` as a bank node so the learned
universal latent is available to the registry, the StackedDAG, and the router exactly like any
other node. Transformer node: its ``transform`` emits the (T, latent_dim) shared latent that
downstream neurons read (the "embedding every neuron reads"); its ``predict`` emits a compact
Belief summarising the latent's activation. Domain-agnostic (Rule 23): operates on any (T, D).
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..embedding import SelfSupervisedEncoder
from ..node import Node
from ..registry import register


@register
class UniversalEmbeddingNode(Node):
    """Learned self-supervised universal market embedding (VISION P1, the curriculum's Stage 1)."""

    layer = "signal"
    node_type = "universal_embedding"
    is_transformer = True
    requires_y = False
    cost = "med"

    def __init__(self, name=None, latent_dim: int = 8, window: int = 16,
                 mask_frac: float = 0.3, contrastive_weight: float = 0.1,
                 epochs: int = 300, seed: int = 0, **params):
        super().__init__(name=name, latent_dim=latent_dim, window=window,
                         mask_frac=mask_frac, contrastive_weight=contrastive_weight,
                         epochs=epochs, seed=seed, **params)
        self._enc = SelfSupervisedEncoder(
            latent_dim=latent_dim, window=window, mask_frac=mask_frac,
            contrastive_weight=contrastive_weight, epochs=epochs, seed=seed)

    def _fit(self, X: np.ndarray, y=None) -> None:
        self._enc.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._enc.transform(X)

    def _predict(self, X: np.ndarray) -> Belief:
        Z = self._enc.transform(X)
        latest = Z[-1]
        # confidence = how "active" / non-degenerate the latent is (mean |tanh| in (0,1))
        conf = float(np.clip(np.mean(np.abs(latest)), 0.0, 1.0))
        r2 = float(self._enc.reconstruction_r2(X))
        return Belief(
            "embedding",
            payload={"latent": latest.tolist(),
                     "latent_dim": int(self._enc.latent_dim),
                     "reconstruction_r2": r2},
            confidence=conf,
            source=self.name,
        )
