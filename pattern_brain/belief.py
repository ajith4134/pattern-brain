"""The Universal Belief Space unit (DISCUSSION_NOTES.md Block 5 / PLAN.md §6).

Every node, whatever its internals, emits its output as a `Belief` — a small,
auditable, typed record with an explicit confidence. This is the shared
"voltage/signal standard" (Block 16) that lets heterogeneous nodes feed each
other without per-pair adapters.

Domain-agnostic by construction (PLAN.md §0 / Rule 23): a Belief knows nothing
about candles, order books, or any stock-specific shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

SCHEMA_VERSION = "0.1"


@dataclass
class Belief:
    """A single typed assertion produced by a node.

    Attributes
    ----------
    type : str
        What kind of belief this is, e.g. "forecast", "regime", "cluster",
        "anomaly", "equation", "decision", "action", "denoised", "signal",
        "spectral", "density", "next_state".
    payload : dict
        Type-specific content. Arrays are stored as plain Python lists so the
        whole Belief stays JSON-serializable.
    confidence : float
        The node's own confidence in this belief, clamped to [0, 1].
        Confidence is a first-class field (Block 11 — "The Synthesist" states
        confidence explicitly), never an afterthought.
    source : str
        The name of the node that produced this belief.
    schema_version : str
        Interlingua version (Block 7g). A minimal version field today; the full
        versioning/drift scheme is build-tracker step 5.
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "unknown"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Confidence is always a clean probability-like scalar, regardless of the
        # heuristic a node used to compute it.
        try:
            c = float(self.confidence)
        except (TypeError, ValueError):
            c = 0.0
        if c != c:  # NaN guard
            c = 0.0
        self.confidence = max(0.0, min(1.0, c))
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("Belief.type must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:  # compact, readable audit line
        keys = ",".join(sorted(self.payload)[:4])
        return (
            f"Belief(type={self.type!r}, conf={self.confidence:.3f}, "
            f"source={self.source!r}, payload[{keys}])"
        )
