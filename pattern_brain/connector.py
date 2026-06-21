"""Connector Intelligence v0 — execute ONE hardcoded pathway end-to-end
(build-tracker step 2; PLAN.md §6 / Block 16 Layer-2).

This is a **Walking Skeleton** (Cockburn) / **Tracer Bullet** (Hunt & Thomas):
prove the whole architecture end-to-end on the smallest possible slice *before*
adding any routing intelligence. There is deliberately **no learned routing
here** — the pathway is a fixed, ordered list of nodes. Learned routing
(MoE/LLM-tool-calling/GNN) is build-tracker step 4.

How data + beliefs flow (the design decision behind v0)
-------------------------------------------------------
Every node consumes a generic ``(T, D)`` array and *emits* a :class:`Belief`
(the shared "voltage/signal standard", Block 5/16). A pathway therefore threads
a single **signal bus** — the working ``(T, D)`` array — through its nodes:

* a **transformer** node (``is_transformer``, e.g. ``difference``) rewrites the
  bus for everything downstream;
* a **non-transformer** node (``hdbscan``, ``gaussian_hmm``, ``threshold_policy``)
  taps the current bus, emits a Belief, and passes the bus through unchanged.

Every node's Belief is recorded, in order, into a JSON-serializable belief
stream (the auditable trail — precursor to the §8 dashboard "Belief-space
stream"). This needs **no per-pair adapters** (which §6 explicitly avoids) and
respects the one common node interface exactly.

Domain-agnostic by construction (PLAN.md §0 / Rule 23): a Connector knows only
about node types and a generic ``(T, D)`` bus. Nothing here references candles,
order books, or any stock-specific shape.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .belief import Belief
from .node import Node
from .registry import create

PathwaySpec = Sequence[Union[str, Node]]


@dataclass
class Hop:
    """One node's execution inside a pathway run."""
    step: int
    node_type: str
    name: str
    layer: str
    belief: Belief
    transformed: bool      # did this hop rewrite the signal bus for downstream nodes?
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "node_type": self.node_type,
            "name": self.name,
            "layer": self.layer,
            "transformed": self.transformed,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "belief": self.belief.to_dict(),
        }


@dataclass
class PathwayResult:
    """The full, auditable result of running one pathway end-to-end."""
    pathway: List[str]                 # node_types, in execution order
    hops: List[Hop] = field(default_factory=list)
    output: Optional[Belief] = None    # terminal node's belief
    elapsed_ms: float = 0.0

    def beliefs(self) -> List[Belief]:
        """The ordered belief stream — one Belief per node."""
        return [h.belief for h in self.hops]

    @property
    def decision(self) -> Optional[Belief]:
        """The last decision-typed belief on the path, if any (generic helper)."""
        for h in reversed(self.hops):
            if h.belief.type == "decision":
                return h.belief
        return None

    def to_dict(self) -> Dict[str, Any]:
        """A single JSON-serializable trace of the whole run — this is exactly
        what the §8 dashboard's belief-stream / living-graph views consume."""
        return {
            "pathway": list(self.pathway),
            "elapsed_ms": round(self.elapsed_ms, 4),
            "hops": [h.to_dict() for h in self.hops],
            "output": self.output.to_dict() if self.output is not None else None,
        }


class Connector:
    """Connector Intelligence v0: run a single, fixed, ordered pathway.

    Parameters
    ----------
    pathway : sequence of (str | Node)
        Node types (looked up in the registry) or ready Node instances, in the
        order they execute. Strings keep the pathway a pure, serializable
        ``list[str]`` spec — the unit Feature 3 will later attach reputation to.
    name : str
        A label for this connector (audit/telemetry).
    """

    def __init__(self, pathway: PathwaySpec, name: str = "connector-v0") -> None:
        if not pathway:
            raise ValueError("pathway must contain at least one node")
        self.name = name
        self._nodes: List[Node] = [n if isinstance(n, Node) else create(n) for n in pathway]
        self.pathway: List[str] = [n.node_type for n in self._nodes]

    def iter_run(self, X: Any, y: Optional[Any] = None):
        """Thread ``X`` (the signal bus) through the nodes, yielding each
        :class:`Hop` the moment it completes. This is the streaming source the
        §8 dashboard's WebSocket consumes to animate belief-flow live; `run()`
        below is just this generator collected into a result."""
        bus = X
        for i, node in enumerate(self._nodes):
            t0 = time.perf_counter()
            belief = node.process(bus, y) if node.requires_y else node.process(bus)
            transformed = False
            if node.is_transformer:
                bus = node.transform(bus)   # already fitted by process(); rewrites bus
                transformed = True
            elapsed = (time.perf_counter() - t0) * 1000.0
            yield Hop(i, node.node_type, node.name, node.layer,
                      belief, transformed, elapsed)

    def run(self, X: Any, y: Optional[Any] = None) -> PathwayResult:
        """Thread ``X`` (the signal bus) through every node in order, recording
        each node's Belief into the stream and rewriting the bus at transformer
        nodes. Returns the full auditable :class:`PathwayResult`."""
        t_start = time.perf_counter()
        hops: List[Hop] = list(self.iter_run(X, y))
        total = (time.perf_counter() - t_start) * 1000.0
        return PathwayResult(
            pathway=list(self.pathway),
            hops=hops,
            output=hops[-1].belief if hops else None,
            elapsed_ms=total,
        )

    def __repr__(self) -> str:
        return f"<Connector {self.name!r} pathway={'->'.join(self.pathway)}>"


# The hardcoded pathway for the v0 walking skeleton (PLAN.md build-tracker step 2).
# Spans four functional layers (signal -> pattern -> sequence -> decision), built
# only from node types delivered in step 1. `difference` is a transformer (rewrites
# the bus into a generic returns-like series); the rest tap that bus and emit
# beliefs, ending in a discrete decision. Generic data only — no domain meaning.
DEFAULT_PATHWAY: List[str] = ["difference", "hdbscan", "gaussian_hmm", "threshold_policy"]


def default_connector() -> Connector:
    """The step-2 walking-skeleton connector over :data:`DEFAULT_PATHWAY`."""
    return Connector(DEFAULT_PATHWAY, name="walking-skeleton")
