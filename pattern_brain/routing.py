"""Connector Intelligence v1 — LLM-tool-calling routing (build-tracker step 4;
PLAN.md §6 / Block 7c/16). The first *learned/dynamic* routing, replacing the
step-2 fixed pathway.

What changes from v0 (``connector.py``)
---------------------------------------
v0 ran a hardcoded ``list[str]`` pathway. v1 **decides the next node step by
step**: the model bank's nodes are presented as callable "tools", and a
:class:`Router` chooses which tool to call next given (a) the candidate nodes,
(b) the belief stream so far, and (c) the functional layers already visited —
then the Connector executes that node, feeds its Belief back, and repeats until a
decision node fires, the router stops, or ``max_hops`` is reached. The
dynamically *discovered* pathway is exactly the unit Feature 3 will later attach
reputation to.

Why LLM-tool-calling first (PLAN.md build-tracker step 4): of the three routing
candidates (MoE-gating, LLM-tool-calling, GNN), MoE and GNN both need to be
*trained* on logged routing outcomes that do not exist yet — a chicken-and-egg
problem. Tool-calling needs no training data and is immediately inspectable.

The LLM is a SWAPPABLE BOUNDARY, not a hard dependency
-----------------------------------------------------
Everything here runs offline and deterministically:

* :class:`HeuristicRouter` — a real, inspectable routing policy (advance through
  Block-18 layers toward a decision). The default; no API key, fully tested.
* :class:`LLMRouter` — depends only on one injected ``complete(...)`` callable
  (the LLM boundary). Tested with a *fake* completer so the whole tool-calling
  loop is proven without a network. A real Anthropic / Ollama client is added
  later as that one callable — consistent with §0b ("the implementation bends to
  fit the interface") and with steps 1-3 staying offline-testable.

Domain-agnostic (PLAN.md §0 / Rule 23): the router reasons over generic node
metadata + Beliefs only. Nothing here references candles/order books. The
Block-18 layer-adjacency constraint ("search within and across adjacent layers,
not an unconstrained complete graph") is enforced structurally by the Connector;
the router only *chooses* among the legal candidates it is offered.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .belief import Belief
from .connector import Hop, PathwayResult
from .node import Node
from .registry import default_bank

# Block-18 functional-layer ordering: signal -> noise -> pattern -> sequence ->
# probability -> equation -> decision -> rl. Routing moves forward through it.
LAYER_ORDER: List[str] = [
    "signal", "noise", "pattern", "sequence",
    "probability", "equation", "decision", "rl",
]
_LAYER_IX: Dict[str, int] = {l: i for i, l in enumerate(LAYER_ORDER)}
# A pathway ends when it fires a terminal-layer node. Both a discrete decision
# (buy/sell/hold) and an RL action are legitimate pathway endings (Block-18's
# worked examples end in either `...->PySR->PPO` or `...->threshold_policy`).
TERMINAL_LAYERS = {"decision", "rl"}


class RouterError(RuntimeError):
    """Raised when a router makes an illegal choice (no silent degradation)."""


# --------------------------------------------------------------------------- IO
@dataclass
class RoutingState:
    """Everything a router sees to pick the next node (generic — no domain shape)."""
    candidates: List[Dict[str, Any]]      # Model-Genome metadata of legal next nodes
    beliefs: List[Belief]                 # the stream produced so far, in order
    visited_layers: List[str]             # functional layers already used
    last_layer: Optional[str]             # layer of the most recent node, if any
    step: int
    max_hops: int

    def candidate_types(self) -> List[str]:
        return [c["node_type"] for c in self.candidates]


@dataclass
class RouterAction:
    """The router's decision for one hop: call a node, or stop."""
    node_type: Optional[str]              # None => stop
    reason: str = ""

    @property
    def is_stop(self) -> bool:
        return self.node_type is None


@dataclass
class RouterDecision:
    """An auditable record of one routing choice (for the dashboard/notes)."""
    step: int
    chosen: Optional[str]
    layer: Optional[str]
    reason: str
    candidates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "chosen": self.chosen, "layer": self.layer,
            "reason": self.reason, "candidates": list(self.candidates),
        }


@dataclass
class RoutedResult:
    """A routed run plus the per-hop routing rationale (richer than PathwayResult,
    which stays the dashboard/audit-compatible core)."""
    result: PathwayResult
    decisions: List[RouterDecision] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.result.to_dict()
        d["decisions"] = [x.to_dict() for x in self.decisions]
        return d


# ------------------------------------------------------------------ tool specs
def tool_spec(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a node's Model-Genome metadata into a JSON tool schema — the shape
    a real LLM tool-calling API (Anthropic/OpenAI/Ollama) consumes. Generic; no
    domain fields."""
    return {
        "name": meta["node_type"],
        "description": (
            f"A {meta['layer']}-layer node ({meta['node_type']}). "
            f"requires_y={meta['requires_y']}, is_transformer={meta['is_transformer']}. "
            "Call it to advance the analysis pathway one hop."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


def tool_specs(state: RoutingState) -> List[Dict[str, Any]]:
    """The candidate tools for this hop, plus a synthetic ``stop`` tool."""
    tools = [tool_spec(c) for c in state.candidates]
    tools.append({
        "name": "stop",
        "description": "End the pathway now and return the last belief produced.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    })
    return tools


# ---------------------------------------------------------------- the routers
class Router(ABC):
    """Picks the next node (or stop) given the current :class:`RoutingState`."""

    name: str = "router"

    @abstractmethod
    def choose(self, state: RoutingState) -> RouterAction: ...


class HeuristicRouter(Router):
    """Deterministic, offline default: advance through Block-18 layers toward a
    decision. Emulates what a tool-calling LLM should do, with no API.

    Policy: keep *advancing* into the next un-entered feature layer (the lowest
    non-terminal candidate strictly beyond the current layer), so it walks
    signal -> noise -> pattern -> ... one layer at a time. Only when no forward
    feature layer remains does it *terminate* by firing a terminal node,
    preferring a ``decision`` over an ``rl`` action. Stop only if nothing legal
    remains. This is what a competent tool-calling LLM ought to do, deterministic.
    """

    name = "heuristic"

    def choose(self, state: RoutingState) -> RouterAction:
        if not state.candidates:
            return RouterAction(None, reason="no legal candidates left")
        last_ix = _LAYER_IX.get(state.last_layer, -1) if state.last_layer else -1
        forward_feature = [c for c in state.candidates
                           if c["layer"] not in TERMINAL_LAYERS
                           and _LAYER_IX.get(c["layer"], 99) > last_ix]
        if forward_feature:
            pick = min(forward_feature, key=lambda c: (_LAYER_IX[c["layer"]], c["node_type"]))
            return RouterAction(pick["node_type"], reason=f"advance into {pick['layer']} layer")
        # no further feature layer -> terminate, preferring a decision over rl
        terminals = [c for c in state.candidates if c["layer"] in TERMINAL_LAYERS]
        if terminals and state.visited_layers:
            decisions = [c for c in terminals if c["layer"] == "decision"]
            pick = min(decisions or terminals, key=lambda c: c["node_type"])
            return RouterAction(pick["node_type"],
                                reason=f"terminate in {pick['layer']} after building features")
        # nothing forward and nothing to terminate with -> stop
        nonterminal = [c for c in state.candidates if c["layer"] not in TERMINAL_LAYERS]
        if nonterminal:
            pick = min(nonterminal, key=lambda c: (_LAYER_IX[c["layer"]], c["node_type"]))
            return RouterAction(pick["node_type"], reason=f"stay in {pick['layer']} (no forward layer)")
        return RouterAction(None, reason="no feature layer left and nothing to terminate with")


class LLMRouter(Router):
    """LLM-tool-calling router. Depends ONLY on an injected ``complete`` callable
    (the LLM boundary), so it runs offline against a fake in tests and against a
    real Anthropic/Ollama client in production — the callable is the only thing
    that changes.

    ``complete(system, user, tools) -> dict`` must return a tool choice as
    ``{"name": <one of the offered tool names>}``. An invalid name raises
    :class:`RouterError` (no silent degradation).
    """

    name = "llm"

    SYSTEM = (
        "You route a generic pattern-analysis pipeline. Each tool is a model node "
        "in one functional layer (signal, noise, pattern, sequence, probability, "
        "equation, decision, rl). Choose the next node to advance toward a final "
        "decision, moving forward through layers; call 'stop' to finish. Reason "
        "only about the generic node metadata and beliefs provided — the data is "
        "domain-independent."
    )

    def __init__(self, complete: Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]],
                 name: str = "llm") -> None:
        self._complete = complete
        self.name = name

    def choose(self, state: RoutingState) -> RouterAction:
        tools = tool_specs(state)
        legal = {t["name"] for t in tools}
        user = self._render_state(state)
        resp = self._complete(self.SYSTEM, user, tools)
        if not isinstance(resp, dict) or "name" not in resp:
            raise RouterError(f"router completer returned no tool choice: {resp!r}")
        name = resp["name"]
        if name not in legal:
            raise RouterError(
                f"router chose {name!r}, not among offered tools {sorted(legal)}")
        if name == "stop":
            return RouterAction(None, reason=resp.get("reason", "router chose stop"))
        return RouterAction(name, reason=resp.get("reason", "router tool-call"))

    @staticmethod
    def _render_state(state: RoutingState) -> str:
        beliefs = "; ".join(f"{b.source}:{b.type}({b.confidence:.2f})"
                            for b in state.beliefs) or "(none yet)"
        cands = ", ".join(f"{c['node_type']}[{c['layer']}]" for c in state.candidates)
        return (f"hop {state.step}/{state.max_hops}. visited layers: "
                f"{state.visited_layers or '[]'}. beliefs so far: {beliefs}. "
                f"available tools: {cands}, plus stop.")


# ----------------------------------------------------------- routed connector
class RoutedConnector:
    """Connector Intelligence v1: a router decides the pathway hop-by-hop.

    Reuses :class:`~pattern_brain.connector.Hop` / :class:`PathwayResult` so a
    routed run is consumed by the dashboard/audit exactly like a v0 run — one
    trace shape, not a second one that could drift.

    Parameters
    ----------
    router : Router
        The routing policy (e.g. :class:`HeuristicRouter`, :class:`LLMRouter`).
    bank : sequence of Node, optional
        The available nodes (the "tools"). Defaults to one of every registered
        node (:func:`~pattern_brain.registry.default_bank`).
    max_hops : int
        Hard cap on pathway length (prevents runaway routing).
    max_layer_jump : int
        Block-18 adjacency window: a candidate may be at most this many layers
        ahead of the last node's layer (so routing can't leap signal->rl). This
        is the structural form of "search within and across adjacent layers".
    """

    def __init__(self, router: Router, bank: Optional[Sequence[Node]] = None, *,
                 max_hops: int = 8, max_layer_jump: int = 2,
                 name: str = "routed-v1") -> None:
        self.router = router
        self.name = name
        self.max_hops = int(max_hops)
        self.max_layer_jump = int(max_layer_jump)
        nodes = list(bank) if bank is not None else default_bank()
        # de-dup by node_type, keep first
        self._bank: Dict[str, Node] = {}
        for n in nodes:
            self._bank.setdefault(n.node_type, n)

    def _candidates(self, used: set, last_ix: int, has_y: bool) -> List[Node]:
        """Legal next nodes: not already used, forward-only, within the adjacency
        window, and (without labels) excluding supervised nodes so a routed run
        never throws on a missing ``y`` (Rule 21 — breaks nothing)."""
        out = []
        for nt, node in self._bank.items():
            if nt in used:
                continue
            li = _LAYER_IX.get(node.layer, 99)
            if li < last_ix:                       # forward-only (no layer regress)
                continue
            is_terminal = node.layer in TERMINAL_LAYERS
            # feature layers respect the adjacency window; a terminal node is
            # always reachable forward (you may choose to finish anytime).
            if not is_terminal and last_ix >= 0 and li > last_ix + self.max_layer_jump:
                continue
            if node.requires_y and not has_y:
                continue
            out.append(node)
        return sorted(out, key=lambda n: (_LAYER_IX.get(n.layer, 99), n.node_type))

    def _iter(self, X: Any, y: Optional[Any], decisions_out: List[RouterDecision]):
        bus = X
        used: set = set()
        last_ix = -1
        visited_layers: List[str] = []
        beliefs: List[Belief] = []
        has_y = y is not None
        for step in range(self.max_hops):
            cands = self._candidates(used, last_ix, has_y)
            state = RoutingState(
                candidates=[n.metadata() for n in cands],
                beliefs=list(beliefs),
                visited_layers=list(visited_layers),
                last_layer=(visited_layers[-1] if visited_layers else None),
                step=step, max_hops=self.max_hops,
            )
            action = self.router.choose(state)
            decisions_out.append(RouterDecision(
                step=step, chosen=action.node_type,
                layer=(self._bank[action.node_type].layer if action.node_type else None),
                reason=action.reason, candidates=state.candidate_types(),
            ))
            if action.is_stop or not cands:
                break
            node = self._bank[action.node_type]
            t0 = time.perf_counter()
            belief = node.process(bus, y) if node.requires_y else node.process(bus)
            transformed = False
            if node.is_transformer:
                bus = node.transform(bus)
                transformed = True
            elapsed = (time.perf_counter() - t0) * 1000.0
            beliefs.append(belief)
            used.add(node.node_type)
            last_ix = _LAYER_IX.get(node.layer, last_ix)
            visited_layers.append(node.layer)
            yield Hop(step, node.node_type, node.name, node.layer,
                      belief, transformed, elapsed)
            if node.layer in TERMINAL_LAYERS:      # a decision terminates the path
                break

    def iter_run(self, X: Any, y: Optional[Any] = None):
        """Stream each :class:`Hop` as the router selects and runs it (the §8
        dashboard WebSocket source). Discards the rationale; use :meth:`route`
        to keep it."""
        yield from self._iter(X, y, [])

    def run(self, X: Any, y: Optional[Any] = None) -> PathwayResult:
        """Run the routed pathway end-to-end -> a standard PathwayResult."""
        t0 = time.perf_counter()
        hops = list(self._iter(X, y, []))
        total = (time.perf_counter() - t0) * 1000.0
        return PathwayResult(
            pathway=[h.node_type for h in hops], hops=hops,
            output=hops[-1].belief if hops else None, elapsed_ms=total,
        )

    def route(self, X: Any, y: Optional[Any] = None) -> RoutedResult:
        """Like :meth:`run`, but also returns the per-hop routing rationale."""
        decisions: List[RouterDecision] = []
        t0 = time.perf_counter()
        hops = list(self._iter(X, y, decisions))
        total = (time.perf_counter() - t0) * 1000.0
        res = PathwayResult(
            pathway=[h.node_type for h in hops], hops=hops,
            output=hops[-1].belief if hops else None, elapsed_ms=total,
        )
        return RoutedResult(result=res, decisions=decisions)

    def __repr__(self) -> str:
        return (f"<RoutedConnector {self.name!r} router={self.router.name} "
                f"bank={len(self._bank)} max_hops={self.max_hops}>")


def default_routed_connector() -> RoutedConnector:
    """The step-4 routed connector over the full default bank, using the offline
    :class:`HeuristicRouter` (no LLM/API needed to run or test)."""
    return RoutedConnector(HeuristicRouter(), name="routed-heuristic")
