"""Pathway reputation — closing the loop (PLAN.md §6 / Block 17:
"score CONNECTIONS, not models — the graph itself becomes the memory").

This is what makes the pieces cohere: the Evaluator scores a candidate's outcome
stream, and that score is *remembered on the pathway and its edges* so future
routing can prefer connections that have proven themselves. Three coupled pieces:

* :class:`PathwayReputation` — the memory: per-pathway usage + a recency-weighted
  (discounted-UCB) outcome score, plus per-edge aggregates, optionally
  regime-conditioned (Block 17). The unit of memory is the edge/pathway.
* :class:`ReputationRouter` — a :class:`~pattern_brain.routing.Router` that picks
  the next node by the reputation of the edge it would create, falling back to a
  heuristic layer-advance for never-seen edges (curiosity).
* wiring — :class:`~pattern_brain.evolution.PathwayEvolver` records every tested
  pathway here, so after evolution the store can drive routing and the §8
  dashboard's Pathway Leaderboard (view #3) + reputation-coloured edges (view #1).

Domain-agnostic (PLAN.md §0 / Rule 23): reputation is keyed by generic node-type
sequences and scored from generic outcome statistics; no candle/order-book refs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evaluator import discounted_ucb_score
from .routing import Router, RouterAction, RoutingState, HeuristicRouter, _LAYER_IX

Pathway = Tuple[str, ...]


@dataclass
class PathwayStat:
    """Accumulated reputation for one pathway (optionally one regime)."""
    pathway: Pathway
    uses: int = 0
    score: float = 0.0          # recency-weighted outcome value (discounted UCB)
    mean_outcome: float = 0.0   # plain running mean (for display)
    last_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"pathway": list(self.pathway), "uses": self.uses,
                "score": round(self.score, 4), "mean_outcome": round(self.mean_outcome, 4),
                "report": self.last_report}


class PathwayReputation:
    """The pathway/edge memory. ``record`` ingests an outcome series (and/or an
    EvaluationReport) for a pathway; scores are queryable per pathway and per edge,
    regime-conditioned when a ``regime`` key is supplied (Block 17's World Model)."""

    def __init__(self, gamma: float = 0.95) -> None:
        self.gamma = gamma
        self._by_key: Dict[Tuple[str, Pathway], PathwayStat] = {}

    @staticmethod
    def _edges(pathway: Sequence[str]) -> List[Tuple[str, str]]:
        return list(zip(pathway[:-1], pathway[1:]))

    def record(self, pathway: Sequence[str], outcomes: Optional[Sequence[float]] = None,
               report: Optional[Any] = None, regime: str = "all") -> PathwayStat:
        key = (regime, tuple(pathway))
        st = self._by_key.get(key) or PathwayStat(tuple(pathway))
        st.uses += 1
        if outcomes is not None and len(outcomes) > 0:
            st.score = float(discounted_ucb_score(outcomes, gamma=self.gamma))
            st.mean_outcome = float(sum(outcomes) / len(outcomes))
        elif report is not None:
            st.score = float(getattr(report, "ucb", 0.0))
            st.mean_outcome = float(getattr(report, "sharpe", 0.0))
        if report is not None:
            st.last_report = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        self._by_key[key] = st
        return st

    def pathway_score(self, pathway: Sequence[str], regime: str = "all") -> float:
        st = self._by_key.get((regime, tuple(pathway)))
        return st.score if st else 0.0

    def edge_score(self, a: str, b: str, regime: str = "all") -> Optional[float]:
        """Mean pathway score over recorded pathways containing the edge a->b
        (None if that edge has never been seen — lets routing explore it)."""
        scores = [st.score for (rg, pw), st in self._by_key.items()
                  if rg == regime and (a, b) in self._edges(pw)]
        return sum(scores) / len(scores) if scores else None

    def leaderboard(self, regime: str = "all", top: int = 20) -> List[PathwayStat]:
        stats = [st for (rg, _), st in self._by_key.items() if rg == regime]
        return sorted(stats, key=lambda s: s.score, reverse=True)[:top]

    def known_regimes(self) -> List[str]:
        return sorted({rg for rg, _ in self._by_key})

    def to_dict(self, regime: str = "all", top: int = 20) -> Dict[str, Any]:
        lb = self.leaderboard(regime, top)
        edges: Dict[str, float] = {}
        for (rg, pw), st in self._by_key.items():
            if rg != regime:
                continue
            for a, b in self._edges(pw):
                edges[f"{a}->{b}"] = max(edges.get(f"{a}->{b}", -1e18), st.score)
        return {"regime": regime, "n_pathways": len(lb),
                "leaderboard": [s.to_dict() for s in lb],
                "edges": {k: round(v, 4) for k, v in edges.items()}}


class ReputationRouter(Router):
    """Routes by learned edge reputation: among the legal candidates, pick the
    node whose edge from the last node has the highest reputation; fall back to a
    heuristic layer-advance when no candidate edge has been seen (exploration).
    This is Block 17's "the graph is the memory" made operational in routing."""

    name = "reputation"

    def __init__(self, reputation: PathwayReputation, regime: str = "all",
                 fallback: Optional[Router] = None) -> None:
        self.reputation = reputation
        self.regime = regime
        self.fallback = fallback or HeuristicRouter()

    def choose(self, state: RoutingState) -> RouterAction:
        last = state.last_layer
        last_nt = None
        # the most recent node_type is the source of the next edge
        if state.beliefs:
            last_nt = state.beliefs[-1].source
        if last_nt is not None and state.candidates:
            scored = []
            for c in state.candidates:
                s = self.reputation.edge_score(last_nt, c["node_type"], self.regime)
                if s is not None:
                    scored.append((s, c))
            if scored:
                scored.sort(key=lambda t: t[0], reverse=True)
                best = scored[0][1]
                return RouterAction(best["node_type"],
                                    reason=f"edge reputation {scored[0][0]:.3f}")
        # unseen territory -> defer to the heuristic (explore)
        act = self.fallback.choose(state)
        if act.node_type is not None:
            act.reason = "no edge reputation yet -> " + act.reason
        return act
