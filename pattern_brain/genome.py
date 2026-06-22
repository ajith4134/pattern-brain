"""W3 — Model Genome Library: per-node metadata (cost / role / out-kind) and a
type-directed composition predicate (PLAN §17.1 W3).

The owner's vision: every model declares its input/output type + cost, so the
search can BUDGET (screen cheap nodes first) and PRUNE incompatible connections
*before* spending compute — "never propose the bad combo" instead of "try it then
fail". This module derives that metadata from the registry (Rule 23: generic, no
domain coupling) so it works for all 199 nodes without hand-annotating each; a
node may override its cost via the optional ``Node.cost`` class attribute.

Used by the W1 scheduler (``scheduler.py``) to drop terminal nodes from forecaster
candidate pools and to order/budget evaluation, and exposable on the dashboard as
the Model-Genome cards.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .registry import _REGISTRY, all_node_types

# Block-18 functional-layer ordering: composition flows left → right.
LAYER_ORDER: List[str] = ["signal", "noise", "pattern", "sequence",
                          "probability", "equation", "decision", "rl"]
_ORDER = {l: i for i, l in enumerate(LAYER_ORDER)}

_FEATURE_LAYERS = {"signal", "noise", "pattern"}          # front-of-pathway transforms/features
_PREDICT_LAYERS = {"sequence", "probability", "equation"}  # mid-pathway predictors
_TERMINAL_LAYERS = {"decision", "rl"}                     # terminal decisions / policies

# Rough cost heuristic for the light stack (torch nodes are detected by module).
_MED_COST_KEYS = ("gaussian_process", "_hmm", "hmm_", "gmm", "bayesian", "arima",
                  "sarima", "prophet", "garch", "reservoir", "esn", "hawkes",
                  "persistent_homology", "neural_ode", "pinn")


def _cls(nt: str):
    return _REGISTRY.get(nt)


def _module(nt: str) -> str:
    c = _cls(nt)
    return getattr(c, "__module__", "") if c is not None else ""


def node_cost(nt: str) -> str:
    """Estimated relative cost class: 'low' | 'med' | 'high'. Honors an explicit
    ``Node.cost`` override; otherwise: torch nodes = high, known heavier light
    families = med, everything else = low."""
    c = _cls(nt)
    if c is not None and getattr(c, "cost", None):
        return str(c.cost)
    if _module(nt).endswith("nodes.deep"):
        return "high"
    if any(k in nt for k in _MED_COST_KEYS):
        return "med"
    return "low"


def role(nt: str) -> str:
    """'feature' | 'predictor' | 'terminal' | 'other' from the node's layer."""
    c = _cls(nt)
    layer = getattr(c, "layer", "") if c is not None else ""
    if layer in _FEATURE_LAYERS:
        return "feature"
    if layer in _PREDICT_LAYERS:
        return "predictor"
    if layer in _TERMINAL_LAYERS:
        return "terminal"
    return "other"


def out_kind(nt: str) -> str:
    """What the node yields downstream: 'transform' (a derived series),
    'terminal' (an action/decision), 'predict', or 'feature'."""
    c = _cls(nt)
    if c is None:
        return "unknown"
    if getattr(c, "is_transformer", False):
        return "transform"
    layer = getattr(c, "layer", "")
    if layer in _TERMINAL_LAYERS:
        return "terminal"
    if layer in _PREDICT_LAYERS:
        return "predict"
    return "feature"


def compatible(a: str, b: str) -> bool:
    """Can node ``b`` meaningfully follow node ``a`` in a pathway? Type-directed
    composition rule (PLAN §17.1 W3): respect the layer ordering (no going
    backwards) and never chain OUT of a terminal (a decision/policy is an
    endpoint, not an upstream feature). Unknown nodes default to allowed so
    nothing is silently excluded (Rule: ask/allow, don't guess-exclude)."""
    ca, cb = _cls(a), _cls(b)
    if ca is None or cb is None:
        return True
    if ca.layer in _TERMINAL_LAYERS:
        return False
    return _ORDER.get(cb.layer, 99) >= _ORDER.get(ca.layer, 0)


def genome_card(nt: str) -> Dict[str, Any]:
    c = _cls(nt)
    return {"node_type": nt, "layer": getattr(c, "layer", None) if c else None,
            "role": role(nt), "out_kind": out_kind(nt), "cost": node_cost(nt),
            "is_transformer": bool(getattr(c, "is_transformer", False)) if c else False}


def genome_library() -> Dict[str, Dict[str, Any]]:
    """The full genome metadata table for every registered node (W3 / the owner's
    'Model Genome Library')."""
    return {nt: genome_card(nt) for nt in all_node_types()}


__all__ = ["LAYER_ORDER", "node_cost", "role", "out_kind", "compatible",
           "genome_card", "genome_library"]
