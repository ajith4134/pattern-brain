"""The model-bank registry (DISCUSSION_NOTES.md Block 18 / PLAN.md §5-§6).

A catalog of named, typed node classes decoupled from any pipeline logic — the
seed of the "Model Genome Database". Mirrors Kedro's node+catalog idea (Block 35)
at a minimal scale; the vector-DB-backed similarity layer (Block 29) is a later
upgrade, not needed to register and look up nodes.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .node import Node

_REGISTRY: Dict[str, Type[Node]] = {}


def register(cls: Type[Node]) -> Type[Node]:
    """Class decorator: add a node class to the bank, keyed by its node_type."""
    nt = getattr(cls, "node_type", None)
    if not nt or nt == "abstract":
        raise ValueError(f"{cls.__name__} must set a concrete node_type before @register")
    if nt in _REGISTRY and _REGISTRY[nt] is not cls:
        raise ValueError(f"node_type {nt!r} already registered by {_REGISTRY[nt].__name__}")
    _REGISTRY[nt] = cls
    return cls


def create(node_type: str, **kwargs) -> Node:
    """Instantiate a registered node by type."""
    if node_type not in _REGISTRY:
        raise KeyError(f"unknown node_type {node_type!r}; known: {all_node_types()}")
    return _REGISTRY[node_type](**kwargs)


def all_node_types() -> List[str]:
    return sorted(_REGISTRY)


def layers() -> List[str]:
    return sorted({cls.layer for cls in _REGISTRY.values()})


def by_layer(layer: str) -> List[str]:
    return sorted(nt for nt, cls in _REGISTRY.items() if cls.layer == layer)


def genome_manifest() -> Dict[str, object]:
    """The "Model Genome Library" deliverable (resolves §7's open 'cpu ml models'
    format question): a JSON-serializable manifest of the whole bank — every
    node's metadata, grouped by layer — plus the requirements needed to
    materialize it. The deliverable is the catalog + a build recipe (manifest +
    requirements), NOT committed weight files: every node here is constructed on
    demand from light deps (numpy/scipy/sklearn) or optional torch."""
    nodes = [n.metadata() for n in default_bank()]
    return {
        "schema": "pattern-brain/genome-manifest@1",
        "n_nodes": len(nodes),
        "layers": layers(),
        "by_layer": {l: by_layer(l) for l in layers()},
        "nodes": nodes,
        "requirements": {
            "light": ["numpy>=1.26", "scipy>=1.11", "scikit-learn>=1.4"],
            "deep_optional": ["torch>=2.2"],
        },
    }


def default_bank() -> List[Node]:
    """One default instance of every registered node, ordered by functional layer
    (Block 18: signal -> noise -> pattern -> sequence -> probability -> equation
    -> decision -> rl) then node_type."""
    order = {
        "signal": 0, "noise": 1, "pattern": 2, "sequence": 3,
        "probability": 4, "equation": 5, "decision": 6, "rl": 7,
    }
    items = sorted(_REGISTRY.items(), key=lambda kv: (order.get(kv[1].layer, 99), kv[0]))
    return [cls() for _, cls in items]
