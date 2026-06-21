"""Pattern Brain — a domain-independent pattern-finding / sequence-prediction
model bank (build-tracker step 1).

Everything here operates on a generic (T, D) sequence of D-dimensional vectors
and emits Belief objects through one common interface — no candle/order-book/
stock-specific shape anywhere (PLAN.md §0 / Rule 23). Stock data is plugged in
later via a separate adapter (build-tracker step 3).
"""
from .belief import Belief, SCHEMA_VERSION
from .node import Node
from . import nodes  # noqa: F401  (import for side effect: registers built-in nodes)
from .registry import (
    register, create, all_node_types, by_layer, layers, default_bank,
)
from .connector import (
    Connector, PathwayResult, Hop, DEFAULT_PATHWAY, default_connector,
)
from .routing import (
    Router, HeuristicRouter, LLMRouter, RouterAction, RouterDecision, RouterError,
    RoutingState, RoutedConnector, RoutedResult, LAYER_ORDER, tool_spec, tool_specs,
    default_routed_connector,
)
from .interlingua import (
    INTERLINGUA_VERSION, InterlinguaError, BeliefTypeSchema, register_belief_type,
    belief_type_schema, known_belief_types, catalog, register_migration,
    has_migration_path, migrate_belief, Violation, validate_belief, detect_drift,
    ConformanceReport, conformance_report, CoherenceNote, interlingua_coherence,
)

__all__ = [
    "Belief", "SCHEMA_VERSION", "Node",
    "register", "create", "all_node_types", "by_layer", "layers", "default_bank",
    "Connector", "PathwayResult", "Hop", "DEFAULT_PATHWAY", "default_connector",
    "Router", "HeuristicRouter", "LLMRouter", "RouterAction", "RouterDecision",
    "RouterError", "RoutingState", "RoutedConnector", "RoutedResult", "LAYER_ORDER",
    "tool_spec", "tool_specs", "default_routed_connector",
    "INTERLINGUA_VERSION", "InterlinguaError", "BeliefTypeSchema",
    "register_belief_type", "belief_type_schema", "known_belief_types", "catalog",
    "register_migration", "has_migration_path", "migrate_belief", "Violation",
    "validate_belief", "detect_drift", "ConformanceReport", "conformance_report",
    "CoherenceNote", "interlingua_coherence",
]
