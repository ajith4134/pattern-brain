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
    register, create, all_node_types, by_layer, layers, default_bank, genome_manifest,
)
from .personas import Persona, PERSONAS, persona_for, all_personas
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
from .evaluator import (
    Evaluator, EvaluationReport, ObjectiveSpec, DEFAULT_OBJECTIVES,
    purged_walk_forward_splits, sharpe, max_drawdown, stability,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, min_track_record_length,
    expected_max_sharpe, discounted_ucb_score, sliding_window_ucb_score,
    nondominated_sort, pareto_front, cscv_pbo, anchor_check,
)
from .evolution import (
    Evolver, ModelEvolver, PathwayEvolver, AlgorithmEvolver,
    Individual, EvolutionResult, directional_candidate_fn,
)
from .reputation import (
    PathwayReputation, PathwayStat, ReputationRouter,
)
from .knowledge import (
    Embedder, HashingEmbedder, STEmbedder, OllamaEmbedder, default_embedder, chunk_text,
    VectorStore, Passage, Retrieval, KnowledgeBase, BookRef,
    BOOK_MANIFEST, book_manifest, DEFAULT_STORE_DIR,
)
from .llm import (
    auto_completer, auto_text_completer, default_llm_router, llm_backend_status,
    ollama_completer, anthropic_completer, ollama_available, anthropic_available,
    parse_anthropic_response, parse_ollama_response, parse_openai_response,
    parse_openai_text, openai_compatible_completer, openai_compatible_chat,
    available_cloud_providers, CLOUD_PROVIDERS, ALL_BACKENDS,
    ollama_chat, anthropic_chat,
)

__all__ = [
    "Belief", "SCHEMA_VERSION", "Node",
    "register", "create", "all_node_types", "by_layer", "layers", "default_bank",
    "genome_manifest", "Persona", "PERSONAS", "persona_for", "all_personas",
    "Connector", "PathwayResult", "Hop", "DEFAULT_PATHWAY", "default_connector",
    "Router", "HeuristicRouter", "LLMRouter", "RouterAction", "RouterDecision",
    "RouterError", "RoutingState", "RoutedConnector", "RoutedResult", "LAYER_ORDER",
    "tool_spec", "tool_specs", "default_routed_connector",
    "INTERLINGUA_VERSION", "InterlinguaError", "BeliefTypeSchema",
    "register_belief_type", "belief_type_schema", "known_belief_types", "catalog",
    "register_migration", "has_migration_path", "migrate_belief", "Violation",
    "validate_belief", "detect_drift", "ConformanceReport", "conformance_report",
    "CoherenceNote", "interlingua_coherence",
    "Evaluator", "EvaluationReport", "ObjectiveSpec", "DEFAULT_OBJECTIVES",
    "purged_walk_forward_splits", "sharpe", "max_drawdown", "stability",
    "probabilistic_sharpe_ratio", "deflated_sharpe_ratio", "min_track_record_length",
    "expected_max_sharpe", "discounted_ucb_score", "sliding_window_ucb_score",
    "nondominated_sort", "pareto_front", "cscv_pbo", "anchor_check",
    "Evolver", "ModelEvolver", "PathwayEvolver", "AlgorithmEvolver",
    "Individual", "EvolutionResult", "directional_candidate_fn",
    "PathwayReputation", "PathwayStat", "ReputationRouter",
    "Embedder", "HashingEmbedder", "STEmbedder", "OllamaEmbedder", "default_embedder", "chunk_text",
    "VectorStore", "Passage", "Retrieval", "KnowledgeBase", "BookRef",
    "BOOK_MANIFEST", "book_manifest", "DEFAULT_STORE_DIR",
    "auto_completer", "auto_text_completer", "default_llm_router", "llm_backend_status",
    "ollama_completer", "anthropic_completer", "ollama_available", "anthropic_available",
    "parse_anthropic_response", "parse_ollama_response", "parse_openai_response",
    "parse_openai_text", "openai_compatible_completer", "openai_compatible_chat",
    "available_cloud_providers", "CLOUD_PROVIDERS", "ALL_BACKENDS",
    "ollama_chat", "anthropic_chat",
]
