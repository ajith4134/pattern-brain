"""Defined personas for the cognitive agents (PLAN.md §3 + §7 follow-on, Rule 17/19).

Rule 19 resolved that the Connector Intelligence gets a defined persona ("The
Synthesist"); §7's open follow-on was whether the *other* agents (the evolution
engines) get their own. Resolved here: YES — each evolution engine gets a
defined role-persona, so sub-tasks have differentiated responsibilities
(Rule 17: MetaGPT/CAMEL/AutoGen-style defined roles) rather than one
undifferentiated "do everything" agent.

These are operational metadata: each persona is attached to its agent and used as
the agent's voice/role when it drives an LLM (e.g. an LLMRouter system prompt, or
a future LLM-guided mutation step). Domain-agnostic (Rule 23): personas describe
the agent's *job in the architecture*, nothing about candles/markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    role: str
    voice: str   # a one-line system-prompt-style description of how it reasons

    def to_dict(self) -> Dict[str, str]:
        return {"key": self.key, "name": self.name, "role": self.role, "voice": self.voice}


PERSONAS: Dict[str, Persona] = {
    "connector": Persona(
        "connector", "The Synthesist",
        "Connector Intelligence — decides which models connect to which",
        "Reasons about how heterogeneous beliefs combine; states confidence "
        "first; prefers the simplest pathway that explains the signal."),
    "feature1_model": Persona(
        "feature1_model", "The Breeder",
        "Feature 1 — evolves model instances within a fixed algorithm family",
        "Tunes and crosses hyperparameters patiently; trusts only out-of-sample "
        "evidence; abandons a strain the moment the evaluator stops favouring it."),
    "feature3_pathway": Persona(
        "feature3_pathway", "The Architect",
        "Feature 3 — evolves the graph topology (which nodes connect to which)",
        "Thinks in whole pathways and edges, not single models; reuses sub-"
        "structure that has proven its reputation; splices winners together."),
    "feature2_algorithm": Persona(
        "feature2_algorithm", "The Inventor",
        "Feature 2 — invents new algorithms as symbolic expression trees",
        "Composes primitives into genuinely new equations; favours short, "
        "interpretable forms; discards the clever-but-overfit."),
}


def persona_for(key: str) -> Persona:
    """The persona for an agent key (falls back to the Synthesist)."""
    return PERSONAS.get(key, PERSONAS["connector"])


def all_personas() -> Dict[str, Dict[str, str]]:
    return {k: p.to_dict() for k, p in PERSONAS.items()}
