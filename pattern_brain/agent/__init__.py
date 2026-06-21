"""ML Engineer Agent (PLAN.md §9 / Block 56) — the autonomous orchestration layer.

ENG-2 ships the tools (:class:`Toolbox`); ENG-3 adds the loop (:class:`MLEngineerAgent`).
Kept domain-agnostic (Rule 23) and writing all runtime files inside the project
folder (Rule 1 clarification).
"""
from .tools import (
    Toolbox, RunResult, synthetic_dataset, DEFAULT_DATA_DIR,
)
from .engineer import (
    MLEngineerAgent, StepResult, PERSONA, FEATURES, DEFAULT_STATE_DIR,
)
from .advisor import IdeationAdvisor, Idea, KNOWN_GAPS

__all__ = ["Toolbox", "RunResult", "synthetic_dataset", "DEFAULT_DATA_DIR",
           "MLEngineerAgent", "StepResult", "PERSONA", "FEATURES", "DEFAULT_STATE_DIR",
           "IdeationAdvisor", "Idea", "KNOWN_GAPS"]
