"""Domain adapters (build-tracker step 3+).

This subpackage is the ONLY place domain-specific (stock/candle) code lives
(Rule 23 / PLAN.md §0). It is deliberately NOT imported by ``pattern_brain``'s
top-level ``__init__`` — the core stays domain-agnostic and importable without
touching any adapter. The dependency only ever flows adapter -> core.

A future second domain plugs in by adding a sibling module here (e.g.
``adapters/<other>.py``) exposing its own adapter, without changing the model
bank, the Connector, or the belief format — that portability is the test of
PLAN.md §0 (build domain-independent first; bend the data to the system second).
"""
from .stock import (
    CandleSeries,
    StockDataAdapter,
    frac_diff,
    frac_diff_weights,
    align_to_grid,
    align_sources,
)

__all__ = [
    "CandleSeries",
    "StockDataAdapter",
    "frac_diff",
    "frac_diff_weights",
    "align_to_grid",
    "align_sources",
]
