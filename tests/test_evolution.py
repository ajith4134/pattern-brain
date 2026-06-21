"""Contract + behavior tests for Features 1/2/3 — the evolution engines
(final phase, slice 2; PLAN.md §4).

Run: python3 tests/test_evolution.py
Encodes the spec per Rule 25: each engine evolves candidates and promotes ONLY
through the shared Evaluator (the hard gate). The decisive properties:
 * all three run end-to-end (init -> score -> select -> breed) with an audit
   history (born/tested/promoted) — the §8 "Evolution feed";
 * the gate REJECTS a pure random walk (no real edge) for ALL three — the
   anti-overfitting guarantee, with proper multiple-testing deflation + CSCV/PBO;
 * a matched engine (Feature 1 on autoregressive data) DOES promote through the
   gate — the promotion path works.
Domain-agnostic (Rule 23): generic directional-reward fitness; no candle refs.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.evolution import (
    ModelEvolver, PathwayEvolver, AlgorithmEvolver, Individual, EvolutionResult,
)
from pattern_brain.routing import _LAYER_IX

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def ar_edge(T=700, coef=0.6, seed=0):
    """A sequence with a genuine one-step-predictable edge: AR(1) increments."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = coef * x[t - 1] + rng.normal(0, 0.3)
    return np.column_stack([np.cumsum(x) + 50.0, rng.normal(0, 1, T)])


def random_walk(T=700, seed=7):
    """The proper null: a pure random walk — next move independent of the past,
    so NO directional edge exists and the gate must admit nothing."""
    rng = np.random.default_rng(seed)
    return np.column_stack([np.cumsum(rng.normal(0, 1, T)),
                            np.cumsum(rng.normal(0, 1, T))])


ENGINES = [ModelEvolver, PathwayEvolver, AlgorithmEvolver]


def test_engines_run_breed_and_audit():
    X = ar_edge(T=400)
    for Eng in ENGINES:
        res = Eng(population=6, generations=3, seed=1).run(X)
        check(isinstance(res, EvolutionResult), f"{Eng.__name__}: no EvolutionResult")
        check(res.best is not None and res.best.report is not None,
              f"{Eng.__name__}: no scored best")
        events = {h["event"] for h in res.history}
        check("tested" in events, f"{Eng.__name__}: nothing tested")
        check("born" in events, f"{Eng.__name__}: no offspring bred")
        origins = {h["origin"] for h in res.history}
        check({"mutate", "crossover"} & origins, f"{Eng.__name__}: no mutate/crossover")
        json.dumps(res.to_dict())  # the Evolution-feed payload must serialize
        print(f"  {res.feature:20} ran: {len(res.history)} events, best ucb={res.best.fitness():.3f}")


def test_gate_rejects_random_walk():
    """THE anti-overfitting property: on a pure random walk none of the engines
    may promote anything (proper deflation + CSCV/PBO catch the best-of-many)."""
    rw = random_walk(T=700)
    for Eng in ENGINES:
        res = Eng(population=8, generations=4, seed=2).run(rw)
        check(len(res.admitted) == 0,
              f"{Eng.__name__}: admitted {len(res.admitted)} on a random walk (overfit leak)")
    print("  gate: random-walk null -> 0 admitted for all three engines")


def test_matched_engine_promotes_on_edge():
    """Feature 1 (autoregressive model) is the matched tool for AR data and must
    clear the gate — proving the promotion path actually works end-to-end."""
    X = ar_edge(T=800, coef=0.65)
    res = ModelEvolver(population=8, generations=4, seed=2).run(X)
    check(len(res.admitted) >= 1,
          f"ModelEvolver admitted nothing on a genuine AR edge: best DSR={res.best.report.dsr:.3f}")
    for ind in res.admitted:
        check(ind.report.passed, "admitted candidate not marked passed")
        check(ind.report.psr >= 0.5, "admitted candidate has implausibly low PSR")
    print(f"  promotion: ModelEvolver admitted {len(res.admitted)} on AR edge "
          f"(best DSR={res.best.report.dsr:.3f})")


def test_mutate_crossover_valid():
    X = ar_edge(T=300)
    for Eng in ENGINES:
        ev = Eng(population=4, generations=1, seed=3)
        a, b = ev.random_genome(), ev.random_genome()
        for g in (ev.mutate(a), ev.crossover(a, b)):
            # a valid genome must be scorable without raising
            try:
                cand = pb.directional_candidate_fn(ev.make_predictor_fn(g))
                out = ev.evaluator.walk_forward_outcomes(cand, X)
                check(out.size > 0, f"{Eng.__name__}: bred genome produced no outcomes")
            except Exception as e:
                check(False, f"{Eng.__name__}: bred genome not scorable: {type(e).__name__}: {e}")
    print("  breeding: mutate + crossover produce valid, scorable genomes")


def test_pathway_genomes_are_layer_legal():
    """Feature 3 genomes must respect Block-18 layer ordering and end in a
    forecaster head (so the evolved graph is always runnable)."""
    ev = PathwayEvolver(seed=5)
    heads = set(PathwayEvolver._HEADS)
    for _ in range(40):
        g = ev.mutate(ev.crossover(ev.random_genome(), ev.random_genome()))
        check(g[-1] in heads, f"pathway does not end in a forecaster head: {g}")
        ixs = [_LAYER_IX[pb.create(nt).layer] for nt in g]
        check(all(b >= a for a, b in zip(ixs, ixs[1:])), f"pathway not layer-ordered: {g}")
        check(len(g) == len(set(g)), f"pathway has duplicate nodes: {g}")
    print("  feature3: evolved pathways stay layer-legal and head-terminated")


def test_algorithm_trees_evaluate():
    """Feature 2 expression trees evaluate to a finite scalar on a window."""
    ev = AlgorithmEvolver(seed=6)
    w = np.linspace(1.0, 2.0, 16)
    for _ in range(50):
        g = ev.mutate(ev.random_genome())
        v = ev._eval(g, w)
        check(np.isfinite(v), f"expression tree gave non-finite value: {g} -> {v}")
    print("  feature2: evolved expression trees evaluate to finite scalars")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "evolution.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in evolution.py: {hits}")
    print("  domain-independence: evolution has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — Features 1/2/3 evolution engines: contract tests")
    print("=" * 70)
    test_engines_run_breed_and_audit()
    test_gate_rejects_random_walk()
    test_matched_engine_promotes_on_edge()
    test_mutate_crossover_valid()
    test_pathway_genomes_are_layer_legal()
    test_algorithm_trees_evaluate()
    test_domain_independence()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
