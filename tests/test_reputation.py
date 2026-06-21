"""Contract + behavior tests for pathway reputation — closing the loop
(PLAN.md §6 / Block 17: "the graph itself becomes the memory").

Run: python3 tests/test_reputation.py
Spec (Rule 25): the Evaluator's score is remembered on the pathway + its edges;
routing can then prefer proven edges; evolution populates the store. Domain-
agnostic (Rule 23): generic node sequences + outcome stats, no candle refs.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.reputation import PathwayReputation, ReputationRouter
from pattern_brain.routing import RoutingState, HeuristicRouter

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_record_and_score():
    rep = PathwayReputation()
    good = ["difference", "autoregressive", "sign_vote"]
    bad = ["difference", "drift_forecast", "sign_vote"]
    rep.record(good, outcomes=np.full(60, 0.4))    # strong positive stream
    rep.record(bad, outcomes=np.full(60, -0.2))    # negative stream
    check(rep.pathway_score(good) > rep.pathway_score(bad),
          "the better-performing pathway should score higher")
    check(rep.edge_score("difference", "autoregressive") is not None,
          "a recorded edge should have a score")
    check(rep.edge_score("difference", "never_seen") is None,
          "an unseen edge must return None (so routing can explore it)")
    print(f"  record: good {rep.pathway_score(good):.3f} > bad {rep.pathway_score(bad):.3f}")


def test_leaderboard_and_serialization():
    rep = PathwayReputation()
    for i in range(5):
        rep.record(["difference", "autoregressive", "sign_vote"] if i % 2 else
                   ["ewma", "theta_forecast", "sign_vote"],
                   outcomes=np.full(40, 0.1 * (i + 1)))
    lb = rep.leaderboard(top=3)
    check(len(lb) >= 2 and lb[0].score >= lb[-1].score, "leaderboard not sorted desc")
    json.dumps(rep.to_dict())  # dashboard payload must serialize
    check("edges" in rep.to_dict() and "leaderboard" in rep.to_dict(), "to_dict missing keys")
    print(f"  leaderboard: {len(lb)} pathways, top score {lb[0].score:.3f}, serializes")


def test_regime_conditioning():
    rep = PathwayReputation()
    pw = ["difference", "autoregressive", "sign_vote"]
    rep.record(pw, outcomes=np.full(40, 0.5), regime="bull")
    rep.record(pw, outcomes=np.full(40, -0.5), regime="bear")
    check(rep.pathway_score(pw, "bull") > rep.pathway_score(pw, "bear"),
          "same pathway should score differently per regime (Block 17 World Model)")
    check(set(rep.known_regimes()) >= {"bull", "bear"}, "regimes not tracked")
    print("  regime: same pathway scored per-regime (bull > bear)")


def test_reputation_router_prefers_proven_edge():
    rep = PathwayReputation()
    # teach it that difference -> autoregressive is great, difference -> drift bad
    rep.record(["difference", "autoregressive"], outcomes=np.full(60, 0.6))
    rep.record(["difference", "drift_forecast"], outcomes=np.full(60, -0.6))
    router = ReputationRouter(rep)
    # state where the last node was 'difference' and both are candidates
    state = RoutingState(
        candidates=[pb.create("autoregressive").metadata(), pb.create("drift_forecast").metadata()],
        beliefs=[pb.Belief("signal", {}, 0.5, "difference")],
        visited_layers=["signal"], last_layer="signal", step=1, max_hops=8)
    action = router.choose(state)
    check(action.node_type == "autoregressive",
          f"router should pick the higher-reputation edge, picked {action.node_type}")
    check("reputation" in action.reason, "router reason should cite edge reputation")
    # unseen edge -> falls back to heuristic, not a crash
    empty = ReputationRouter(PathwayReputation())
    a2 = empty.choose(state)
    check(a2.node_type in {"autoregressive", "drift_forecast"}, "fallback produced no choice")
    check("no edge reputation yet" in a2.reason, "fallback reason not annotated")
    print("  router: prefers proven edge; explores (heuristic) on unseen edges")


def test_evolution_populates_store():
    """Feature 3 must record every tested pathway into its reputation store —
    the wiring that closes the loop."""
    rng = np.random.default_rng(0)
    T = 500
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng.normal(0, 0.3)
    X = np.column_stack([np.cumsum(x) + 50, rng.normal(0, 1, T)])
    ev = pb.PathwayEvolver(population=6, generations=2, seed=1)
    ev.run(X)
    board = ev.reputation.leaderboard(top=999)
    check(len(board) >= 3, f"evolution recorded too few pathways: {len(board)}")
    check(all(s.uses >= 1 for s in board), "recorded pathways have no usage count")
    print(f"  wiring: PathwayEvolver recorded {len(board)} pathways into reputation")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "reputation.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in reputation.py: {hits}")
    print("  domain-independence: reputation has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — pathway reputation (closing the loop): tests")
    print("=" * 70)
    test_record_and_score()
    test_leaderboard_and_serialization()
    test_regime_conditioning()
    test_reputation_router_prefers_proven_edge()
    test_evolution_populates_store()
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
