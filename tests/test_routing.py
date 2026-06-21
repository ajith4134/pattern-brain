"""Contract + behavior tests for Connector Intelligence v1 — LLM-tool-calling
routing (build-tracker step 4).

Run: python3 tests/test_routing.py
Exits non-zero if any check fails. Encodes the step-4 spec (PLAN.md tracker
item 4) as checkable conditions per Rule 25: a router decides the pathway
hop-by-hop (not a fixed list), respecting Block-18 layer adjacency; the LLM is a
swappable injected boundary proven offline with a fake; routed runs reuse the
same PathwayResult so the dashboard/audit consume them identically.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.belief import Belief
from pattern_brain.connector import PathwayResult, DEFAULT_PATHWAY
from pattern_brain.routing import (
    LAYER_ORDER, _LAYER_IX, RouterError,
    HeuristicRouter, LLMRouter, RoutedConnector, RoutingState, RouterAction,
    tool_specs, default_routed_connector,
)

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


def make_synth(T=200, D=3, seed=0):
    rng = np.random.default_rng(seed)
    half = T // 2
    t = np.arange(T)
    X = np.zeros((T, D))
    X[:half] = (0.03 * t[:half])[:, None] + rng.normal(0, 0.3, (half, D))
    X[half:] = (2.0 - 0.02 * (t[half:] - half))[:, None] + rng.normal(0, 0.3, (T - half, D))
    return X


def _is_forward(layers):
    """layer indices never decrease along the path."""
    ix = [_LAYER_IX[l] for l in layers]
    return all(b >= a for a, b in zip(ix, ix[1:]))


# ----------------------------------------------------- heuristic router (offline)
def test_heuristic_runs_and_terminates_in_decision():
    """The offline default router assembles a real multi-layer pathway that ends
    in a decision and returns a standard PathwayResult."""
    res = default_routed_connector().run(make_synth())
    check(isinstance(res, PathwayResult), "routed run did not return a PathwayResult")
    check(len(res.hops) >= 2, f"path too short to be routed: {res.pathway}")
    layers = [h.layer for h in res.hops]
    check(_is_forward(layers), f"path not forward through layers: {layers}")
    check(res.output is not None and res.output.type == "decision",
          f"routed path did not terminate in a decision: {res.output}")
    check(res.hops[-1].layer == "decision", f"last hop not a decision: {layers}")
    print(f"  heuristic route: {' -> '.join(res.pathway)}  (layers {layers})")


def test_routing_is_dynamic_not_the_fixed_pathway():
    """Routing must be CHOSEN from the bank, not the v0 hardcoded list — restrict
    the bank and the discovered path changes accordingly."""
    full = default_routed_connector().run(make_synth()).pathway
    # restrict the bank to a different set of nodes -> a different routed path
    subset = [pb.create(nt) for nt in
              ("difference", "fft", "kmeans", "autoregressive", "sign_vote")]
    restricted = RoutedConnector(HeuristicRouter(), bank=subset).run(make_synth()).pathway
    check(restricted != full, "restricted bank produced the same path (not routing?)")
    check(set(restricted) <= {n.node_type for n in subset},
          f"routed path used nodes outside the given bank: {restricted}")
    check(restricted != DEFAULT_PATHWAY or full != DEFAULT_PATHWAY,
          "routed path is just the v0 fixed DEFAULT_PATHWAY")
    print(f"  dynamic: full={full}  vs restricted={restricted}")


def test_layer_adjacency_window_enforced():
    """Block-18 constraint: a candidate can't be more than max_layer_jump layers
    ahead of the last node — routing can't leap signal -> rl."""
    conn = RoutedConnector(HeuristicRouter(), max_layer_jump=1)
    res = conn.run(make_synth())
    ix = [_LAYER_IX[h.layer] for h in res.hops]
    jumps = [b - a for a, b in zip(ix, ix[1:])]
    check(all(0 <= j <= 1 for j in jumps), f"layer jump exceeded window 1: {jumps}")
    print(f"  adjacency: jumps within window: {jumps}")


# ------------------------------------------------------- LLM boundary (offline fake)
def test_llm_router_tool_calling_loop():
    """LLMRouter drives the loop purely through an injected complete(); a fake
    completer choosing specific tools proves the whole tool-calling loop works
    with no network. It must receive tools each hop and its choices must execute."""
    seen_tools = []
    plan = iter(["difference", "hdbscan", "gaussian_hmm", "threshold_policy"])

    def fake_complete(system, user, tools):
        seen_tools.append([t["name"] for t in tools])
        check("domain-independent" in system or "generic" in system,
              "LLM system prompt should state domain independence")
        try:
            nxt = next(plan)
        except StopIteration:
            return {"name": "stop", "reason": "plan exhausted"}
        # only choose it if it's actually offered this hop, else stop (loud-safe)
        offered = {t["name"] for t in tools}
        return {"name": nxt if nxt in offered else "stop", "reason": "fake plan"}

    conn = RoutedConnector(LLMRouter(fake_complete), name="routed-llm")
    res = conn.run(make_synth())
    check(res.pathway == ["difference", "hdbscan", "gaussian_hmm", "threshold_policy"],
          f"LLM-routed path did not follow the fake plan: {res.pathway}")
    check(len(seen_tools) >= len(res.hops), "completer not called once per hop")
    check(all("stop" in names for names in seen_tools),
          "the synthetic 'stop' tool was not offered every hop")
    check(res.output.type == "decision", "LLM route did not end in a decision")
    print(f"  llm loop: {' -> '.join(res.pathway)}  ({len(seen_tools)} completer calls)")


def test_llm_router_rejects_illegal_choice():
    """An LLM returning a tool not on offer fails loudly (no silent degradation)."""
    def bad_complete(system, user, tools):
        return {"name": "not_a_real_tool"}
    conn = RoutedConnector(LLMRouter(bad_complete))
    try:
        conn.run(make_synth())
        check(False, "illegal tool choice did not raise RouterError")
    except RouterError:
        pass
    # also: a malformed response (no 'name') raises
    try:
        RoutedConnector(LLMRouter(lambda s, u, t: {"oops": 1})).run(make_synth())
        check(False, "malformed completer response did not raise RouterError")
    except RouterError:
        pass
    print("  llm safety: illegal/malformed tool choices -> RouterError")


def test_tool_specs_are_json_and_domain_free():
    """The tool schemas handed to a real LLM must be JSON-serializable and carry
    no stock-specific fields (Rule 23)."""
    conn = default_routed_connector()
    state = RoutingState(
        candidates=[pb.create(nt).metadata() for nt in ("difference", "fft")],
        beliefs=[], visited_layers=[], last_layer=None, step=0, max_hops=8,
    )
    tools = tool_specs(state)
    s = json.dumps(tools)
    check("stop" in [t["name"] for t in tools], "stop tool missing from specs")
    low = s.lower()
    for bad in ("candle", "ohlcv", "orderbook", "order_book"):
        check(bad not in low, f"tool spec leaked domain word {bad!r}")
    print(f"  tool specs: {len(tools)} JSON tools (incl. stop), domain-free")


# --------------------------------------------------------- safety / compatibility
def test_max_hops_caps_runaway():
    """A router that never stops is bounded by max_hops."""
    class NeverStop(HeuristicRouter):
        def choose(self, state):
            if not state.candidates:
                return RouterAction(None, "forced")
            return RouterAction(state.candidates[0]["node_type"], "first always")
    conn = RoutedConnector(NeverStop(), max_hops=3, max_layer_jump=7)
    res = conn.run(make_synth())
    check(len(res.hops) <= 3, f"max_hops not enforced: {len(res.hops)} hops")
    print(f"  safety: max_hops cap honored ({len(res.hops)} <= 3)")


def test_supervised_node_not_offered_without_labels():
    """Routing must never offer a requires_y node when no y is given (else the run
    throws) — Rule 21 breaks-nothing."""
    seen = set()

    def spy_complete(system, user, tools):
        for t in tools:
            seen.add(t["name"])
        return {"name": "stop"}
    RoutedConnector(LLMRouter(spy_complete)).run(make_synth())  # no y
    check("logistic_regression" not in seen,
          "a requires_y node was offered without labels")
    print("  safety: supervised nodes withheld when no labels present")


def test_route_keeps_rationale_and_is_serializable():
    """route() returns the per-hop rationale; the whole thing serializes (the §8
    dashboard 'router decisions' source)."""
    routed = default_routed_connector().route(make_synth())
    check(len(routed.decisions) >= len(routed.result.hops),
          "fewer decisions than hops recorded")
    d = json.loads(json.dumps(routed.to_dict()))
    check("decisions" in d and "hops" in d, "routed trace missing keys")
    check(all("reason" in dec and "candidates" in dec for dec in d["decisions"]),
          "router decisions missing reason/candidates")
    print(f"  rationale: {len(routed.decisions)} decisions, trace JSON-serializable")


def test_domain_independence():
    """Rule 23: the routing module couples to no stock-specific data shape."""
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "routing.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in routing.py: {hits}")
    print("  domain-independence: routing has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — Connector v1 routing (step 4): contract + behavior tests")
    print("=" * 70)
    test_heuristic_runs_and_terminates_in_decision()
    test_routing_is_dynamic_not_the_fixed_pathway()
    test_layer_adjacency_window_enforced()
    test_llm_router_tool_calling_loop()
    test_llm_router_rejects_illegal_choice()
    test_tool_specs_are_json_and_domain_free()
    test_max_hops_caps_runaway()
    test_supervised_node_not_offered_without_labels()
    test_route_keeps_rationale_and_is_serializable()
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
