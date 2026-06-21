"""Tests for the real LLM backends behind the tool-calling router
(PLAN.md step 4 / Block 45).

Run: python3 tests/test_llm.py
No live backend needed: the response PARSERS are pure functions (tested with mock
responses), availability degrades gracefully, and the LLMRouter loop is driven
through a fake completer end-to-end. If a real backend happens to be reachable,
that's reported but not required.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain import llm
from pattern_brain.routing import RoutedConnector, LLMRouter, tool_specs, RoutingState

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_parse_anthropic():
    class Block:
        def __init__(self, t, n=None):
            self.type = t; self.name = n
    class Resp:
        content = [Block("text"), Block("tool_use", "gaussian_hmm")]
    out = llm.parse_anthropic_response(Resp())
    check(out["name"] == "gaussian_hmm", f"anthropic parse picked {out}")
    # dict form + no tool_use -> stop
    check(llm.parse_anthropic_response({"content": [{"type": "text"}]})["name"] == "stop",
          "no tool_use should map to stop")
    print("  parse: anthropic tool_use extracted; no-tool -> stop")


def test_parse_ollama():
    resp = {"message": {"tool_calls": [{"function": {"name": "difference"}}]}}
    check(llm.parse_ollama_response(resp)["name"] == "difference", "ollama parse failed")
    check(llm.parse_ollama_response({"message": {}})["name"] == "stop",
          "no tool_call should map to stop")
    print("  parse: ollama tool_call extracted; no-tool -> stop")


def test_availability_and_auto_graceful():
    # these must never raise, regardless of environment
    a = llm.anthropic_available()
    o = llm.ollama_available(timeout=0.5)
    check(isinstance(a, bool) and isinstance(o, bool), "availability not boolean")
    completer, name = llm.auto_completer()
    if completer is None:
        check(name is None and pb.default_llm_router() is None,
              "no backend -> default_llm_router must be None (heuristic fallback)")
    else:
        check(name in ("ollama", "anthropic"), f"unexpected backend {name}")
    status = llm.llm_backend_status()
    check("router" in status and "active_backend" in status, "status missing keys")
    print(f"  availability: ollama={o} anthropic={a} -> backend={name or 'heuristic'}")


def test_llmrouter_loop_with_fake_backend():
    """The whole tool-calling loop runs against a fake completer (no network),
    proving the LLM wiring is correct independent of any real backend."""
    plan = iter(["difference", "hdbscan", "gaussian_hmm", "threshold_policy"])

    def fake_complete(system, user, tools):
        offered = {t["name"] for t in tools}
        try:
            nxt = next(plan)
        except StopIteration:
            return {"name": "stop"}
        return {"name": nxt if nxt in offered else "stop"}

    X = np.cumsum(np.random.default_rng(0).normal(size=(200, 2)), axis=0)
    conn = RoutedConnector(LLMRouter(fake_complete), name="routed-llm")
    res = conn.run(X)
    check(res.pathway == ["difference", "hdbscan", "gaussian_hmm", "threshold_policy"],
          f"LLM-routed path diverged: {res.pathway}")
    check(res.output is not None and res.output.type == "decision", "no terminal decision")
    print(f"  loop: fake-LLM tool-calling drove {' -> '.join(res.pathway)}")


def test_tool_schema_conversion():
    state = RoutingState(candidates=[pb.create("difference").metadata()],
                         beliefs=[], visited_layers=[], last_layer=None, step=0, max_hops=8)
    tools = tool_specs(state)
    anth = llm._to_anthropic_tools(tools)
    olla = llm._to_ollama_tools(tools)
    check(all("input_schema" in t for t in anth), "anthropic tool schema malformed")
    check(all(t.get("type") == "function" and "function" in t for t in olla),
          "ollama tool schema malformed")
    print("  schema: generic tool specs convert to anthropic + ollama formats")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "llm.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in llm.py: {hits}")
    print("  domain-independence: llm.py has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — real LLM backends for the router: tests")
    print("=" * 70)
    test_parse_anthropic()
    test_parse_ollama()
    test_availability_and_auto_graceful()
    test_llmrouter_loop_with_fake_backend()
    test_tool_schema_conversion()
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
