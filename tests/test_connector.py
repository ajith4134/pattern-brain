"""Contract + behavior tests for Connector Intelligence v0 (build-tracker step 2).

Run: python3 tests/test_connector.py
Exits non-zero if any check fails. Encodes the step-2 spec (PLAN.md tracker
item 2) as checkable conditions per Rule 25: wire ONE hardcoded pathway
end-to-end through the Universal Belief Space, data-agnostic, no learned routing.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.belief import Belief
from pattern_brain.connector import (
    Connector, PathwayResult, Hop, DEFAULT_PATHWAY, default_connector,
)

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


def make_synth(T=200, D=3, seed=0):
    """A generic regime-switching multivariate sequence (no domain meaning)."""
    rng = np.random.default_rng(seed)
    half = T // 2
    t = np.arange(T)
    X = np.zeros((T, D))
    X[:half] = (0.03 * t[:half])[:, None] + rng.normal(0, 0.3, (half, D))
    X[half:] = (2.0 - 0.02 * (t[half:] - half))[:, None] \
        + 0.5 * np.sin(0.3 * t[half:])[:, None] + rng.normal(0, 0.3, (T - half, D))
    return X


def test_runs_end_to_end():
    """The hardcoded pathway runs end-to-end and returns a PathwayResult with one
    hop per node, in order."""
    X = make_synth()
    conn = default_connector()
    check(conn.pathway == DEFAULT_PATHWAY, f"pathway spec drifted: {conn.pathway}")
    res = conn.run(X)
    check(isinstance(res, PathwayResult), "run() did not return a PathwayResult")
    check([h.node_type for h in res.hops] == DEFAULT_PATHWAY,
          f"hops out of order/incomplete: {[h.node_type for h in res.hops]}")
    check([h.step for h in res.hops] == list(range(len(DEFAULT_PATHWAY))),
          "hop step indices not sequential")
    check(res.elapsed_ms >= 0.0, "elapsed_ms negative")
    print(f"  end-to-end: {' -> '.join(res.pathway)}  ({res.elapsed_ms:.2f} ms)")


def test_every_hop_is_a_valid_belief():
    """Every node on the path emits a valid Belief into the stream (Block 5)."""
    res = default_connector().run(make_synth())
    stream = res.beliefs()
    check(len(stream) == len(DEFAULT_PATHWAY), "belief stream length mismatch")
    for h in res.hops:
        b = h.belief
        check(isinstance(b, Belief), f"{h.node_type}: hop.belief not a Belief")
        check(isinstance(b.type, str) and b.type, f"{h.node_type}: empty belief.type")
        check(0.0 <= b.confidence <= 1.0,
              f"{h.node_type}: confidence {b.confidence} out of [0,1]")
        check(b.source == h.name, f"{h.node_type}: belief.source {b.source!r} != {h.name!r}")


def test_trace_is_json_serializable():
    """The whole run trace must be JSON-serializable — the dashboard/audit
    requirement of the shared belief space."""
    res = default_connector().run(make_synth())
    try:
        s = json.dumps(res.to_dict())
    except (TypeError, ValueError) as e:
        check(False, f"PathwayResult.to_dict() not JSON-serializable: {e}")
        return
    d = json.loads(s)
    check(d["pathway"] == DEFAULT_PATHWAY, "serialized pathway mismatch")
    check(len(d["hops"]) == len(DEFAULT_PATHWAY), "serialized hop count mismatch")
    check(d["output"] is not None and "belief" not in d["output"],
          "serialized output is not a belief dict")
    print(f"  trace JSON: {len(s)} bytes, {len(d['hops'])} hops")


def test_terminal_decision():
    """This pathway ends in a generic discrete decision; the helper resolves it."""
    res = default_connector().run(make_synth())
    check(res.output is not None and res.output.type == "decision",
          f"terminal belief is not a decision: {res.output}")
    check(res.decision is res.output, "decision helper did not resolve the terminal decision")
    action = res.output.payload.get("action")
    check(action in {"buy", "sell", "hold"}, f"unexpected generic action: {action}")
    print(f"  terminal decision: action={action} conf={res.output.confidence:.3f}")


def test_signal_bus_is_threaded():
    """The transformer node rewrites the bus; non-transformers pass it through.
    Proves the bus is actually threaded, not each node re-reading the raw input."""
    res = default_connector().run(make_synth())
    flags = {h.node_type: h.transformed for h in res.hops}
    check(flags["difference"] is True, "difference (a transformer) did not rewrite the bus")
    for nt in ("hdbscan", "gaussian_hmm", "threshold_policy"):
        check(flags[nt] is False, f"{nt} unexpectedly flagged as transforming the bus")

    # Behavioral proof of threading: the downstream threshold_policy sees the
    # DIFFERENCED bus, so its z-score differs from running it on the raw input.
    X = make_synth()
    raw = pb.create("threshold_policy").predict(X).payload["z"]
    threaded = res.decision.payload["z"]
    check(abs(raw - threaded) > 1e-9,
          "threshold_policy saw the same input as raw -> bus not threaded")
    print(f"  bus threaded: raw z={raw:.3f} vs differenced z={threaded:.3f}")


def test_accepts_instances_and_1d():
    """Connector accepts Node instances (not just type strings) and a 1-D series."""
    nodes = [pb.create(nt) for nt in DEFAULT_PATHWAY]
    conn = Connector(nodes, name="from-instances")
    check(conn.pathway == DEFAULT_PATHWAY, "instance-built pathway spec mismatch")
    x = np.cumsum(np.random.default_rng(1).normal(size=120))  # 1-D
    res = conn.run(x)
    check(res.output is not None, "1-D input did not produce a terminal belief")
    print("  flexibility: Node-instance pathway + 1-D input both run")


def test_unknown_node_and_empty_pathway():
    """Construction fails loudly on a bad spec (no silent degradation)."""
    try:
        Connector(["difference", "not_a_real_node"])
        check(False, "unknown node_type did not raise")
    except KeyError:
        pass
    try:
        Connector([])
        check(False, "empty pathway did not raise")
    except ValueError:
        pass
    print("  validation: unknown node -> KeyError, empty pathway -> ValueError")


def test_iter_run_streams_hops_matching_run():
    """``iter_run`` (added for the §8 dashboard's WebSocket, Block 43) must yield
    the identical hop sequence ``run`` collects — it's a generator wrapping the
    same execution, not a second code path that could drift from it."""
    X = make_synth()
    streamed = list(default_connector().iter_run(X))
    collected = default_connector().run(X).hops

    check(len(streamed) == len(DEFAULT_PATHWAY), "iter_run hop count mismatch")
    check([h.node_type for h in streamed] == [h.node_type for h in collected],
          "iter_run node order diverges from run()")
    check([h.transformed for h in streamed] == [h.transformed for h in collected],
          "iter_run transform flags diverge from run()")
    check([h.belief.type for h in streamed] == [h.belief.type for h in collected],
          "iter_run belief types diverge from run()")
    check(streamed[-1].belief.payload.get("action")
          == collected[-1].belief.payload.get("action"),
          "iter_run terminal action diverges from run()")
    print("  iter_run: streaming generator matches run()'s collected hops")


def test_domain_independence():
    """Rule 23: the connector couples to no stock-specific data shape. Inspect
    code identifiers only (tokenize), not docstrings — prose legitimately names
    candles/order books to explain what the code avoids."""
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "connector.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers found in connector.py: {hits}")
    print("  domain-independence: no candle/ohlcv/orderbook coupling (Rule 23 ok)")


def main():
    print("=" * 70)
    print("Pattern Brain — Connector Intelligence v0: contract + behavior tests")
    print("=" * 70)
    test_runs_end_to_end()
    test_every_hop_is_a_valid_belief()
    test_trace_is_json_serializable()
    test_terminal_decision()
    test_signal_bus_is_threaded()
    test_accepts_instances_and_1d()
    test_unknown_node_and_empty_pathway()
    test_iter_run_streams_hops_matching_run()
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
