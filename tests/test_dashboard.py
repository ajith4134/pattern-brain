"""Contract + behavior tests for the §8 dashboard backend (build-tracker step 2
cross-cutting track, PLAN.md §8 / Block 43).

Requires the extra dashboard deps (requirements-dashboard.txt) — fastapi,
httpx, websockets — which the core model bank does not need. Run:
    python3 tests/test_dashboard.py
Exits non-zero if any check fails. Uses FastAPI's TestClient (no real socket,
no separate server process) so this runs the same way in CI as the other
test_*.py files.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"))

from fastapi.testclient import TestClient

import server  # dashboard/server.py
from pattern_brain.connector import DEFAULT_PATHWAY

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


client = TestClient(server.app)


def test_index_serves_html():
    r = client.get("/")
    check(r.status_code == 200, f"/ status {r.status_code}")
    check("text/html" in r.headers.get("content-type", ""), "/ did not serve html")
    check("Pattern Brain" in r.text, "/ html missing title")
    print("  GET / -> 200 html")


def test_bank_endpoint():
    r = client.get("/api/bank")
    check(r.status_code == 200, f"/api/bank status {r.status_code}")
    d = r.json()
    check(d["n_nodes"] == len(d["nodes"]), "n_nodes mismatch with nodes list")
    check(d["n_nodes"] >= 30, f"expected >=30 bank nodes, got {d['n_nodes']}")
    check(set(d["layers"]) >= {"signal", "pattern", "sequence", "decision"},
          f"layers missing expected entries: {d['layers']}")
    print(f"  GET /api/bank -> {d['n_nodes']} nodes, layers={d['layers']}")


def test_pathway_endpoint():
    r = client.get("/api/pathway")
    check(r.status_code == 200, f"/api/pathway status {r.status_code}")
    d = r.json()
    check(d["pathway"] == DEFAULT_PATHWAY, f"pathway drifted: {d['pathway']}")
    check([n["node_type"] for n in d["nodes"]] == DEFAULT_PATHWAY,
          "pathway node metadata out of order")
    print(f"  GET /api/pathway -> {' -> '.join(d['pathway'])}")


def test_run_endpoint():
    r = client.get("/api/run")
    check(r.status_code == 200, f"/api/run status {r.status_code}")
    d = r.json()
    check(d["pathway"] == DEFAULT_PATHWAY, "run pathway drifted")
    check(len(d["hops"]) == len(DEFAULT_PATHWAY), "run hop count mismatch")
    check(d["output"]["type"] == "decision", "run output not a decision belief")
    print(f"  GET /api/run -> {len(d['hops'])} hops, output={d['output']['type']}")


def test_adapter_endpoint():
    """Step-3 Adapter View: candles -> StockDataAdapter -> generic (T,D) -> the
    SAME core pathway. The endpoint must return all three so the tab can show the
    full handoff (PLAN.md §0)."""
    r = client.get("/api/adapter")
    check(r.status_code == 200, f"/api/adapter status {r.status_code}")
    d = r.json()
    check(d["n_candles"] > 0, "no candles returned")
    for k in ("open", "high", "low", "close", "volume", "timestamp"):
        check(len(d["candles"][k]) == d["n_candles"], f"candle column {k} length mismatch")
    T, D = d["feature_shape"]
    check(len(d["feature_names"]) == D, "feature_names != D")
    check(len(d["features"]) == T and (T == 0 or len(d["features"][0]) == D),
          "feature matrix shape mismatch")
    check(d["pathway"] == DEFAULT_PATHWAY, "adapter pathway drifted from core")
    check(d["trace"]["output"]["type"] == "decision",
          "core did not produce a decision on adapter output")
    print(f"  GET /api/adapter -> {d['n_candles']} candles -> (T,D)={d['feature_shape']} "
          f"-> {d['trace']['output']['type']}")


def test_ws_run_streams_start_hops_done():
    with client.websocket_connect("/ws/run") as ws:
        start = ws.receive_json()
        check(start["event"] == "start", f"first frame not 'start': {start}")
        check(start["pathway"] == DEFAULT_PATHWAY, "ws start pathway drifted")
        hops = []
        for _ in range(len(DEFAULT_PATHWAY)):
            msg = ws.receive_json()
            check(msg["event"] == "hop", f"expected hop frame, got {msg.get('event')}")
            hops.append(msg["hop"])
        check([h["node_type"] for h in hops] == DEFAULT_PATHWAY,
              "ws hop order diverges from DEFAULT_PATHWAY")
        done = ws.receive_json()
        check(done["event"] == "done", f"final frame not 'done': {done}")
        check(done["output"]["type"] == "decision", "ws done output not a decision")
    print("  WS /ws/run -> start, "
          f"{len(hops)} hop frame(s), done(output={done['output']['type']})")


def main():
    print("=" * 70)
    print("Pattern Brain — §8 dashboard backend: contract + behavior tests")
    print("=" * 70)
    test_index_serves_html()
    test_bank_endpoint()
    test_pathway_endpoint()
    test_run_endpoint()
    test_adapter_endpoint()
    test_ws_run_streams_start_hops_done()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
