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
    check(d["n_nodes"] >= 55, f"expected the step-6-expanded bank (>=55), got {d['n_nodes']}")
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


def test_overview_endpoint():
    r = client.get("/api/overview")
    check(r.status_code == 200, f"/api/overview status {r.status_code}")
    d = r.json()
    check(d["models"] >= 55, f"overview models too few: {d['models']}")
    check(d["n_files"] >= 10, f"overview n_files too few: {d['n_files']}")
    check(len(d["subsystems"]) >= 6, "overview missing subsystems")
    check(all(b["done"] for b in d["build"]), "overview build steps not all done")
    print(f"  GET /api/overview -> {d['models']} models, {d['n_files']} files, "
          f"{len(d['subsystems'])} subsystems")


def test_files_endpoint():
    r = client.get("/api/files")
    check(r.status_code == 200, f"/api/files status {r.status_code}")
    d = r.json()
    check(d["n_files"] == len(d["files"]), "files count mismatch")
    check(any(f["path"].startswith("pattern_brain/") for f in d["files"]), "no core files listed")
    check(any(f["path"].startswith("tests/") for f in d["files"]), "no test files listed")
    check(all("lines" in f and "summary" in f for f in d["files"]), "file entries missing fields")
    print(f"  GET /api/files -> {d['n_files']} files, {d['total_lines']} lines")


def test_evaluator_endpoint():
    r = client.get("/api/evaluator")
    check(r.status_code == 200, f"/api/evaluator status {r.status_code}")
    d = r.json()
    check(len(d["folds"]) >= 3, "evaluator missing folds")
    check(len(d["layers"]) == 5, "evaluator should expose 5 layers")
    check(d["edge"]["passed"] is True, f"edge candidate should pass: {d['edge']}")
    check(d["noise"]["passed"] is False, "noise candidate should be rejected")
    print(f"  GET /api/evaluator -> edge passes={d['edge']['passed']}, noise passes={d['noise']['passed']}")


def test_evolution_endpoint():
    r = client.get("/api/evolution")
    check(r.status_code == 200, f"/api/evolution status {r.status_code}")
    d = r.json()
    check(set(d["features"]) >= {"feature1_model", "feature3_pathway", "feature2_algorithm"},
          f"evolution missing features: {list(d['features'])}")
    for f, fd in d["features"].items():
        check(len(fd["events"]) > 0, f"{f}: no evolution events")
    print(f"  GET /api/evolution -> {list(d['features'])}")


def test_llm_endpoint():
    r = client.get("/api/llm")
    check(r.status_code == 200, f"/api/llm status {r.status_code}")
    d = r.json()
    check("router" in d and "active_backend" in d, "llm status missing keys")
    check(isinstance(d["ollama_available"], bool), "ollama_available not boolean")
    print(f"  GET /api/llm -> router={d['router']}")


def test_reputation_endpoint():
    r = client.get("/api/reputation")
    check(r.status_code == 200, f"/api/reputation status {r.status_code}")
    d = r.json()
    check(len(d["leaderboard"]) >= 1, "reputation leaderboard empty")
    check(all("pathway" in p and "score" in p for p in d["leaderboard"]),
          "leaderboard entries missing fields")
    check(isinstance(d["edges"], dict) and len(d["edges"]) >= 1, "no edge reputations")
    print(f"  GET /api/reputation -> {len(d['leaderboard'])} pathways, {len(d['edges'])} edges")


def test_route_endpoint():
    """Step-4 Connector v1: /api/route returns a router-discovered pathway plus
    the per-hop rationale (the dashboard's routed-pathway card source)."""
    r = client.get("/api/route")
    check(r.status_code == 200, f"/api/route status {r.status_code}")
    d = r.json()
    check(len(d["pathway"]) >= 2, f"routed pathway too short: {d['pathway']}")
    check(len(d["decisions"]) >= len(d["pathway"]), "fewer decisions than hops")
    check(all("reason" in dec and "candidates" in dec for dec in d["decisions"]),
          "router decisions missing reason/candidates")
    check(d["output"] is not None, "routed run produced no terminal belief")
    print(f"  GET /api/route -> {' -> '.join(d['pathway'])} (router={d['router']})")


def test_conformance_endpoint():
    """Step-5 interlingua: /api/conformance returns the version + catalog, a drift
    report over a run, and coherence notes (the conformance card's source)."""
    r = client.get("/api/conformance")
    check(r.status_code == 200, f"/api/conformance status {r.status_code}")
    d = r.json()
    check(d["version"], "no interlingua version returned")
    check(len(d["catalog"]) >= 10, f"catalog too small: {len(d['catalog'])}")
    check(d["run_report"]["conforms"] is True,
          f"a default run should conform: {d['run_report']}")
    check(isinstance(d["coherence"], list) and len(d["coherence"]) >= 1,
          "expected at least one coherence note over the bank")
    print(f"  GET /api/conformance -> v{d['version']}, {len(d['catalog'])} types, "
          f"run conforms={d['run_report']['conforms']}, {len(d['coherence'])} coherence notes")


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


def test_coverage_endpoint():
    """Catalog-vs-built coverage meter: per-family ✅/🟡/❌ with built/missing lists."""
    r = client.get("/api/coverage")
    check(r.status_code == 200, f"/api/coverage status {r.status_code}")
    d = r.json()
    check(d["n_registered_nodes"] >= 100, "too few registered nodes")
    check(len(d["families"]) >= 15, "coverage should cover >=15 families")
    fam = {f["family"]: f for f in d["families"]}
    check(fam["Pattern mining"]["status"] == "complete", "pattern-mining should be complete (7b)")
    check(all(f["n_built"] + len(f["missing"]) == f["n_target"] for f in d["families"]),
          "built+missing must equal target per family")
    check(0 <= d["curated_pct"] <= 100, "curated_pct out of range")
    print(f"  GET /api/coverage -> {d['curated_built']}/{d['curated_target']} ({d['curated_pct']}%), "
          f"{len(d['families'])} families")


def test_concept_bank_endpoint():
    """Concept-equation-bank build status: built vs the open 'to implement' list."""
    r = client.get("/api/concept_bank")
    check(r.status_code == 200, f"/api/concept_bank status {r.status_code}")
    d = r.json()
    check(d["built_node"] + d["built_module"] == d["done"], "done must equal built Node+module")
    check(d["done"] + d["not_built"] == d["buildable"], "done+not_built must equal buildable")
    check(d["not_built"] == len(d["to_implement"]), "to_implement list must match not_built count")
    check(len(d["tiers"]) == 3, "expected 3 tiers")
    check(all("frontier" in x and "tier" in x for x in d["to_implement"]), "each todo needs tier+frontier")
    check(0 <= d["done_pct"] <= 100, "done_pct out of range")
    print(f"  GET /api/concept_bank -> {d['done']}/{d['buildable']} built ({d['done_pct']}%), "
          f"{d['not_built']} to implement")


def test_galton_endpoint():
    """The Galton board: bins, random counts, exact Binomial, and Normal overlay."""
    r = client.get("/api/galton?rows=10&balls=3000")
    check(r.status_code == 200, f"/api/galton status {r.status_code}")
    d = r.json()
    check(len(d["bins"]) == d["rows"] + 1, "wrong number of bins")
    check(sum(d["counts"]) == d["balls"], "ball counts don't sum to balls dropped")
    check(len(d["binomial"]) == d["rows"] + 1 and len(d["normal"]) == d["rows"] + 1,
          "binomial/normal overlays wrong length")
    check(abs(d["mean"] - d["rows"] / 2) < 1e-9, "mean should be rows/2")
    # peak of the random histogram should be near the center bin (law of large numbers)
    peak = d["counts"].index(max(d["counts"]))
    check(abs(peak - d["rows"] / 2) <= 2, f"histogram peak {peak} not near center")
    print(f"  GET /api/galton -> {d['balls']} balls, {d['rows']} rows, peak near bin {peak}")


def test_leaderboard_endpoint():
    """Phase-8 Stacked-DAG leaderboard: summary + ranked rows (demo if unpopulated)."""
    r = client.get("/api/leaderboard")
    check(r.status_code == 200, f"/api/leaderboard status {r.status_code}")
    d = r.json()
    check("summary" in d and "top" in d, "leaderboard payload missing summary/top")
    check(isinstance(d["top"], list) and len(d["top"]) >= 1, "leaderboard should have >=1 row")
    row = d["top"][0]
    for k in ("combiner", "n_nodes", "mean_crps", "crps_skill", "spec_json"):
        check(k in row, f"leaderboard row missing {k}")
    check(d["summary"]["n_runs"] >= 1, "summary n_runs should be >=1")
    print(f"  GET /api/leaderboard -> {d['summary']['n_runs']} runs, "
          f"best skill {d['summary']['best_skill']} (demo={d.get('demo')})")


def test_capstone_endpoint():
    """Phase-8 capstone: forward-test result + per-step predicted-vs-realized paths."""
    r = client.get("/api/capstone")
    check(r.status_code == 200, f"/api/capstone status {r.status_code}")
    d = r.json()
    check(d.get("verdict") in ("PASS", "WEAK", "FAIL"), f"bad verdict {d.get('verdict')}")
    fp = d.get("forward_paths", {})
    check(all(k in fp for k in ("index", "realized", "q10", "q50", "q90")), "missing forward paths")
    n = len(fp.get("index", []))
    check(n > 0 and len(fp["q50"]) == n == len(fp["realized"]), "path arrays length mismatch")
    check(all(a <= b for a, b in zip(fp["q10"], fp["q90"])), "q10 must be <= q90 per step")
    print(f"  GET /api/capstone -> verdict={d['verdict']}, {n} forward steps, skill={d['forward_skill']:.3f}")


def test_node_skill_endpoint():
    """Per-node OOF skill is computed in a background thread; the endpoint returns
    a progress payload immediately (non-blocking)."""
    r = client.get("/api/node_skill")
    check(r.status_code == 200, f"/api/node_skill status {r.status_code}")
    d = r.json()
    check(all(k in d for k in ("ready", "n_total", "n_done", "skills")), "node_skill payload incomplete")
    check(isinstance(d["skills"], dict), "skills should be a dict")
    print(f"  GET /api/node_skill -> ready={d['ready']}, {d['n_done']}/{d['n_total']} (non-blocking)")


def test_knowledge_endpoint():
    """ENG-4: /api/knowledge returns the book manifest + KB stats + LLM status."""
    r = client.get("/api/knowledge")
    check(r.status_code == 200, f"/api/knowledge status {r.status_code}")
    d = r.json()
    check(len(d["books"]) >= 6, "book manifest too small")
    check("passages" in d["stats"] and "archival" in d["stats"], "kb stats missing keys")
    check("chain" in d["llm"], "llm status missing chain")
    print(f"  GET /api/knowledge -> {len(d['books'])} books, embedder={d['stats']['embedder']}")


def test_agent_status_and_step_and_chat():
    """ENG-4: agent status, one loop step (Observe→…→Rank), and a chat turn."""
    s = client.get("/api/agent/status")
    check(s.status_code == 200 and "steps_done" in s.json(), "agent status malformed")
    st = client.post("/api/agent/step")
    check(st.status_code == 200, f"/api/agent/step status {st.status_code}")
    d = st.json()
    check(d["result"]["feature"] in
          ("feature1_model", "feature2_algorithm", "feature3_pathway"),
          f"step feature bad: {d['result'].get('feature')}")
    check(d["status"]["steps_done"] >= 1, "step did not advance the counter")
    c = client.post("/api/agent/chat", json={"message": "what is the state of the bank?"})
    check(c.status_code == 200 and isinstance(c.json()["reply"], str) and c.json()["reply"],
          "agent chat returned no reply")
    print(f"  POST /api/agent/step -> {d['result']['feature']}; chat reply ok")


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
    test_overview_endpoint()
    test_files_endpoint()
    test_evaluator_endpoint()
    test_evolution_endpoint()
    test_llm_endpoint()
    test_reputation_endpoint()
    test_route_endpoint()
    test_conformance_endpoint()
    test_adapter_endpoint()
    test_coverage_endpoint()
    test_leaderboard_endpoint()
    test_capstone_endpoint()
    test_node_skill_endpoint()
    test_galton_endpoint()
    test_knowledge_endpoint()
    test_agent_status_and_step_and_chat()
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
