"""Pattern Brain dashboard — backend (build-tracker step 2, §8 walking skeleton).

A FastAPI app dedicated ONLY to Pattern Brain (no trading-bot coupling, Rule 1).
At step 2 the only live data is the Connector running one hardcoded pathway and
emitting a belief stream, so this serves exactly that:

* ``GET  /``            -> the single-file React dashboard.
* ``GET  /api/bank``    -> the model-bank genome (every node's metadata, by layer).
* ``GET  /api/pathway`` -> the step-2 hardcoded pathway + its nodes' metadata.
* ``GET  /api/run``     -> run the pathway once on synthetic data, full trace.
* ``GET  /api/route``   -> Connector v1 (step 4): a router decides the pathway
                          hop-by-hop; returns the discovered path + per-hop reasons.
* ``GET  /api/conformance`` -> interlingua (step 5): version + catalog, a drift
                          report over a run, and coherence notes over the bank.
* ``GET  /api/overview``    -> whole-system summary: counts for every subsystem,
                          build status, models-by-layer (the landing view).
* ``GET  /api/files``       -> every source file (core/tests/dashboard): path,
                          lines, kind, docstring summary.
* ``GET  /api/evaluator``   -> the 5-layer gate on a demo edge vs noise candidate.
* ``GET  /api/evolution``   -> Features 1/2/3 run; the born->tested->promoted feed.
* ``WS   /ws/run``      -> run the pathway, streaming each hop as it completes so
                          the living graph can animate belief-flow (the reason §8
                          chose a WebSocket backend).

Domain-agnostic core (Rule 23 / §8): the six core views above are generic
node/belief data on a synthetic ``(T, D)`` sequence — no candles/order books.
The ONLY stock-specific surface is the separate "Adapter View" (build-tracker
step 3), served by ``GET /api/adapter`` and rendered in its own tab — candles
in, generic ``(T, D)`` out, then the SAME core pathway runs on it. A future
domain lights up the same core via a different adapter, only this tab changing.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

app = FastAPI(title="Pattern Brain — Living Graph (step 2 walking skeleton)")


# --------------------------------------------- chat history persistence (ENG idea 2)
# The browser-side transcript is ephemeral (lost on reload); persist every turn to
# a tiny SQLite log keyed by session so past ML-Engineer conversations can be
# reviewed/replayed. stdlib sqlite3, its own file — no coupling to the leaderboard.
import sqlite3 as _sqlite3
import time as _chat_time

_CHAT_DB = os.path.join(DATA_DIR, "chat_history.sqlite")
_CHAT_DB_LOCK = threading.Lock()


def _chat_db() -> "_sqlite3.Connection":
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _sqlite3.connect(_CHAT_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_log ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session TEXT NOT NULL,"
        "  ts REAL NOT NULL,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  reasoning TEXT DEFAULT '',"
        "  backend TEXT DEFAULT '')")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_chat_session ON chat_log(session, id)")
    return conn


def _chat_log(session: str, role: str, content: str,
              reasoning: str = "", backend: str = "") -> None:
    """Append one chat turn. Best-effort: a logging failure never breaks the chat."""
    try:
        with _CHAT_DB_LOCK:
            conn = _chat_db()
            conn.execute(
                "INSERT INTO chat_log(session, ts, role, content, reasoning, backend) "
                "VALUES (?,?,?,?,?,?)",
                (session or "default", _chat_time.time(), role, content or "", reasoning or "", backend or ""))
            conn.commit()
            conn.close()
    except Exception:
        pass


def synthetic_sequence(T: int = 240, D: int = 3, seed: int = 7) -> np.ndarray:
    """A generic regime-switching multivariate sequence (no domain meaning)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = np.zeros((T, D))
    third = T // 3
    X[:third] = (0.05 * t[:third])[:, None] + rng.normal(0, 0.3, (third, D))
    X[third:2 * third] = rng.normal(0, 0.6, (third, D))
    X[2 * third:] = np.sin(0.35 * t[2 * third:])[:, None] + rng.normal(0, 0.2, (T - 2 * third, D))
    return X


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


# The bank, the routed run, and the conformance report are all DETERMINISTIC
# (fixed bank, fixed synthetic data), and some became expensive after the step-6
# expansion to ~60 nodes. They auto-fire on page load, so compute each ONCE and
# cache it — otherwise repeated/concurrent loads recompute the whole bank and can
# saturate the single-process dev server. (Restart the server to refresh.)
@lru_cache(maxsize=1)
def _bank_payload() -> dict:
    nodes = [n.metadata() for n in pb.default_bank()]
    return {"layers": pb.layers(), "n_nodes": len(nodes), "nodes": nodes}


@app.get("/api/bank")
def bank() -> JSONResponse:
    """The model-genome map: one metadata record per registered node, grouped by
    functional layer (the precursor to §8's full genome view). Cached."""
    return JSONResponse(_bank_payload())


@app.get("/api/pathway")
def pathway() -> JSONResponse:
    """The step-2 hardcoded pathway and each node's metadata, in order."""
    conn = pb.default_connector()
    meta = {n["node_type"]: n for n in (pb.create(nt).metadata() for nt in conn.pathway)}
    return JSONResponse({
        "name": conn.name,
        "pathway": conn.pathway,
        "nodes": [meta[nt] for nt in conn.pathway],
    })


@app.get("/api/run")
def run() -> JSONResponse:
    """Run the pathway once on synthetic data; return the full auditable trace."""
    res = pb.default_connector().run(synthetic_sequence())
    return JSONResponse(res.to_dict())


@lru_cache(maxsize=1)
def _route_payload() -> dict:
    # use a real LLM backend if one is configured (Ollama/Anthropic), else the
    # offline heuristic — step-4 routing wired to a real model (Block 45).
    from pattern_brain.routing import RoutedConnector
    llm_router = pb.default_llm_router()
    router = llm_router or pb.HeuristicRouter()
    conn = RoutedConnector(router, name="routed")
    routed = conn.route(synthetic_sequence())
    return {
        "router": router.name,
        "llm": pb.llm_backend_status(),
        "pathway": routed.result.pathway,
        "decisions": [d.to_dict() for d in routed.decisions],
        "output": routed.result.output.to_dict() if routed.result.output else None,
        "trace": routed.result.to_dict(),
    }


@app.get("/api/llm")
def llm() -> JSONResponse:
    """Which LLM backend the router would use right now (Ollama / Anthropic /
    heuristic fallback) — step-4 LLM wiring made visible."""
    return JSONResponse(pb.llm_backend_status())


@app.get("/api/route")
def route() -> JSONResponse:
    """Connector v1 (build step 4): a router DECIDES the pathway hop-by-hop over
    the full bank (offline HeuristicRouter — no LLM/API needed). Returns the
    discovered pathway + the per-hop routing rationale. Cached (deterministic)."""
    return JSONResponse(_route_payload())


def synthetic_candles(T: int = 360, seed: int = 11):
    """A synthetic OHLCV candle series (geometric random walk) — used ONLY by the
    Adapter View to demonstrate the step-3 stock-data adapter. This is the one
    place in the dashboard that touches a domain shape (Rule 23 / §8)."""
    from pattern_brain.adapters import CandleSeries
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.012, T)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, T)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, T)))
    vol = rng.lognormal(mean=8.0, sigma=0.5, size=T)
    ts = np.arange(T, dtype=float) * 60.0
    return CandleSeries(open_, high, low, close, vol, timestamp=ts)


@app.get("/api/adapter")
def adapter() -> JSONResponse:
    """Step-3 Adapter View: synthetic candles -> the StockDataAdapter -> the
    generic ``(T, D)`` the core already wants -> the SAME default pathway runs on
    it. Returns the candles (for the candlestick — the only stock-specific
    visual), the emitted feature matrix, and the resulting pathway trace, so the
    tab shows the whole "data bends to the system" handoff (PLAN.md §0)."""
    from pattern_brain.adapters import StockDataAdapter
    cs = synthetic_candles()
    ad = StockDataAdapter()
    X = ad.transform(cs)
    res = pb.default_connector().run(X)
    return JSONResponse({
        "candles": {
            "timestamp": cs.timestamp.tolist(),
            "open": cs.open.tolist(),
            "high": cs.high.tolist(),
            "low": cs.low.tolist(),
            "close": cs.close.tolist(),
            "volume": cs.volume.tolist(),
        },
        "n_candles": len(cs),
        "feature_names": ad.feature_names,
        "features": X.tolist(),
        "feature_shape": list(X.shape),
        "pathway": res.pathway,
        "trace": res.to_dict(),
    })


@lru_cache(maxsize=1)
def _conformance_payload() -> dict:
    run = pb.default_connector().run(synthetic_sequence())
    rep = pb.conformance_report(run.beliefs())
    # coherence needs multiple emitters per type -> compute over the whole bank
    bank_beliefs = []
    Xb = synthetic_sequence()
    for node in pb.default_bank():
        try:
            b = node.process(Xb) if not node.requires_y else None
            if b is not None:
                bank_beliefs.append(b)
        except Exception:
            pass
    coherence = [n.to_dict() for n in pb.interlingua_coherence(bank_beliefs)]
    return {
        "version": pb.INTERLINGUA_VERSION,
        "catalog": pb.catalog(),
        "run_report": rep.to_dict(),
        "coherence": coherence,
    }


@app.get("/api/conformance")
def conformance() -> JSONResponse:
    """Step-5 interlingua view: the versioned shared-belief contract made visible.
    Returns the interlingua version + catalog, a conformance report over a default
    pathway run (drift detection), and coherence notes over the full bank. Cached
    (processes all ~60 nodes — the most expensive endpoint after step 6)."""
    return JSONResponse(_conformance_payload())


ROOT = os.path.dirname(HERE)  # the pattern-brain project root


@lru_cache(maxsize=1)
def _files_payload() -> dict:
    """Walk every source file in the project (pattern_brain/, tests/, dashboard/)
    so the dashboard can show ANY file, not just the model nodes. Each entry:
    path, lines, kind, and the first docstring/comment line as a summary."""
    import ast
    groups = {"pattern_brain": "core", "tests": "tests", "dashboard": "dashboard"}
    out = []
    total_lines = 0
    for top, kind in groups.items():
        base = os.path.join(ROOT, top)
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv")]
            for fn in sorted(files):
                if not (fn.endswith(".py") or fn.endswith(".html")):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                try:
                    text = open(full, encoding="utf-8").read()
                except Exception:
                    continue
                n = text.count("\n") + 1
                total_lines += n
                summary = ""
                if fn.endswith(".py"):
                    try:
                        summary = (ast.get_docstring(ast.parse(text)) or "").split("\n")[0]
                    except Exception:
                        summary = ""
                if not summary:
                    for line in text.splitlines():
                        s = line.strip().lstrip("#<!-").strip()
                        if s:
                            summary = s[:120]; break
                out.append({"path": rel, "lines": n, "kind": kind, "summary": summary[:160]})
    out.sort(key=lambda f: (f["kind"], f["path"]))
    return {"files": out, "n_files": len(out), "total_lines": total_lines}


@lru_cache(maxsize=1)
def _overview_payload() -> dict:
    """Everything at a glance: live counts for every subsystem + build status, so
    opening the dashboard shows the WHOLE system rather than one 4-node pathway."""
    bank = pb.default_bank()
    from collections import Counter
    layer_counts = dict(Counter(n.layer for n in bank))
    files = _files_payload()
    try:
        import pattern_brain.nodes.deep as _d
        torch_on = _d.TORCH_AVAILABLE
    except Exception:
        torch_on = False
    n_test_suites = sum(1 for f in files["files"] if f["path"].startswith("tests/")
                        and os.path.basename(f["path"]).startswith("test_"))
    subsystems = [
        {"name": "Model Bank", "tab": "bank", "count": len(bank),
         "desc": f"{len(bank)} models across {len(layer_counts)} functional layers"},
        {"name": "Connector v0", "tab": "living", "count": len(pb.DEFAULT_PATHWAY),
         "desc": "fixed hardcoded pathway, run hop-by-hop"},
        {"name": "Routing v1", "tab": "living", "count": len(pb.default_bank()),
         "desc": "LLM-tool-calling router decides the pathway dynamically"},
        {"name": "Stock Adapter", "tab": "adapter", "count": 4,
         "desc": "candles -> generic (T,D); the only domain-specific code"},
        {"name": "Interlingua", "tab": "living", "count": len(pb.catalog()),
         "desc": f"versioned belief contract (v{pb.INTERLINGUA_VERSION})"},
        {"name": "Evaluator", "tab": "evaluator", "count": 5,
         "desc": "5-layer admission gate (walk-forward, PSR/DSR, UCB, Pareto/PBO, anchors)"},
        {"name": "Evolution F1/2/3", "tab": "evolution", "count": 3,
         "desc": "model / pathway / algorithm evolution, gated by the Evaluator"},
        {"name": "Pathway Reputation", "tab": "pathways", "count": 0,
         "desc": "the graph IS the memory: pathways/edges scored + ranked (Block 17)"},
    ]
    build = [
        {"step": "1 Model bank", "done": True},
        {"step": "2 Connector v0 + dashboard", "done": True},
        {"step": "3 Stock adapter", "done": True},
        {"step": "4 LLM-tool-calling routing", "done": True},
        {"step": "5 Interlingua versioning", "done": True},
        {"step": "6 Bank expansion (123, +PyTorch)", "done": True},
        {"step": "Final: Evaluator + Features 1/2/3", "done": True},
    ]
    return {
        "models": len(bank), "layers": layer_counts, "belief_types": len(pb.catalog()),
        "n_files": files["n_files"], "total_lines": files["total_lines"],
        "n_test_suites": n_test_suites, "torch": torch_on,
        "interlingua_version": pb.INTERLINGUA_VERSION,
        "subsystems": subsystems, "build": build,
    }


def _ar_edge(T=420, coef=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = coef * x[t - 1] + rng.normal(0, 0.3)
    return np.column_stack([np.cumsum(x) + 50.0, rng.normal(0, 1, T)])


@lru_cache(maxsize=1)
def _evaluator_payload() -> dict:
    """The 5-layer gate made visible: an edge candidate vs a noise candidate, the
    walk-forward folds, and each one's significance verdict."""
    X = _ar_edge()
    ev = pb.Evaluator(n_splits=5, embargo=0.02, psr_threshold=0.9, dsr_threshold=0.9)
    splits = pb.purged_walk_forward_splits(X.shape[0], 5, 0.02)

    def edge(train, test):
        rng = np.random.default_rng(int(abs(test.sum())) % 9999)
        return 0.45 + rng.normal(0, 1.0, test.shape[0])

    def noise(train, test):
        rng = np.random.default_rng(int(abs(test.sum())) % 9999 + 1)
        return rng.normal(0, 1.0, test.shape[0])

    rep_e = ev.evaluate_candidate(edge, X)
    rep_n = ev.evaluate_candidate(noise, X)
    return {
        "folds": [{"train": int(len(tr)), "test_lo": int(te[0]), "test_hi": int(te[-1])}
                  for tr, te in splits],
        "n_samples": int(X.shape[0]),
        "edge": rep_e.to_dict(), "noise": rep_n.to_dict(),
        "layers": ["L0 purged walk-forward", "L1 PSR/DSR/minTRL", "L2 discounted-UCB",
                   "L3 NSGA-II Pareto + CSCV/PBO", "L4 anchor episodes"],
    }


@lru_cache(maxsize=1)
def _evolution_payload() -> dict:
    """Run all three evolution engines (small) on edge data and return the
    Evolution feed (born -> tested -> promoted) + best genome for each."""
    X = _ar_edge(T=700, coef=0.65)
    feats = {}
    for Eng in (pb.ModelEvolver, pb.PathwayEvolver, pb.AlgorithmEvolver):
        res = Eng(population=8, generations=3, seed=1).run(X)
        hist = res.history[-24:]
        feats[res.feature] = {
            "best": res.best.to_dict() if res.best else None,
            "admitted": len(res.admitted),
            "events": [{"event": h["event"], "origin": h.get("origin"),
                        "genome": h.get("genome"),
                        "ucb": (h["report"]["ucb"] if h.get("report") else None),
                        "passed": (h["report"]["passed"] if h.get("report") else None)}
                       for h in hist],
        }
    return {"features": feats}


@lru_cache(maxsize=1)
def _reputation_payload() -> dict:
    """Closing the loop (Block 17): run pathway evolution so its reputation store
    fills, then return the Pathway Leaderboard + edge reputations (§8 view #3)."""
    X = _ar_edge(T=600, coef=0.65)
    ev = pb.PathwayEvolver(population=8, generations=3, seed=1)
    ev.run(X)
    return ev.reputation.to_dict(top=15)


@app.get("/api/reputation")
def reputation() -> JSONResponse:
    return JSONResponse(_reputation_payload())


@app.get("/api/files")
def files() -> JSONResponse:
    return JSONResponse(_files_payload())


@app.get("/api/overview")
def overview() -> JSONResponse:
    return JSONResponse(_overview_payload())


def _codegen_quality_payload() -> dict:
    """Parse the code-skills benchmark history (code_skills_test/SCORECARD_HISTORY.md)
    so the dashboard can show whether generated-code quality is holding/drifting as the
    model bank grows. Each row: date, easy-panel EXCELLENT count, per-model verdicts,
    and the three hard-regime contract A/B transitions (scaling/separability/drift)."""
    path = os.path.join(ROOT, "code_skills_test", "SCORECARD_HISTORY.md")
    cols = ["date", "easy_excellent", "per_model", "scaling", "separability",
            "drift", "robust"]
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("|") or line.startswith("| date") \
                        or set(line) <= set("|-: "):
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= len(cols):
                    rows.append(dict(zip(cols, cells[:len(cols)])))
    except FileNotFoundError:
        return {"available": False,
                "note": "No benchmark history yet — run "
                        "`python3 code_skills_test/run_benchmark.py`."}
    return {"available": True, "n_runs": len(rows), "rows": rows,
            "latest": rows[-1] if rows else None}


@app.get("/api/codegen_quality")
def codegen_quality() -> JSONResponse:
    return JSONResponse(_codegen_quality_payload())


def _verify_blocks_payload() -> dict:
    """The ML Engineer Agent's cross-session Phase-4 'avoid-list': model families and
    features that repeatedly failed the verification gate (agent_state/verify_blocks.json).
    Makes the agent's learned memory human-inspectable."""
    import json as _json
    path = os.path.join(ROOT, "agent_state", "verify_blocks.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = _json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {"available": False,
                "note": "No Phase-4 blocks recorded yet — the agent hasn't had a "
                        "model fail verification."}
    by_family = sorted(d.get("by_family", {}).items(), key=lambda kv: -kv[1])
    by_feature = sorted(d.get("by_feature", {}).items(), key=lambda kv: -kv[1])
    return {"available": True,
            "by_family": [{"family": k, "blocks": v} for k, v in by_family],
            "by_feature": [{"feature": k, "blocks": v} for k, v in by_feature],
            "total_blocks": sum(d.get("by_family", {}).values())}


@app.get("/api/verify_blocks")
def verify_blocks() -> JSONResponse:
    return JSONResponse(_verify_blocks_payload())


@app.get("/api/evaluator")
def evaluator() -> JSONResponse:
    return JSONResponse(_evaluator_payload())


@app.get("/api/evolution")
def evolution() -> JSONResponse:
    return JSONResponse(_evolution_payload())


@app.get("/api/coverage")
def coverage_endpoint() -> JSONResponse:
    """Catalog-vs-built coverage meter (PLAN §11): per-family ✅/🟡/❌ toward ~500."""
    return JSONResponse(pb.coverage())


@lru_cache(maxsize=1)
def _concept_bank_payload() -> dict:
    """CONCEPT_EQUATION_BANK.md build status (the 'what's left to implement' menu).

    Parses the committed status markers via tools/reconcile_concept_bank.py::summarize —
    the doc is the single source of truth, this just surfaces it on the Coverage tab.
    """
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.reconcile_concept_bank import summarize
    return summarize()


@app.get("/api/concept_bank")
def concept_bank_endpoint() -> JSONResponse:
    """Concept-equation-bank build status: built vs the open 'to implement' list."""
    return JSONResponse(_concept_bank_payload())


@lru_cache(maxsize=1)
def _leaderboard_payload() -> dict:
    """Phase-8 §12 slice 6: the Stacked-DAG leaderboard. Shows the persistent
    leaderboard if the search has populated it; otherwise scores a couple of small
    demo DAGs on a synthetic series (in-memory) so the tab is alive before slice 7."""
    from pattern_brain.leaderboard import Leaderboard
    from pattern_brain.network import DAGSpec, StackedDAG
    real = Leaderboard()
    if real.count() > 0:
        out = {"demo": False, "summary": real.summary(), "top": real.top(15)}
        real.close()
        return out
    real.close()
    rng = np.random.default_rng(3)
    a = np.zeros(150)
    for t in range(1, 150):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    X = a.reshape(-1, 1)
    demo = Leaderboard(":memory:")
    for spec in (DAGSpec([["drift_forecast"]], "mixture"),
                 DAGSpec([["drift_forecast", "theta_forecast"]], "mixture"),
                 DAGSpec([["drift_forecast", "theta_forecast", "naive_mean_forecast"]], "stacked")):
        demo.record(spec, StackedDAG(spec, min_train=40, n_samples=80).score(X), dataset_id="demo_ar1")
    out = {"demo": True, "summary": demo.summary(), "top": demo.top(15)}
    demo.close()
    return out


@app.get("/api/leaderboard")
def leaderboard_endpoint() -> JSONResponse:
    """Phase-8 Stacked-DAG leaderboard: which network combinations score best."""
    return JSONResponse(_leaderboard_payload())


@lru_cache(maxsize=1)
def _capstone_payload() -> dict:
    """Phase-8 §12.1 capstone: a freeze→forward-test with per-step predicted
    quantiles for the fan chart (small synthetic run so the tab is fast/offline;
    the live crypto capstone is `adapters.crypto.run_crypto_capstone`)."""
    from pattern_brain.capstone import run_capstone
    rng = np.random.default_rng(7)
    n = 240
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.55 * r[t - 1] + rng.normal(scale=0.4)
    X = np.column_stack([r, np.cumsum(r), rng.normal(size=n)])
    rep = run_capstone(X, holdout_frac=0.3, strategy="random", budget=6,
                       base_pool=["drift_forecast", "theta_forecast", "naive_mean_forecast"],
                       min_train=45, n_samples=80, seed=0, return_paths=True,
                       meta={"symbol": "DEMO/AR1", "source": "synthetic", "target": "return"})
    try:                                                # persist so the chat's read_capstone tool can see it
        import json as _json
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "agent_state"), exist_ok=True)
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "agent_state", "last_capstone.json"), "w") as fh:
            _json.dump(rep, fh)
    except Exception:
        pass
    return rep


@app.get("/api/capstone")
def capstone_endpoint() -> JSONResponse:
    """The freeze→forward-test acceptance result + per-step predicted-vs-realized paths."""
    return JSONResponse(_capstone_payload())


# --- per-node OOF skill (slice 4): which of the modules actually forecast? --------
# Computed in a BACKGROUND THREAD (the sweep is slow) so this endpoint never blocks
# the single-process server; the frontend polls until ready.
_NODE_SKILL = {"ready": False, "started": False, "skills": {}, "n_total": 0, "n_done": 0}
_NODE_SKILL_LOCK = threading.Lock()


def _compute_node_skill() -> None:
    from pattern_brain.oof import OOFHarness, forecast_node_types
    rng = np.random.default_rng(5)
    n = 160
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    X = a.reshape(-1, 1)
    h = OOFHarness(min_train=40, stride=5, n_samples=50, seed=0)
    try:
        nts = forecast_node_types()
    except Exception:
        nts = []
    _NODE_SKILL["n_total"] = len(nts)
    for nt in nts:
        try:
            s = h.score_node(nt, X)
            sk, cr = s["crps_skill"], s["mean_crps"]
            _NODE_SKILL["skills"][nt] = {
                "skill": (float(sk) if sk == sk else None),
                "crps": (float(cr) if cr == cr else None), "n": int(s["n"])}
        except Exception:
            pass
        _NODE_SKILL["n_done"] += 1
    _NODE_SKILL["ready"] = True


@app.get("/api/node_skill")
def node_skill_endpoint() -> JSONResponse:
    """Per-forecast-node out-of-fold CRPS-skill (background-computed, polled).
    Non-forecast modules simply have no entry (they contribute as features, not forecasts)."""
    with _NODE_SKILL_LOCK:
        if not _NODE_SKILL["started"]:
            _NODE_SKILL["started"] = True
            threading.Thread(target=_compute_node_skill, daemon=True).start()
    return JSONResponse({k: _NODE_SKILL[k] for k in ("ready", "n_total", "n_done", "skills")})


# --------------------------------------------------- ML Engineer Agent (ENG-4)
_AGENT = None
# RLock (reentrant) so a helper that already holds it can call _agent() without
# self-deadlocking. Used ONLY to guard lazy singleton CREATION — never held during
# a long-running agent operation (those use _RUN_LOCK, below), so the read-only
# status/knowledge/advisor endpoints the ML Engineer tab polls never block behind
# a heavy run/confirm.
_AGENT_LOCK = threading.RLock()
_RUN_LOCK = threading.Lock()         # serializes heavy agent runs (step/confirm) off the read path


_AGENT_BACKEND = None        # name of the resolved chat backend (for telemetry/persistence)


def _agent():
    """Lazy singleton agent shared across requests. Wired to a real LLM text
    backend if one is configured (cloud key or Ollama), else the offline templated
    chat — the same graceful degradation as everywhere else."""
    global _AGENT, _AGENT_BACKEND
    with _AGENT_LOCK:
        if _AGENT is None:
            from pattern_brain.agent import MLEngineerAgent
            # reasoning model first (Cerebras gpt-oss-120b) so the chat/advisor
            # deliberate instead of blurting — owner asked it to "think" (2026-06-23).
            # resilient_* re-resolves per call, so a 429 on the reasoner falls through
            # to the next free cloud instead of erroring the chat.
            chat, _name = pb.resilient_text_completer(prefer=pb.CHAT_PREFER)
            _AGENT_BACKEND = _name
            _AGENT = MLEngineerAgent(llm_chat=chat)
            _AGENT.ensure_books()      # self-populate the book vector DB in the background
        return _AGENT


@app.post("/api/agent/ingest_books")
def agent_ingest_books() -> JSONResponse:
    """Convert the curated ML-engineering book corpus into the vector DB (sync)."""
    res = _agent().ingest_books()
    return JSONResponse({"ingested": res, "stats": _agent().toolbox.knowledge.stats()})


# --------------------------------- Ideation & Research Advisor (Rule 26 / §10)
_ADVISOR = None


def _advisor():
    global _ADVISOR
    a = _agent()                                         # get the agent FIRST (its own lock)
    with _AGENT_LOCK:
        if _ADVISOR is None:
            from pattern_brain.agent import IdeationAdvisor
            _ADVISOR = IdeationAdvisor(toolbox=a.toolbox, llm_chat=a.llm_chat)
        return _ADVISOR


@app.get("/api/advisor/ideas")
def advisor_ideas() -> JSONResponse:
    adv = _advisor()
    return JSONResponse({"ideas": adv.list_ideas(), "context": adv.gather_context()})


@app.post("/api/advisor/generate")
def advisor_generate() -> JSONResponse:
    added = _advisor().generate(n=5)
    return JSONResponse({"added": [i.to_dict() for i in added],
                         "ideas": _advisor().list_ideas()})


@app.post("/api/advisor/status")
async def advisor_status(request: Request) -> JSONResponse:
    body = await request.json()
    ok = _advisor().set_status(body.get("id", ""), body.get("status", "proposed"))
    return JSONResponse({"ok": ok, "ideas": _advisor().list_ideas()})


@app.get("/api/knowledge")
def knowledge() -> JSONResponse:
    """ENG-1 view data: book manifest + knowledge-base stats + LLM engine status."""
    a = _agent()
    return JSONResponse({"stats": a.toolbox.knowledge.stats(),
                         "books": pb.book_manifest(),
                         "llm": pb.llm_backend_status()})


@app.get("/api/agent/status")
def agent_status() -> JSONResponse:
    return JSONResponse(_agent().status())


@app.post("/api/agent/step")
def agent_step() -> JSONResponse:
    """Run ONE OODA loop step (Observe→Orient→Decide→Act→Rank/Record)."""
    a = _agent()
    with _RUN_LOCK:                                       # heavy run — must NOT block read endpoints
        r = a.step()
    return JSONResponse({"result": r.to_dict(), "status": a.status()})


def _galton_payload(rows: int = 12, balls: int = 2000, seed: int = 0) -> dict:
    """A Galton board (bean machine): drop `balls` through `rows` of pegs (each a
    50/50 left/right bounce) and count the bins. The histogram is Binomial(rows,
    ½) and converges to a Normal — the Central Limit Theorem you can watch. This is
    the intuition pump for the probability layer (individual chance -> collective law)."""
    import math
    rows = max(2, min(int(rows), 30))
    balls = max(10, min(int(balls), 200_000))
    rng = np.random.default_rng(seed)
    landing = rng.integers(0, 2, size=(balls, rows)).sum(axis=1)   # rights per ball = bin
    counts = np.bincount(landing, minlength=rows + 1)[: rows + 1]
    bins = list(range(rows + 1))
    binom = [math.comb(rows, k) / (2 ** rows) * balls for k in bins]   # exact expectation
    mean, std = rows / 2.0, math.sqrt(rows * 0.25)
    normal = [balls * (1.0 / (std * math.sqrt(2 * math.pi)))
              * math.exp(-0.5 * ((k - mean) / std) ** 2) for k in bins]
    return {"rows": rows, "balls": balls, "bins": bins, "counts": counts.tolist(),
            "binomial": [round(b, 2) for b in binom], "normal": [round(v, 2) for v in normal],
            "mean": mean, "std": round(std, 3)}


@app.get("/api/galton")
def galton(rows: int = 12, balls: int = 2000) -> JSONResponse:
    import random as _r
    return JSONResponse(_galton_payload(rows, balls, seed=_r.randint(0, 10 ** 9)))


@app.post("/api/agent/chat")
async def agent_chat(request: Request) -> JSONResponse:
    """Converse with the agent (the owner's 'talk to it like this session').
    May return a ``pending`` write action (idea 2) the user must Confirm to run."""
    body = await request.json()
    message = (body or {}).get("message", "")
    history = (body or {}).get("history", [])
    session = (body or {}).get("session", "default") or "default"
    effort = (body or {}).get("reasoning_effort")        # low|medium|high|None
    a = _agent()
    _chat_log(session, "user", message)
    # chat() blocks on LLM HTTP calls (a bounded ReAct loop). Run it OFF the event
    # loop so one slow/hung backend can't freeze the whole dashboard, and cap the
    # total wait so the endpoint ALWAYS returns (the agent's own deadline is lower).
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(lambda: a.chat(message, history, reasoning_effort=effort)),
            timeout=70)
    except asyncio.TimeoutError:
        return JSONResponse({"reply": "⏳ The agent took too long to respond (a model backend may be "
                             "slow right now). Please try again — simpler questions answer fastest.",
                             "reasoning": "", "pending": None})
    reasoning = getattr(a, "last_reasoning", "") or ""
    backend = getattr(a.llm_chat, "last_backend", None) or _AGENT_BACKEND or ""
    _chat_log(session, "assistant", reply, reasoning=reasoning, backend=backend)
    return JSONResponse({"reply": reply, "reasoning": reasoning, "backend": backend,
                         "pending": a.pending_action()})


@app.get("/api/agent/backends")
def agent_backends() -> JSONResponse:
    """Reasoner health for the chat header badge: which backend the chat prefers,
    whether it's currently rate-limited (in cooldown), and the full fallback chain."""
    cds = pb.provider_cooldowns()
    reasoner = pb.CHAT_PREFER[0] if pb.CHAT_PREFER else None
    return JSONResponse({
        "reasoner": reasoner,
        "reasoner_cooldown": cds.get(reasoner, 0.0),     # seconds remaining; 0 = live
        "cooldowns": cds,
        "chain": list(pb.CHAT_PREFER),
        "available": [p["name"] for p in pb.available_cloud_providers()],
    })


@app.get("/api/agent/chat_sessions")
def agent_chat_sessions() -> JSONResponse:
    """List past chat sessions (id, auto-title, turn count, last-activity) for the
    replay UI. The title is the session's first user message (idea: titles)."""
    try:
        with _CHAT_DB_LOCK:
            conn = _chat_db()
            rows = conn.execute(
                "SELECT c.session, COUNT(*) n, MAX(c.ts) last, "
                "  (SELECT content FROM chat_log w WHERE w.session=c.session AND w.role='user' "
                "   ORDER BY w.id ASC LIMIT 1) title "
                "FROM chat_log c GROUP BY c.session ORDER BY last DESC LIMIT 100").fetchall()
            conn.close()
        def _title(t):
            t = (t or "").strip().replace("\n", " ")
            return (t[:60] + "…") if len(t) > 60 else (t or "(empty)")
        return JSONResponse({"sessions": [
            {"session": r[0], "turns": r[1], "last": r[2], "title": _title(r[3])} for r in rows]})
    except Exception as e:
        return JSONResponse({"sessions": [], "error": str(e)})


@app.post("/api/agent/chat_delete")
async def agent_chat_delete(request: Request) -> JSONResponse:
    """Delete one chat session's transcript (idea: delete). Keeps the replay list tidy."""
    body = await request.json()
    sid = (body or {}).get("session", "")
    if not sid:
        return JSONResponse({"ok": False, "error": "no session"}, status_code=400)
    try:
        with _CHAT_DB_LOCK:
            conn = _chat_db()
            n = conn.execute("DELETE FROM chat_log WHERE session=?", (sid,)).rowcount
            conn.commit()
            conn.close()
        return JSONResponse({"ok": True, "deleted": n})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/agent/chat_history")
def agent_chat_history(session: str = "default", limit: int = 200) -> JSONResponse:
    """Return the persisted transcript for one session (idea 2: review/replay)."""
    try:
        with _CHAT_DB_LOCK:
            conn = _chat_db()
            rows = conn.execute(
                "SELECT ts, role, content, reasoning, backend FROM chat_log "
                "WHERE session=? ORDER BY id ASC LIMIT ?",
                (session or "default", max(1, min(int(limit), 1000)))).fetchall()
            conn.close()
        msgs = [{"ts": r[0], "role": r[1], "content": r[2], "reasoning": r[3], "backend": r[4]} for r in rows]
        return JSONResponse({"session": session, "messages": msgs})
    except Exception as e:
        return JSONResponse({"session": session, "messages": [], "error": str(e)})


# --------------------------------- async job registry (idea 1: non-blocking runs)
import time as _time
import uuid as _uuid
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOBS_MAX = 50


def _start_job(fn) -> str:
    """Run a blocking ``fn(report)`` in a daemon thread; track status/progress/result
    by job id so the UI can poll instead of waiting on one long request (idea 1).
    ``fn`` receives a ``report(progress_dict)`` callback for rung-level progress."""
    jid = _uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        if len(_JOBS) >= _JOBS_MAX:                       # drop oldest finished jobs
            for k in [k for k, v in sorted(_JOBS.items(), key=lambda kv: kv[1].get("started", 0))
                      if v.get("status") != "running"][:10]:
                _JOBS.pop(k, None)
        _JOBS[jid] = {"status": "running", "started": _time.time(), "result": None, "progress": None}

    def report(p):
        with _JOBS_LOCK:
            if jid in _JOBS:
                _JOBS[jid]["progress"] = p

    def _work():
        try:
            res = fn(report)
            with _JOBS_LOCK:
                _JOBS[jid].update(status="done", result=res, ended=_time.time())
        except Exception as e:                            # surface failures to the poller
            with _JOBS_LOCK:
                _JOBS[jid].update(status="error", error=str(e), ended=_time.time())
    threading.Thread(target=_work, daemon=True).start()
    return jid


@app.get("/api/agent/job/{jid}")
def agent_job(jid: str) -> JSONResponse:
    """Poll a background job's status/result (idea 1)."""
    with _JOBS_LOCK:
        j = _JOBS.get(jid)
        if j is None:
            return JSONResponse({"status": "unknown"}, status_code=404)
        out = {"status": j["status"],
               "elapsed": round((j.get("ended") or _time.time()) - j["started"], 1),
               "progress": j.get("progress")}
        if j["status"] == "done":
            out["result"] = j["result"]
        elif j["status"] == "error":
            out["error"] = j.get("error")
    return JSONResponse(out)


@app.post("/api/agent/confirm")
async def agent_confirm(request: Request) -> JSONResponse:
    """Confirm (or cancel) a staged write action (idea 2). A real RUN is dispatched
    as a BACKGROUND JOB (idea 1) — the endpoint returns a ``job_id`` immediately and
    the UI polls ``/api/agent/job/<id>`` for progress, so the chat never blocks on a
    multi-second compute. Cancels / no-op confirms return instantly."""
    body = await request.json()
    aid = (body or {}).get("id", "")
    approve = bool((body or {}).get("approve", True))
    a = _agent()
    pa = a.pending_action()
    if not approve or pa is None or pa.get("id") != aid:     # instant: no heavy compute
        return JSONResponse({**a.confirm_action(aid, approve=approve), "async": False})

    def _run(report):
        with _RUN_LOCK:                                      # serialize heavy runs; never block reads
            return a.confirm_action(aid, approve=True, progress_cb=report)
    jid = _start_job(_run)
    return JSONResponse({"async": True, "job_id": jid, "status": "running",
                         "desc": pa.get("desc")})


@app.get("/api/archive")
def archive_endpoint() -> JSONResponse:
    """W4 Quality-Diversity archive: the best pathway combo per niche
    (symbol × timeframe × volatility-regime) — the owner's 'save all the top
    network connection combinations'."""
    from pattern_brain.archive import QDArchive
    arch = QDArchive()
    return JSONResponse({"entries": arch.all(), "stats": arch.stats()})


@app.get("/api/compute")
def compute_status() -> JSONResponse:
    """W2 Adaptive Compute Manager governor view (live CPU/RAM/workers) for the
    dashboard compute tile."""
    from pattern_brain.compute import ComputeManager
    return JSONResponse(ComputeManager().stats())


@app.post("/api/agent/consolidate")
def agent_consolidate() -> JSONResponse:
    """Promote good chat Q&A into first-class knowledge (idea 3)."""
    a = _agent()
    res = a.consolidate()
    return JSONResponse({"result": res, "stats": a.toolbox.knowledge.stats()})


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    """Stream the pathway run hop-by-hop so the living graph animates belief-flow.
    Protocol: a ``start`` frame (the pathway), one ``hop`` frame per node as it
    fires, then a ``done`` frame with the terminal output."""
    await ws.accept()
    try:
        conn = pb.default_connector()
        X = synthetic_sequence()
        await ws.send_json({"event": "start", "pathway": conn.pathway})
        last = None
        for hop in conn.iter_run(X):
            await ws.send_json({"event": "hop", "hop": hop.to_dict()})
            last = hop
            await asyncio.sleep(0.6)  # pace the animation; the math itself is instant
        await ws.send_json({
            "event": "done",
            "output": last.belief.to_dict() if last else None,
        })
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.on_event("startup")
def _prewarm() -> None:
    """Build the expensive caches (evolution runs ~15s; conformance processes the
    whole bank) in a background thread at boot, so a user's first click on those
    tabs is instant rather than a long stall."""
    import threading

    def warm():
        for fn in (_overview_payload, _files_payload, _bank_payload, _route_payload,
                   _conformance_payload, _evaluator_payload, _evolution_payload,
                   _reputation_payload):
            try:
                fn()
            except Exception:
                pass
    threading.Thread(target=warm, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    # Secure-by-default: localhost only. Set PB_DASH_HOST=0.0.0.0 to expose on the
    # VPS public IP (the dashboard + chat spend API keys, so only do this behind a
    # firewall). Port overridable via PB_DASH_PORT.
    host = os.environ.get("PB_DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("PB_DASH_PORT", "8077"))
    uvicorn.run(app, host=host, port=port)
