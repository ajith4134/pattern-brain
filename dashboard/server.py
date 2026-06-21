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

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

app = FastAPI(title="Pattern Brain — Living Graph (step 2 walking skeleton)")


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


@app.get("/api/bank")
def bank() -> JSONResponse:
    """The model-genome map: one metadata record per registered node, grouped by
    functional layer (the precursor to §8's full genome view)."""
    nodes = [n.metadata() for n in pb.default_bank()]
    return JSONResponse({
        "layers": pb.layers(),
        "n_nodes": len(nodes),
        "nodes": nodes,
    })


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


@app.get("/api/route")
def route() -> JSONResponse:
    """Connector v1 (build step 4): a router DECIDES the pathway hop-by-hop over
    the full bank (offline HeuristicRouter — no LLM/API needed). Returns the
    discovered pathway + the per-hop routing rationale, so the dashboard can show
    *why* each node was chosen, not just a fixed list."""
    routed = pb.default_routed_connector().route(synthetic_sequence())
    return JSONResponse({
        "router": pb.default_routed_connector().router.name,
        "pathway": routed.result.pathway,
        "decisions": [d.to_dict() for d in routed.decisions],
        "output": routed.result.output.to_dict() if routed.result.output else None,
        "trace": routed.result.to_dict(),
    })


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


@app.get("/api/conformance")
def conformance() -> JSONResponse:
    """Step-5 interlingua view: the versioned shared-belief contract made visible.
    Returns the interlingua version + catalog, a conformance report over a default
    pathway run (drift detection), and coherence notes over the full bank (the
    loose types whose payloads vary across emitters)."""
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
    return JSONResponse({
        "version": pb.INTERLINGUA_VERSION,
        "catalog": pb.catalog(),
        "run_report": rep.to_dict(),
        "coherence": coherence,
    })


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8077)
