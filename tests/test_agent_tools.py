"""Tests for the ML Engineer Agent tools (PLAN.md §9, ENG-2).

Run: python3 tests/test_agent_tools.py
Fully offline: synthetic datasets, injected web fetcher, dependency-free KB.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.agent import tools as T
from pattern_brain.knowledge import KnowledgeBase, PROJECT_ROOT
from pattern_brain.registry import all_node_types, create

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _box(**kw):
    return T.Toolbox(knowledge=KnowledgeBase(),
                     data_dir=os.path.join(PROJECT_ROOT, "data", "_test"), **kw)


def test_observe_state():
    box = _box()
    st = box.observe_state()
    check(st["bank"]["n_nodes"] >= 100, f"bank too small: {st['bank']['n_nodes']}")
    check("knowledge" in st and "reputation" in st, "observe_state missing sections")
    print(f"  observe: bank={st['bank']['n_nodes']} nodes, "
          f"layers={len(st['bank']['layers'])}, kb passages={st['knowledge']['passages']}")


def test_fetch_dataset_synthetic_inside_folder():
    box = _box()
    try:
        ds = box.fetch_dataset("toy", source="synthetic", T=300, D=2, seed=1)
        check(ds["shape"] == [300, 2], f"wrong shape {ds['shape']}")
        check(os.path.abspath(ds["path"]).startswith(PROJECT_ROOT + os.sep),
              "dataset cached OUTSIDE the project folder (Rule 1 violation)")
        check(os.path.exists(ds["path"]), "dataset file not written")
    finally:
        shutil.rmtree(box.data_dir, ignore_errors=True)
    print("  data: synthetic (300,2) dataset acquired + cached inside the folder")


def test_injected_dataset_loader():
    box = _box(dataset_loaders={"fake": lambda name, **kw: np.ones((50, 3))})
    try:
        ds = box.fetch_dataset("x", source="fake")
        check(ds["shape"] == [50, 3], "injected loader shape wrong")
    finally:
        shutil.rmtree(box.data_dir, ignore_errors=True)
    print("  data: injectable dataset loader works (OpenML/HF/Kaggle path)")


def test_web_fetch_and_download_injected():
    box = _box(fetch=lambda u: b"hello world " * 10)
    try:
        txt = box.web_fetch("http://example.test/page")
        check(txt.startswith("hello world"), "web_fetch text wrong")
        path = box.download_file("http://example.test/file.txt", "file.txt")
        check(os.path.exists(path) and os.path.abspath(path).startswith(PROJECT_ROOT + os.sep),
              "download not inside folder")
        try:
            box.download_file("http://x", "../../escape.txt")  # basename strips path
            check(os.path.abspath(os.path.join(box.data_dir, "escape.txt")).startswith(PROJECT_ROOT),
                  "path traversal not contained")
        except ValueError:
            pass
    finally:
        shutil.rmtree(box.data_dir, ignore_errors=True)
    print("  internet: web_fetch + download (injected) write inside the folder")


def test_run_evolution_feature1():
    box = _box()
    X = T.synthetic_dataset(T=420, D=1, seed=2)
    run = box.run_evolution("feature1_model", X, population=6, generations=2, seed=3)
    s = run.summary()
    check(s["tested"] >= 6, f"too few tested: {s['tested']}")
    check(run.result.best is not None, "no best individual")
    check(any(h["event"] == "tested" for h in run.result.history), "no history events")
    print(f"  evolve: feature1 tested {s['tested']} candidates, admitted {s['admitted']}")


def test_run_evolution_all_three_features():
    box = _box()
    X = T.synthetic_dataset(T=360, D=1, seed=5)
    for feat in ("feature1_model", "feature2_algorithm", "feature3_pathway"):
        run = box.run_evolution(feat, X, population=5, generations=2, seed=1)
        check(run.result.best is not None, f"{feat}: no best")
    # feature3 records pathway reputation onto the shared store
    check(len(box.reputation.leaderboard(top=999)) >= 1,
          "pathway evolution didn't populate reputation")
    print("  evolve: features 1/2/3 all run; feature3 populated pathway reputation")


def test_promote_to_node_includes_in_bank():
    box = _box()
    X = T.synthetic_dataset(T=420, D=1, seed=7)
    before = len(all_node_types())
    run = box.run_evolution("feature1_model", X, population=6, generations=2, seed=7)
    nt = box.promote_to_node(run, only_if_admitted=False)   # force a promotion for the test
    check(nt is not None and nt in all_node_types(), f"promoted node {nt} not in bank")
    check(len(all_node_types()) == before + 1, "bank count didn't grow by exactly 1")
    # the promoted node is a real, usable bank node emitting a forecast belief
    node = create(nt)
    belief = node.predict(X[-32:, :1])
    check(belief.type == "forecast" and "estimate" in belief.payload,
          "promoted node emits a malformed belief")
    # promotion was recorded into archival memory
    check(any("Promoted" in r["text"] for r in box.recall_memory("promoted node", k=5)),
          "promotion not recorded in archival memory")
    print(f"  promote: {nt} registered as a bank node, predicts a forecast, logged to memory")


def test_specs_and_dispatch():
    box = _box()
    specs = box.specs()
    import json
    json.dumps(specs)  # must be serializable
    check(len(specs) >= 7, "too few tool specs")
    out = box.call("observe_state")
    check("bank" in out, "call(observe_state) failed")
    try:
        box.call("rm_rf"); check(False, "unknown tool should raise")
    except ValueError:
        pass
    print(f"  tools: {len(specs)} JSON tool specs; call() dispatch + unknown-tool guard")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(PROJECT_ROOT, "pattern_brain", "agent", "tools.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in tools.py: {hits}")
    print("  domain-independence: agent/tools.py has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — ML Engineer Agent tools (ENG-2): tests")
    print("=" * 70)
    test_observe_state()
    test_fetch_dataset_synthetic_inside_folder()
    test_injected_dataset_loader()
    test_web_fetch_and_download_injected()
    test_run_evolution_feature1()
    test_run_evolution_all_three_features()
    test_promote_to_node_includes_in_bank()
    test_specs_and_dispatch()
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
