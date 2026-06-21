"""Tests for the knowledge base / vector DB (PLAN.md §9, ENG-1).

Run: python3 tests/test_knowledge.py
Fully offline: the default embedder is dependency-free (numpy hashing BoW), and
URL ingestion is exercised through an injected fake fetcher — no network needed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain import knowledge as kb

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_chunking():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = kb.chunk_text(text, size=400, overlap=60)
    check(len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}")
    check(all(len(c.split()) <= 400 for c in chunks), "a chunk exceeded size")
    # overlap: end of chunk0 reappears at start of chunk1
    tail = chunks[0].split()[-60:]
    head = chunks[1].split()[:60]
    check(tail == head, "overlap words not carried into the next chunk")
    check(kb.chunk_text("", 400, 60) == [], "empty text -> no chunks")
    try:
        kb.chunk_text("a b c", size=10, overlap=10); check(False, "overlap>=size should raise")
    except ValueError:
        pass
    print(f"  chunking: 1000 words -> {len(chunks)} chunks, overlap preserved")


def test_hashing_embedder_deterministic():
    e = kb.HashingEmbedder(dim=256)
    v1, v2 = e.embed("machine learning engineering"), e.embed("machine learning engineering")
    check(v1.shape == (256,), "wrong embedding dim")
    check(np.allclose(v1, v2), "embedder not deterministic")
    check(abs(np.linalg.norm(v1) - 1.0) < 1e-9, "embedding not L2-normalized")
    check(not np.allclose(v1, e.embed("reinforcement learning rewards")), "distinct text -> same vec")
    print("  embedder: deterministic, L2-normalized, distinct texts differ")


def test_vector_store_search_and_upsert():
    e = kb.HashingEmbedder(dim=256)
    store = kb.VectorStore(e.dim)
    docs = {"a": "gradient boosting decision trees xgboost",
            "b": "convolutional neural networks image classification",
            "c": "hidden markov models sequence probability"}
    for k, t in docs.items():
        store.add(kb.Passage(id=k, text=t), e.embed(t))
    check(len(store) == 3, "store length wrong")
    top = store.search(e.embed("xgboost boosted trees"), k=1)
    check(top and top[0].passage.id == "a", f"nearest should be 'a', got {top and top[0].passage.id}")
    # upsert: re-adding same id doesn't grow the store
    store.add(kb.Passage(id="a", text="updated"), e.embed("updated"))
    check(len(store) == 3, "upsert grew the store")
    try:
        store.add(kb.Passage(id="z", text="x"), np.zeros(5)); check(False, "dim mismatch should raise")
    except ValueError:
        pass
    print("  store: cosine search returns nearest; upsert + dim-guard work")


def test_persistence_roundtrip_inside_folder():
    e = kb.HashingEmbedder(dim=128)
    store = kb.VectorStore(e.dim)
    store.add(kb.Passage(id="x", text="alpha beta", source="s", metadata={"n": 1}), e.embed("alpha beta"))
    tmp = tempfile.mkdtemp(dir=kb.PROJECT_ROOT)        # INSIDE the project folder (Rule 1)
    try:
        store.save(tmp, name="t")
        loaded = kb.VectorStore.load(tmp, name="t")
        check(len(loaded) == 1 and loaded._passages[0].metadata["n"] == 1, "roundtrip lost data")
        q = e.embed("alpha beta")
        check(loaded.search(q, 1)[0].passage.id == "x", "loaded store search broken")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # Rule 1 guard: saving outside the project folder is refused
    try:
        store.save("/tmp/should_refuse", name="t"); check(False, "save outside folder should raise")
    except ValueError:
        pass
    print("  persistence: npz+json roundtrip inside folder; outside-folder save refused")


def test_knowledgebase_ingest_retrieve_offline():
    pages = {
        "http://book/ml": "feature engineering pipelines data leakage cross validation " * 40,
        "http://book/dl": "transformers attention backpropagation optimizer scheduling " * 40,
    }
    base = kb.KnowledgeBase(fetch=lambda u: pages[u])   # injected offline fetcher
    n = base.ingest_url("http://book/ml", source="ML Book")
    base.ingest_url("http://book/dl", source="DL Book")
    check(n >= 1, "no chunks ingested")
    hits = base.retrieve("data leakage in cross validation", k=1)
    check(hits and hits[0].passage.source == "ML Book", f"retrieve picked {hits and hits[0].passage.source}")
    stats = base.stats()
    check(stats["passages"] >= 2 and "ML Book" in stats["sources"], "stats missing sources")
    print(f"  KB: ingested 2 docs ({stats['passages']} passages), retrieval routed to the right book")


def test_archival_memory_letta_style():
    base = kb.KnowledgeBase()
    base.remember("Feature-1 mutation of lstm_forecaster beat the AR baseline at depth 2",
                  kind="finding", metadata={"feature": 1})
    base.remember("Pathway difference->kalman_filter->threshold_policy scored UCB 0.41",
                  kind="finding")
    base.ingest_text("noise", "totally unrelated cooking recipe text " * 20, source="other")
    rec = base.recall("which lstm mutation worked", k=2)
    check(rec and rec[0].passage.source == "archival", "recall should only return archival notes")
    check(any("lstm" in r.passage.text for r in rec), "recall didn't surface the lstm finding")
    print("  archival: agent remembers findings + recalls only archival (Letta tier)")


def test_book_manifest_and_locations():
    man = kb.book_manifest()
    check(len(man) >= 6, "manifest too small")
    check(all({"title", "url", "kind"} <= set(b) for b in man), "manifest entry malformed")
    check(any("mlsysbook" in b["url"] for b in man), "expected Machine Learning Systems in manifest")
    check(kb._inside_project(kb.DEFAULT_STORE_DIR), "default store dir not inside project")
    check(not kb._inside_project("/tmp/x"), "guard wrongly allows /tmp")
    print(f"  manifest: {len(man)} open-access ML-eng resources; store dir inside folder")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(kb.PROJECT_ROOT, "pattern_brain", "knowledge.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in knowledge.py: {hits}")
    print("  domain-independence: knowledge.py has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — knowledge base / vector DB (ENG-1): tests")
    print("=" * 70)
    test_chunking()
    test_hashing_embedder_deterministic()
    test_vector_store_search_and_upsert()
    test_persistence_roundtrip_inside_folder()
    test_knowledgebase_ingest_retrieve_offline()
    test_archival_memory_letta_style()
    test_book_manifest_and_locations()
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
