"""Knowledge base / vector DB — the ML Engineer Agent's memory (PLAN.md §9, ENG-1).

Two jobs, both grounded in the Block-56 deep sweep:

1. **A book/literature knowledge base** (Sakana AI-Scientist's "literature search"
   step; the 2026 RAG two-phase pattern: index offline, retrieve online). A
   curated :data:`BOOK_MANIFEST` of *open-access* ML-engineering resources is
   ingested (fetch -> chunk -> embed -> store); the agent retrieves relevant
   passages to ground each experiment idea.
2. **Tiered archival memory** (Letta/MemGPT — the 2026 agent-memory standard):
   the agent writes its own findings with :meth:`KnowledgeBase.remember` and
   reloads them with :meth:`recall`, so the infinite loop survives restarts
   (the fix for "LLM context is stateless: write at session end, inject at start").

Design discipline:
* **Domain-agnostic (PLAN.md §0 / Rule 23):** generic text in, generic passages
  out. Zero candle/order-book coupling. The book corpus is general ML-engineering
  knowledge — domain-independent by nature.
* **Light stack first (§0b):** the default embedder is a dependency-free hashing
  bag-of-words vectorizer (numpy only) and the store is numpy-cosine — it runs on
  this box instantly with no model download. A real CPU sentence-transformer
  (nomic-embed-text / bge-m3, per the sweep) drops in via :class:`STEmbedder`
  *only if installed*; chromadb/qdrant are the documented scale path, not a
  starting requirement.
* **Everything stays inside the project folder (Rule 1 clarification, 2026-06-21):**
  the store persists under ``pattern-brain/knowledge_store/`` — never /tmp or a
  home cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------- locations
# Project root = the folder containing this package's parent (pattern-brain/).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)
DEFAULT_STORE_DIR = os.path.join(PROJECT_ROOT, "knowledge_store")


def _inside_project(path: str) -> bool:
    """Rule 1 guard: a store path must live inside the project folder."""
    return os.path.abspath(path).startswith(PROJECT_ROOT + os.sep) or \
        os.path.abspath(path) == PROJECT_ROOT


# ------------------------------------------------------------------ chunking
_WORD = re.compile(r"\w+")


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> List[str]:
    """Recursive-style word chunking — ~``size`` words per chunk with ``overlap``
    words of carry-over (the 2026 RAG consensus: ~300-512 tokens, 10-20% overlap,
    approximated here in words to stay dependency-free). Returns non-empty chunks."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# ----------------------------------------------------------------- embedders
class Embedder:
    """Common interface (mirrors the bank's node contract spirit): text -> vector."""
    dim: int

    def embed(self, text: str) -> np.ndarray:               # pragma: no cover - abstract
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))


class HashingEmbedder(Embedder):
    """Dependency-free TF bag-of-words via feature hashing, L2-normalized — the
    default so the KB runs with zero downloads on a no-GPU box (§0b). Deterministic
    (the hash is content-seeded), so tests + retrieval are reproducible."""

    name = "hashing-bow"

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _WORD.findall(text.lower()):
            v[self._bucket(tok)] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class STEmbedder(Embedder):
    """Optional CPU sentence-transformer (nomic-embed-text / bge-m3 per the sweep).
    Used only if ``sentence-transformers`` is installed; otherwise the caller keeps
    :class:`HashingEmbedder` — same optional-dependency discipline as the torch nodes."""

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # optional dep
        self.name = f"st:{model}"
        self._m = SentenceTransformer(model, trust_remote_code=True, device="cpu")
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        v = np.asarray(self._m.encode([text], normalize_embeddings=True)[0], dtype=np.float64)
        return v


class OllamaEmbedder(Embedder):
    """CPU embeddings via a (project-local) Ollama server running an embedding
    model — `nomic-embed-text` by default (the sweep's CPU throughput winner).
    Stdlib only; used if Ollama is reachable, else the caller falls back."""

    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None) -> None:
        import urllib.request as _u
        self._u = _u
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.name = f"ollama:{model}"
        self.dim = len(self.embed("dimension probe"))    # discover dim from a real call

    def embed(self, text: str) -> np.ndarray:
        req = self._u.Request(self.host + "/api/embeddings",
                              data=json.dumps({"model": self.model, "prompt": text}).encode(),
                              headers={"Content-Type": "application/json"})
        with self._u.urlopen(req, timeout=30) as r:
            v = np.asarray(json.loads(r.read().decode())["embedding"], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def default_embedder(dim: int = 512) -> Embedder:
    """Prefer the project-local Ollama embedder (nomic-embed-text) if reachable,
    then a CPU sentence-transformer if installed, else the dependency-free hashing
    embedder (always works)."""
    try:
        return OllamaEmbedder()
    except Exception:
        pass
    try:
        return STEmbedder()
    except Exception:
        return HashingEmbedder(dim=dim)


# --------------------------------------------------------------- vector store
@dataclass
class Passage:
    id: str
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Retrieval:
    passage: Passage
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(float(self.score), 4), **self.passage.to_dict()}


class VectorStore:
    """A light numpy-cosine vector store with on-disk persistence (npz + json),
    written INSIDE the project folder (Rule 1). Vectors are stored L2-normalized so
    cosine similarity is a single matrix-vector dot product."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._vecs = np.zeros((0, self.dim), dtype=np.float64)
        self._passages: List[Passage] = []
        self._ids: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._passages)

    def add(self, passage: Passage, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"vector dim {vector.shape[0]} != store dim {self.dim}")
        n = np.linalg.norm(vector)
        vector = vector / n if n > 0 else vector
        if passage.id in self._ids:                       # upsert
            self._vecs[self._ids[passage.id]] = vector
            self._passages[self._ids[passage.id]] = passage
            return
        self._ids[passage.id] = len(self._passages)
        self._passages.append(passage)
        self._vecs = np.vstack([self._vecs, vector[None, :]])

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Retrieval]:
        if len(self) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(q)
        q = q / n if n > 0 else q
        sims = self._vecs @ q
        order = np.argsort(-sims)[:max(0, k)]
        return [Retrieval(self._passages[i], float(sims[i])) for i in order]

    def save(self, directory: str = DEFAULT_STORE_DIR, name: str = "store") -> str:
        if not _inside_project(directory):
            raise ValueError(f"Rule 1: store dir must be inside {PROJECT_ROOT}, got {directory}")
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, f"{name}.npz"), vecs=self._vecs)
        with open(os.path.join(directory, f"{name}.json"), "w") as fh:
            json.dump({"dim": self.dim, "passages": [p.to_dict() for p in self._passages]}, fh)
        return directory

    @classmethod
    def load(cls, directory: str = DEFAULT_STORE_DIR, name: str = "store") -> "VectorStore":
        with open(os.path.join(directory, f"{name}.json")) as fh:
            meta = json.load(fh)
        store = cls(dim=int(meta["dim"]))
        store._vecs = np.load(os.path.join(directory, f"{name}.npz"))["vecs"]
        store._passages = [Passage(**p) for p in meta["passages"]]
        store._ids = {p.id: i for i, p in enumerate(store._passages)}
        return store


# ------------------------------------------------------------- book manifest
@dataclass
class BookRef:
    title: str
    url: str
    kind: str                # "open-book" | "reading-list" | "reference"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Curated from the Block-56 sweep — open-access / freely-readable ML-engineering
# resources a student/engineer actually reads. URLs are fetched on demand by
# KnowledgeBase.ingest_url; nothing is downloaded at import time.
BOOK_MANIFEST: List[BookRef] = [
    BookRef("Machine Learning Systems (Harvard/MIT Press, 2026)", "https://mlsysbook.ai/",
            "open-book", "Principles-first ML systems, single machine -> fleet scale"),
    BookRef("Machine Learning Engineering Open Book (Stas Bekman)",
            "https://github.com/stas00/ml-engineering", "open-book",
            "Living book: practical training/deploying LLMs & multimodal at scale"),
    BookRef("Dive into Deep Learning (d2l.ai)", "https://d2l.ai/", "open-book",
            "Interactive, math + code, runnable notebooks"),
    BookRef("Understanding Deep Learning (Simon Prince, free PDF)",
            "https://udlbook.github.io/udlbook/", "open-book", "Modern DL theory + practice"),
    BookRef("Probabilistic Machine Learning (Kevin Murphy, free PDFs)",
            "https://probml.github.io/pml-book/", "reference", "The probabilistic ML reference"),
    BookRef("The Elements of Statistical Learning (free PDF)",
            "https://hastie.su.domains/ElemStatLearn/", "reference", "Classic statistical learning"),
    BookRef("awesome-machine-learning books list (josephmisiti)",
            "https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md",
            "reading-list", "Curated free-book index across ML/DL/RL"),
    BookRef("ML Engineer reading list 2026 (Javarevisited)",
            "https://medium.com/javarevisited/the-machine-learning-engineer-reading-list-for-2026-5-books-that-matter-210d75135931",
            "reading-list", "The 5 books that matter for an ML engineer in 2026"),
]


def book_manifest() -> List[Dict[str, Any]]:
    return [b.to_dict() for b in BOOK_MANIFEST]


# ---------------------------------------------------------------- knowledge base
class KnowledgeBase:
    """Ties an embedder + a vector store into the agent's two memory roles: a
    retrievable book/literature corpus, and a Letta-style archival memory for the
    agent's own findings. Both live in the SAME store (tagged by ``source``)."""

    def __init__(self, embedder: Optional[Embedder] = None,
                 store_dir: str = DEFAULT_STORE_DIR,
                 fetch: Optional[Callable[[str], str]] = None) -> None:
        self.embedder = embedder or HashingEmbedder()   # dependency-free default
        self.store = VectorStore(self.embedder.dim)
        self.store_dir = store_dir
        self._fetch = fetch or _http_get               # injectable for offline tests

    # ---- ingestion (the offline "indexing" phase) ----
    def ingest_text(self, doc_id: str, text: str, *, source: str = "",
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk_size: int = 400, overlap: int = 60) -> int:
        chunks = chunk_text(text, size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            p = Passage(id=f"{doc_id}#{i}", text=ch, source=source or doc_id,
                        metadata=dict(metadata or {}, chunk=i))
            self.store.add(p, self.embedder.embed(ch))
        return len(chunks)

    def ingest_url(self, url: str, *, doc_id: Optional[str] = None,
                   source: str = "", **kw) -> int:
        """Fetch a URL (internet access) and ingest its text. Uses the injected
        ``fetch`` callable (a real HTTP GET by default; a fake in tests)."""
        text = self._fetch(url)
        return self.ingest_text(doc_id or url, text, source=source or url,
                                metadata={"url": url}, **kw)

    def ingest_manifest(self, refs: Optional[Sequence[BookRef]] = None,
                        max_chars: int = 200_000) -> Dict[str, int]:
        """Ingest the curated book manifest (best-effort: a failed fetch is skipped,
        not fatal — the agent logs and moves on, R&D-Agent error-feedback style)."""
        out: Dict[str, int] = {}
        for ref in (refs or BOOK_MANIFEST):
            try:
                text = self._fetch(ref.url)[:max_chars]
                out[ref.title] = self.ingest_text(ref.url, text, source=ref.title,
                                                  metadata={"url": ref.url, "kind": ref.kind})
            except Exception as e:                         # noqa: BLE001
                out[ref.title] = 0
                _ = e
        return out

    # ---- retrieval (the online phase) ----
    def retrieve(self, query: str, k: int = 5,
                 source_prefix: Optional[str] = None) -> List[Retrieval]:
        hits = self.store.search(self.embedder.embed(query), k=k * 3 if source_prefix else k)
        if source_prefix:
            hits = [h for h in hits if h.passage.source.startswith(source_prefix)][:k]
        return hits[:k]

    # ---- Letta-style archival memory (the agent's own work) ----
    def remember(self, note: str, *, kind: str = "finding",
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        pid = f"archival/{kind}/{int(time.time()*1000)}/{len(self.store)}"
        self.store.add(Passage(id=pid, text=note, source="archival",
                               metadata=dict(metadata or {}, kind=kind, ts=time.time())),
                       self.embedder.embed(note))
        return pid

    def recall(self, query: str, k: int = 5) -> List[Retrieval]:
        return self.retrieve(query, k=k, source_prefix="archival")

    # ---- persistence (inside the folder, Rule 1) ----
    def save(self, name: str = "store") -> str:
        return self.store.save(self.store_dir, name=name)

    def stats(self) -> Dict[str, Any]:
        sources: Dict[str, int] = {}
        for p in self.store._passages:
            sources[p.source] = sources.get(p.source, 0) + 1
        return {"passages": len(self.store), "embedder": getattr(self.embedder, "name", "?"),
                "dim": self.embedder.dim, "store_dir": self.store_dir,
                "sources": sources, "archival": sources.get("archival", 0)}


def _http_get(url: str, timeout: float = 20.0) -> str:
    """Minimal internet fetch (stdlib). HTML is returned as-is; the chunker tolerates
    markup well enough for a light KB (Unstructured.io is the optional upgrade)."""
    req = urllib.request.Request(url, headers={"User-Agent": "pattern-brain-kb/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")
