"""ML Engineer Agent — the tools (PLAN.md §9, ENG-2).

The typed capabilities the agent (and, via the LLM tool-calling boundary, an LLM)
can invoke — "everything an ML engineer does", each behind a small typed surface:

* **observe** the current system state (bank, reputation, evolution, knowledge);
* **retrieve / remember** from the book knowledge base + the agent's own archival
  memory (Letta tier, ENG-1);
* **internet access** — web fetch + file download (into the project's ``data/``);
* **acquire data/tests** — `fetch_dataset` (synthetic always-works default;
  OpenML — standardized train/test splits — optional; injectable for tests);
* **create / mutate models & algorithms** — `run_evolution` drives Features 1/2/3;
* **rank → include in the model nodes** — `promote_to_node` turns a winning
  evolved genome into a registered bank node (the literal "ranking them in order
  for including in the model nodes").

Design discipline:
* **Domain-agnostic (Rule 23):** every tool here operates on generic (T, D)
  arrays / generic text. Stock-specific dataset fetchers belong in `adapters/`,
  NOT here — `fetch_dataset`'s built-in sources (synthetic, OpenML) are
  domain-neutral.
* **All files inside the folder (Rule 1 clarification):** downloads + datasets go
  under ``pattern-brain/data/``; a guard refuses any path outside the folder.
* **No hard new deps (§0b):** OpenML/HuggingFace/Kaggle fetchers are optional and
  injectable; the synthetic source means the agent always has data to train on,
  even fully offline — which keeps the ENG-3 infinite loop resilient.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..belief import Belief
from ..knowledge import KnowledgeBase, PROJECT_ROOT, _inside_project
from ..registry import register, create, all_node_types, by_layer, layers, _REGISTRY
from ..node import Node
from ..reputation import PathwayReputation
from ..evolution import ModelEvolver, PathwayEvolver, AlgorithmEvolver, EvolutionResult, Individual

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

_EVOLVERS = {
    "feature1_model": ModelEvolver,
    "feature2_algorithm": AlgorithmEvolver,
    "feature3_pathway": PathwayEvolver,
    # convenient aliases
    "model": ModelEvolver, "algorithm": AlgorithmEvolver, "pathway": PathwayEvolver,
}


def synthetic_dataset(T: int = 480, D: int = 3, seed: int = 0) -> np.ndarray:
    """An always-available generic (T, D) sequence (a mildly-predictable AR(1)
    process + noise) so the agent can always run the loop — even with no network."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, D))
    for d in range(D):
        e = rng.normal(scale=1.0, size=T)
        for t in range(1, T):
            x[t, d] = 0.6 * x[t - 1, d] + e[t]
    return x


# ----------------------------------------------------- promote a winner -> node
def _make_evolved_node_class(node_type: str, predict_fn: Callable[[np.ndarray], float],
                             confidence: float):
    """Build (don't register) a Node subclass wrapping a fitted evolved predictor.
    It emits the same ``forecast`` belief existing forecasters do, so the bank
    stays interlingua-conformant (ENG-1/Block 46)."""
    conf = float(max(0.0, min(1.0, confidence)))

    class _EvolvedNode(Node):
        layer = "sequence"
        is_transformer = False

        def _predict(self, X: np.ndarray) -> Belief:
            window = np.asarray(X)[:, 0]
            try:
                v = float(predict_fn(window))
                if not np.isfinite(v):
                    v = float(window[-1])
            except Exception:
                v = float(window[-1])
            return Belief(type="forecast",
                          payload={"estimate": v, "next_vector": [v]},
                          confidence=conf)

    _EvolvedNode.node_type = node_type
    _EvolvedNode.__name__ = f"Evolved_{node_type}"
    return _EvolvedNode


@dataclass
class Toolbox:
    """Shared state + the tool methods. One instance is owned by the agent and the
    dashboard, so observations, knowledge, and reputation stay consistent."""

    knowledge: Optional[KnowledgeBase] = None
    reputation: Optional[PathwayReputation] = None
    data_dir: str = DEFAULT_DATA_DIR
    fetch: Optional[Callable[[str], bytes]] = None        # injectable internet GET
    dataset_loaders: Optional[Dict[str, Callable[..., np.ndarray]]] = None
    _promoted: int = 0

    def __post_init__(self) -> None:
        self.knowledge = self.knowledge or KnowledgeBase()
        self.reputation = self.reputation or PathwayReputation()
        self.fetch = self.fetch or _http_get_bytes
        self.dataset_loaders = self.dataset_loaders or {}

    # ----------------------------------------------------------- observe
    def observe_state(self) -> Dict[str, Any]:
        lb = self.reputation.leaderboard(top=5)
        return {
            "bank": {"n_nodes": len(all_node_types()), "layers": layers(),
                     "by_layer": {l: len(by_layer(l)) for l in layers()}},
            "reputation": {"n_pathways": len(self.reputation.leaderboard(top=10_000)),
                           "top": [s.to_dict() for s in lb]},
            "knowledge": self.knowledge.stats(),
            "promoted_this_session": self._promoted,
        }

    # ----------------------------------------------------------- knowledge
    def retrieve_knowledge(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.knowledge.retrieve(query, k=k)]

    def recall_memory(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.knowledge.recall(query, k=k)]

    def remember(self, note: str, kind: str = "finding",
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.knowledge.remember(note, kind=kind, metadata=metadata)

    def ingest_books(self, max_chars: int = 200_000) -> Dict[str, int]:
        return self.knowledge.ingest_manifest(max_chars=max_chars)

    def consolidate_knowledge(self, **kw) -> Dict[str, int]:
        """Promote good chat Q&A into first-class knowledge (idea 3, see
        ``KnowledgeBase.consolidate_qa``)."""
        return self.knowledge.consolidate_qa(**kw)

    # ----------------------------------------------------------- internet
    def web_fetch(self, url: str, max_chars: int = 100_000) -> str:
        raw = self.fetch(url)
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        return text[:max_chars]

    def download_file(self, url: str, filename: str) -> str:
        """Download a URL to ``data/<filename>`` (INSIDE the folder, Rule 1)."""
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, os.path.basename(filename))
        if not _inside_project(path):
            raise ValueError(f"Rule 1: download path must be inside {PROJECT_ROOT}")
        with open(path, "wb") as fh:
            fh.write(self.fetch(url))
        return path

    # ----------------------------------------------------------- data/tests
    def fetch_dataset(self, name: str = "synthetic", source: str = "synthetic",
                      **kw) -> Dict[str, Any]:
        """Acquire a generic (T, D) training array. Sources: 'synthetic' (always
        works), 'openml' (sklearn fetch_openml, standardized — optional dep), or an
        injected loader by name. The array is cached under ``data/`` (Rule 1)."""
        if source in (self.dataset_loaders or {}):
            X = np.asarray(self.dataset_loaders[source](name=name, **kw), dtype=float)
        elif source == "synthetic":
            X = synthetic_dataset(**{k: v for k, v in kw.items() if k in ("T", "D", "seed")})
        elif source == "openml":
            X = _load_openml(name, **kw)
        else:
            raise ValueError(f"unknown dataset source {source!r}")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, f"dataset_{source}_{name}.npz".replace("/", "_"))
        np.savez(path, X=X)
        return {"source": source, "name": name, "shape": list(X.shape), "path": path, "X": X}

    def fetch_crypto_symbol(self, symbol: str = "BTC/USD", timeframe: str = "1h",
                            limit: int = 600, exchange: str = "kraken",
                            prefer_live: bool = True) -> Dict[str, Any]:
        """Acquire ALL the market data for one crypto symbol → a generic (T, D)
        (column 0 = log-return). Live via ccxt, cached under ``data/crypto/`` (Rule 1),
        synthetic fallback offline. The LLM-orchestrated capstone data step (§12.1)."""
        from ..adapters.crypto import load_symbol
        d = load_symbol(symbol, timeframe, limit, exchange, prefer_live=prefer_live)
        return {"symbol": symbol, "source": d["source"], "n_candles": d["n_candles"],
                "shape": list(d["matrix"].shape), "feature_names": d["feature_names"],
                "last_close": d["last_close"], "X": d["matrix"]}

    # ------------------------------------------------ project file access (Rule 1)
    _SKIP_DIRS = ("/.venv/", "/.git/", "/.pw-browsers/", "/ollama-bin/", "/ollama-models/",
                  "/__pycache__/", "/node_modules/", "/data/")

    def _safe(self, path: str) -> str:
        full = os.path.abspath(os.path.join(PROJECT_ROOT, path))
        if not (full == PROJECT_ROOT or full.startswith(PROJECT_ROOT + os.sep)):
            raise ValueError(f"Rule 1: path must be inside {PROJECT_ROOT}")
        return full

    def list_files(self, pattern: str = "**/*.py", max_files: int = 300) -> List[str]:
        """List project files matching a glob (relative to the project root). Skips
        venv/git/caches/binaries. The LLM uses this to navigate the codebase."""
        import glob
        out = []
        for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern), recursive=True)):
            if os.path.isfile(p) and not any(s in p for s in self._SKIP_DIRS):
                out.append(os.path.relpath(p, PROJECT_ROOT))
                if len(out) >= max_files:
                    break
        return out

    def read_file(self, path: str, max_chars: int = 24_000) -> Dict[str, Any]:
        """Read a project file's text (Rule-1 scoped). Full read access for the LLM."""
        full = self._safe(path)
        if not os.path.isfile(full):
            return {"path": path, "error": "not a file"}
        with open(full, errors="replace") as fh:
            txt = fh.read(max_chars + 1)
        return {"path": path, "content": txt[:max_chars], "truncated": len(txt) > max_chars,
                "chars": min(len(txt), max_chars)}

    def search_files(self, query: str, pattern: str = "**/*.py", max_hits: int = 40) -> List[Dict[str, Any]]:
        """Case-insensitive substring search across project files (grep-like)."""
        import glob
        out = []
        ql = query.lower()
        for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern), recursive=True)):
            if not os.path.isfile(p) or any(s in p for s in self._SKIP_DIRS):
                continue
            try:
                with open(p, errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if ql in line.lower():
                            out.append({"file": os.path.relpath(p, PROJECT_ROOT), "line": i,
                                        "text": line.strip()[:200]})
                            if len(out) >= max_hits:
                                return out
            except Exception:
                continue
        return out

    # ------------------------------------------------ runtime results (read-only)
    def read_leaderboard(self, k: int = 12, metric: str = "crps_skill") -> Dict[str, Any]:
        """The Stacked-DAG search leaderboard — which network combinations scored best
        (so the LLM can answer 'why did the search pick X?'). Reads the persistent db."""
        from ..leaderboard import Leaderboard
        lb = Leaderboard()
        try:
            summary = lb.summary()
            if summary["n_runs"] == 0:
                return {"n_runs": 0, "note": "no search persisted yet — run a DAG search to populate it"}
            metric = metric if metric in ("crps_skill", "dsr", "mean_crps") else "crps_skill"
            top = [{c: r.get(c) for c in ("combiner", "n_nodes", "crps_skill", "dsr",
                                          "mean_crps", "calibrated")} for r in lb.top(k, metric=metric)]
            per = {}
            for r in lb.top(300, metric="crps_skill", dedupe=False):
                c = r["combiner"]
                if c not in per or (r["crps_skill"] or -9) > (per[c]["crps_skill"] or -9):
                    per[c] = {"combiner": c, "crps_skill": r["crps_skill"], "dsr": r["dsr"],
                              "mean_crps": r["mean_crps"], "calibrated": bool(r["calibrated"])}
            return {"summary": summary, "top": top, "best_per_combiner": list(per.values())}
        finally:
            lb.close()

    def read_capstone(self) -> Dict[str, Any]:
        """The latest freeze→forward-test result (verdict, forward-skill, DSR,
        calibration, directional hit, next move), if one has been computed."""
        p = os.path.join(PROJECT_ROOT, "agent_state", "last_capstone.json")
        if not os.path.exists(p):
            return {"note": "no capstone result yet — open the Capstone dashboard tab or run run_crypto_capstone"}
        try:
            with open(p) as fh:
                d = json.load(fh)
            return {k: v for k, v in d.items() if k != "forward_paths"}
        except Exception as e:
            return {"error": str(e)}

    # ----------------------------------------------- create / mutate (1/2/3)
    def evolver_for(self, feature: str, **kw):
        if feature not in _EVOLVERS:
            raise ValueError(f"unknown feature {feature!r}; known: {sorted(_EVOLVERS)}")
        cls = _EVOLVERS[feature]
        if cls is PathwayEvolver:
            kw.setdefault("reputation", self.reputation)
        return cls(**kw)

    def run_evolution(self, feature: str, X: np.ndarray,
                      population: int = 6, generations: int = 3, seed: int = 0,
                      **kw) -> "RunResult":
        """Create + mutate models/algorithms/pathways via Features 1/2/3, gated by
        the Evaluator. Returns the result PLUS the handle needed to promote a winner."""
        ev = self.evolver_for(feature, population=population, generations=generations,
                              seed=seed, **kw)
        result = ev.run(np.asarray(X, dtype=float))
        return RunResult(feature=feature, evolver=ev, result=result, X=np.asarray(X, dtype=float))

    # ----------------------------------------- rank -> include as model node
    def promote_to_node(self, run: "RunResult", individual: Optional[Individual] = None,
                        only_if_admitted: bool = True) -> Optional[str]:
        """Turn a winning evolved genome into a REGISTERED bank node (appears in
        all_node_types / genome_manifest / the dashboard Model Bank). This is the
        'ranking them in order for including in the model nodes' step. Returns the
        new node_type, or None if nothing qualified."""
        ind = individual or run.result.best
        if ind is None:
            return None
        if only_if_admitted and ind not in run.result.admitted and run.result.admitted:
            ind = run.result.admitted[0]            # prefer a gate-admitted winner
        if only_if_admitted and not run.result.admitted:
            return None                              # nothing passed the gate
        predict_fn = run.evolver.make_predictor_fn(ind.genome)(run.X)
        conf = 0.5
        if ind.report is not None:
            conf = float(max(0.0, min(1.0, 0.5 + 0.5 * np.tanh(ind.report.sharpe))))
        self._promoted += 1
        node_type = f"evolved_{run.feature.split('_')[0]}_{self._promoted:03d}"
        while node_type in _REGISTRY:
            self._promoted += 1
            node_type = f"evolved_{run.feature.split('_')[0]}_{self._promoted:03d}"
        cls = _make_evolved_node_class(node_type, predict_fn, conf)
        register(cls)
        if self.knowledge is not None:
            self.knowledge.remember(
                f"Promoted {node_type} from {run.feature}: {ind.to_dict()['genome']} "
                f"(sharpe={getattr(ind.report,'sharpe',float('nan')):.3f})",
                kind="promotion", metadata={"node_type": node_type, "feature": run.feature})
        return node_type

    # --------------------------------------------------- LLM tool schemas
    def specs(self) -> List[Dict[str, Any]]:
        """Tool schemas for the LLM tool-calling boundary (same shape as routing)."""
        def spec(name, desc, props=None):
            return {"name": name, "description": desc,
                    "input_schema": {"type": "object", "properties": props or {}}}
        return [
            spec("observe_state", "Observe the current bank, reputation, knowledge state."),
            spec("retrieve_knowledge", "Search the ML-engineering book knowledge base.",
                 {"query": {"type": "string"}, "k": {"type": "integer"}}),
            spec("recall_memory", "Recall the agent's own past findings (archival memory).",
                 {"query": {"type": "string"}, "k": {"type": "integer"}}),
            spec("remember", "Store a finding into archival memory.",
                 {"note": {"type": "string"}, "kind": {"type": "string"}}),
            spec("fetch_dataset", "Acquire a generic (T,D) training dataset.",
                 {"name": {"type": "string"}, "source": {"type": "string"}}),
            spec("fetch_crypto_symbol", "Acquire all market data for one crypto symbol "
                 "as a generic (T,D) (ccxt live, cached, synthetic fallback).",
                 {"symbol": {"type": "string"}, "timeframe": {"type": "string"}}),
            spec("web_fetch", "Fetch text from a URL (internet).", {"url": {"type": "string"}}),
            spec("run_evolution", "Create/mutate models (feature1), algorithms (feature2) or "
                 "pathways (feature3); gated by the Evaluator.",
                 {"feature": {"type": "string"}}),
            spec("promote_to_node", "Register a gate-admitted winner as a new bank model node."),
            spec("list_files", "List project files matching a glob (read access to the codebase).",
                 {"pattern": {"type": "string"}}),
            spec("read_file", "Read a project file's full text content.",
                 {"path": {"type": "string"}}),
            spec("search_files", "Search the project code for a substring (grep-like).",
                 {"query": {"type": "string"}, "pattern": {"type": "string"}}),
            spec("read_leaderboard", "The DAG-search leaderboard: which network combinations scored best."),
            spec("read_capstone", "The latest freeze->forward-test result (verdict, skill, DSR, calibration)."),
        ]

    def call(self, name: str, **kw) -> Any:
        """Dispatch a tool by name (used by the LLM tool-calling loop)."""
        fn = getattr(self, name, None)
        if fn is None or name not in {s["name"] for s in self.specs()}:
            raise ValueError(f"unknown tool {name!r}")
        return fn(**kw)


@dataclass
class RunResult:
    feature: str
    evolver: Any
    result: EvolutionResult
    X: np.ndarray

    def summary(self) -> Dict[str, Any]:
        return {"feature": self.feature, "generations": self.result.generations,
                "tested": sum(1 for h in self.result.history if h["event"] == "tested"),
                "admitted": len(self.result.admitted),
                "best": self.result.best.to_dict() if self.result.best else None}


# ------------------------------------------------------------------ fetchers
def _http_get_bytes(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pattern-brain-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _load_openml(name: str, **kw) -> np.ndarray:
    """Optional: a standardized OpenML dataset as a numeric (T, D) array. Needs
    scikit-learn (already a light dep). Falls back to raising if unavailable so the
    caller can choose the synthetic source instead."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name=name, version=kw.get("version", 1), as_frame=False,
                      parser="liac-arff")
    X = np.asarray(ds.data, dtype=float)
    return X
