"""The ML Engineer Agent — the closed continuous-learning loop (PLAN.md §9, ENG-3).

Grounded in the Block-56 deep sweep (Rule 14, not invented):

* **AIDE Solution-Space Tree Search** (arXiv 2502.13138): each loop step either
  *drafts* a fresh candidate set or *improves* the current best — explore/exploit
  over the solution tree. Our "tree" is the bank + reputation; the metric is the
  Evaluator gate.
* **R&D-Agent dual roles** (arXiv 2505.14738): the **Researcher** (Orient/Decide —
  reads knowledge + reputation + performance, proposes the next experiment) and
  the **Developer** (Act — runs the evolver, handles failures). Rule 17 roles.
* **Sakana AI-Scientist open-ended loop** (Nature): observe → (literature) retrieve
  → experiment → iterate, forever.
* **Letta/MemGPT memory**: findings persist to archival memory each step; state
  reloads on start so the infinite loop survives restarts.

One OODA cycle = :meth:`MLEngineerAgent.step`: **Observe → Orient → Decide → Act →
Rank/Record**. :meth:`run` runs N steps; :meth:`run_forever` is the daemon, with a
hard STOP flag + budget caps (Rule-16 safety rails). :meth:`chat` lets a human
converse with the agent (ENG-4 dashboard) — works offline with a templated reply,
or through any injected LLM text-completer.

Domain-agnostic (Rule 23); all state under ``agent_state/`` inside the folder (Rule 1).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..knowledge import PROJECT_ROOT
from .tools import Toolbox, RunResult, synthetic_dataset

DEFAULT_STATE_DIR = os.path.join(PROJECT_ROOT, "agent_state")

PERSONA = (
    "You are The ML Engineer — an autonomous machine-learning engineer running an "
    "endless improvement loop over a bank of models. You observe the current model "
    "bank and pathway reputation, consult ML-engineering literature, design "
    "experiments (create/mutate models, algorithms, and pathways), evaluate them on "
    "a hard statistical gate, and promote only the winners into the permanent model "
    "nodes. You are rigorous, evidence-first, and you never claim a result the "
    "Evaluator did not admit."
)

FEATURES = ["feature1_model", "feature2_algorithm", "feature3_pathway"]


@dataclass
class StepResult:
    step: int
    observe: Dict[str, Any]
    orient: str                      # Researcher rationale
    feature: str
    mode: str                        # "draft" | "improve" (AIDE)
    act: Dict[str, Any]              # run summary
    promoted: Optional[str]          # new node_type, or None
    finding: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "feature": self.feature, "mode": self.mode,
                "orient": self.orient, "act": self.act, "promoted": self.promoted,
                "finding": self.finding, "ts": self.ts,
                "bank_nodes": self.observe.get("bank", {}).get("n_nodes")}


class MLEngineerAgent:
    def __init__(self, toolbox: Optional[Toolbox] = None,
                 llm_chat: Optional[Callable[..., str]] = None,
                 state_dir: str = DEFAULT_STATE_DIR,
                 features: Optional[List[str]] = None,
                 data_source: str = "synthetic",
                 population: int = 6, generations: int = 2,
                 max_promotions: int = 50, seed: int = 0) -> None:
        self.toolbox = toolbox or Toolbox()
        self.llm_chat = llm_chat
        self.state_dir = state_dir
        self.features = features or list(FEATURES)
        self.data_source = data_source
        self.population = population
        self.generations = generations
        self.max_promotions = max_promotions
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        # learned-so-far state (persisted)
        self.steps_done = 0
        self.promotions = 0
        self.feed: List[Dict[str, Any]] = []          # bounded activity feed (dashboard)
        self.scoreboard: Dict[str, Dict[str, float]] = {f: {"runs": 0.0, "admitted": 0.0}
                                                         for f in self.features}

    # --------------------------------------------------------- the OODA step
    def step(self) -> StepResult:
        n = self.steps_done + 1
        # --- Observe ---
        observe = self.toolbox.observe_state()
        # --- Orient (Researcher): consult literature + memory, pick the next move ---
        feature, mode, orient = self._researcher(observe)
        # --- Act (Developer): acquire data, run the evolver, survive failures ---
        act: Dict[str, Any]
        promoted: Optional[str] = None
        finding: str
        try:
            X = self._dataset(seed=int(self.rng.integers(0, 1_000_000)))
            gens = self.generations + (1 if mode == "improve" else 0)
            run = self.toolbox.run_evolution(feature, X, population=self.population,
                                             generations=gens, seed=self._seed + n)
            act = run.summary()
            self.scoreboard[feature]["runs"] += 1
            self.scoreboard[feature]["admitted"] += len(run.result.admitted)
            # --- Rank / Record: promote a gate-admitted winner into the model nodes ---
            if run.result.admitted and self.promotions < self.max_promotions:
                promoted = self.toolbox.promote_to_node(run, only_if_admitted=True)
                if promoted:
                    self.promotions += 1
            finding = (f"{feature} [{mode}]: tested {act['tested']}, "
                       f"admitted {act['admitted']}"
                       + (f", promoted {promoted}" if promoted else ", nothing passed the gate"))
        except Exception as e:                          # R&D-Agent error feedback
            act = {"feature": feature, "error": str(e)}
            finding = f"{feature} [{mode}] FAILED: {e}"
        # record the finding to archival memory (Letta tier)
        self.toolbox.remember(finding, kind="loop_step",
                              metadata={"step": n, "feature": feature, "mode": mode})
        res = StepResult(n, observe, orient, feature, mode, act, promoted, finding)
        self.steps_done = n
        self.feed.append(res.to_dict())
        self.feed = self.feed[-200:]                     # bound the feed
        return res

    # ----------------------------------------------------- Researcher (Orient)
    def _researcher(self, observe: Dict[str, Any]) -> tuple:
        """Pick (feature, mode, rationale). LLM if available, else a transparent
        AIDE explore/exploit heuristic over the per-feature admission scoreboard."""
        # exploit the best-yielding feature most of the time, else explore (AIDE)
        def yield_of(f):
            s = self.scoreboard[f]
            return (s["admitted"] + 0.5) / (s["runs"] + 1.0)
        explore = self.rng.random() < 0.3 or all(self.scoreboard[f]["runs"] == 0
                                                 for f in self.features)
        if explore:
            feature = self.features[int(self.rng.integers(0, len(self.features)))]
            why = "explore (sweep coverage)"
        else:
            feature = max(self.features, key=yield_of)
            why = f"exploit best admission-yield ({yield_of(feature):.2f})"
        # draft vs improve: improve the current best once a feature has a track record
        mode = "improve" if self.scoreboard[feature]["runs"] >= 1 and self.rng.random() < 0.6 else "draft"
        if self.llm_chat is not None:
            try:
                feature, mode, why = self._llm_decide(observe, feature, mode)
            except Exception:
                pass
        return feature, mode, why

    def _llm_decide(self, observe, feature, mode):
        kn = self.toolbox.retrieve_knowledge("how to improve model selection and "
                                             "avoid overfitting in evolutionary ML", k=2)
        sys = (PERSONA + "\nChoose the next experiment. Reply as compact JSON: "
               '{"feature": "feature1_model|feature2_algorithm|feature3_pathway", '
               '"mode": "draft|improve", "why": "<one line>"}.')
        user = (f"State: {json.dumps(observe)[:1500]}\n"
                f"Scoreboard: {json.dumps(self.scoreboard)}\n"
                f"Relevant literature: {json.dumps(kn)[:800]}\n"
                f"Heuristic suggestion: feature={feature}, mode={mode}.")
        txt = self.llm_chat([{"role": "system", "content": sys},
                             {"role": "user", "content": user}])
        obj = _extract_json(txt)
        f = obj.get("feature", feature)
        if f not in self.features:
            f = feature
        m = obj.get("mode", mode)
        m = m if m in ("draft", "improve") else mode
        return f, m, obj.get("why", "llm choice")

    def _dataset(self, seed: int = 0) -> np.ndarray:
        if self.data_source == "synthetic":
            return synthetic_dataset(T=420, D=1, seed=seed)
        return self.toolbox.fetch_dataset(self.data_source, source=self.data_source,
                                          seed=seed)["X"]

    # ---------------------------------------------------------------- driving
    def run(self, n_steps: int = 3) -> List[StepResult]:
        return [self.step() for _ in range(n_steps)]

    def run_forever(self, max_steps: Optional[int] = None,
                    stop_check: Optional[Callable[[], bool]] = None,
                    sleep: float = 0.0) -> int:
        """The infinite closed loop. Halts on: the STOP flag file, an injected
        ``stop_check`` returning True, the ``max_steps`` budget, or the
        ``max_promotions`` cap — the Rule-16 safety rails. Returns steps run."""
        ran = 0
        while True:
            if self.stop_requested():
                break
            if stop_check is not None and stop_check():
                break
            if max_steps is not None and ran >= max_steps:
                break
            if self.promotions >= self.max_promotions:
                break
            self.step()
            ran += 1
            if sleep > 0:
                time.sleep(sleep)
        self.save_state()
        return ran

    # ----------------------------------------------------------- stop control
    def _stop_path(self) -> str:
        return os.path.join(self.state_dir, "STOP")

    def stop_requested(self) -> bool:
        return os.path.exists(self._stop_path())

    def request_stop(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        with open(self._stop_path(), "w") as fh:
            fh.write(str(time.time()))

    def clear_stop(self) -> None:
        try:
            os.remove(self._stop_path())
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------ persistence
    def save_state(self) -> str:
        if not (os.path.abspath(self.state_dir).startswith(PROJECT_ROOT + os.sep)):
            raise ValueError(f"Rule 1: state dir must be inside {PROJECT_ROOT}")
        os.makedirs(self.state_dir, exist_ok=True)
        path = os.path.join(self.state_dir, "state.json")
        with open(path, "w") as fh:
            json.dump({"steps_done": self.steps_done, "promotions": self.promotions,
                       "scoreboard": self.scoreboard, "feed": self.feed[-50:]}, fh)
        return path

    def load_state(self) -> bool:
        path = os.path.join(self.state_dir, "state.json")
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            st = json.load(fh)
        self.steps_done = int(st.get("steps_done", 0))
        self.promotions = int(st.get("promotions", 0))
        self.scoreboard.update(st.get("scoreboard", {}))
        self.feed = st.get("feed", [])
        return True

    def status(self) -> Dict[str, Any]:
        return {"persona": "The ML Engineer", "steps_done": self.steps_done,
                "promotions": self.promotions, "max_promotions": self.max_promotions,
                "stopped": self.stop_requested(), "scoreboard": self.scoreboard,
                "llm": self.llm_chat is not None, "feed_len": len(self.feed)}

    # ----------------------------------------------------------- conversation
    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Converse with the agent (ENG-4). Works offline (templated reply) or via
        an injected LLM text-completer, grounded in live state + retrieved knowledge."""
        observe = self.toolbox.observe_state()
        kn = self.toolbox.retrieve_knowledge(message, k=3)
        recalls = self.toolbox.recall_memory(message, k=3)
        if self.llm_chat is None:
            return self._templated_reply(message, observe, kn, recalls)
        sys = (PERSONA + "\nAnswer the user grounded in this live state and "
               "retrieved knowledge. Be concrete; cite a node/pathway/number when relevant.\n"
               f"State: {json.dumps(observe)[:1500]}\n"
               f"Knowledge: {json.dumps(kn)[:1000]}\n"
               f"Your past findings: {json.dumps(recalls)[:800]}")
        msgs = [{"role": "system", "content": sys}] + (history or []) + \
               [{"role": "user", "content": message}]
        return self.llm_chat(msgs)

    def _templated_reply(self, message, observe, kn, recalls) -> str:
        bank = observe["bank"]
        top = observe["reputation"]["top"]
        lines = [
            f"[The ML Engineer — offline mode, no LLM backend configured]",
            f"You asked: {message!r}",
            f"Current bank: {bank['n_nodes']} model nodes across {len(bank['layers'])} layers; "
            f"I've run {self.steps_done} loop step(s) and promoted {self.promotions} new node(s).",
        ]
        if top:
            lines.append(f"Top pathway so far: {' -> '.join(top[0]['pathway'])} "
                         f"(score {top[0]['score']}).")
        if kn:
            lines.append(f"Relevant from the ML-engineering library: "
                         f"\"{kn[0]['text'][:160]}…\" (source: {kn[0]['source']}).")
        if recalls:
            lines.append(f"From my own past work: {recalls[0]['text'][:160]}")
        lines.append("Set a cloud LLM key (e.g. ZAI_API_KEY) or run Ollama for full conversation.")
        return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        i, j = text.index("{"), text.rindex("}")
        return json.loads(text[i:j + 1])
    except Exception:
        return {}
