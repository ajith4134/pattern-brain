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
import threading
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
                 max_promotions: int = 50, seed: int = 0,
                 auto_ingest_books: bool = True,
                 consolidate_every: int = 5) -> None:
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
        # the agent "comes up with all the books and converts them into a vector DB"
        # on its own (owner's requirement): on first use it ingests the curated
        # ML-engineering corpus into the KB, in the background so it never blocks
        # the loop. Best-effort — a failed fetch is skipped, not fatal.
        self.auto_ingest_books = auto_ingest_books
        self._books_started = False
        # learned-so-far state (persisted)
        self.steps_done = 0
        self.promotions = 0
        self.feed: List[Dict[str, Any]] = []          # bounded activity feed (dashboard)
        self.scoreboard: Dict[str, Dict[str, float]] = {f: {"runs": 0.0, "admitted": 0.0}
                                                         for f in self.features}
        # operator mode (idea 2): a write action staged by chat, awaiting confirmation
        self._pending_action: Optional[Dict[str, Any]] = None
        # consolidation cadence (idea 3): chat Q&A promoted to knowledge every N turns
        self.consolidate_every = max(0, int(consolidate_every))
        self._qa_since_consolidate = 0

    # ------------------------------------------------- knowledge bootstrapping
    def ingest_books(self, max_chars: int = 40000) -> dict:
        """Fetch the curated ML-engineering book corpus and convert it into the
        vector DB (synchronous). Returns {title: n_chunks}. Best-effort per book."""
        res = self.toolbox.ingest_books(max_chars=max_chars)
        self.toolbox.remember(
            f"Ingested book corpus into the vector DB: "
            f"{sum(res.values())} passages from {sum(1 for v in res.values() if v)} sources.",
            kind="knowledge")
        return res

    def ensure_books(self) -> None:
        """Trigger book ingestion ONCE, in a background thread (non-blocking) so the
        agent self-populates its literature KB without stalling the loop."""
        if not self.auto_ingest_books or self._books_started:
            return
        self._books_started = True

        def _work():
            try:
                self.ingest_books()
            except Exception:
                pass
        threading.Thread(target=_work, daemon=True).start()

    # --------------------------------------------------------- the OODA step
    def step(self) -> StepResult:
        self.ensure_books()
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
    def plan_dag_search(self, X, rounds: int = 2, n_per_round: int = 4,
                        base_pool=None, dataset_id: str = "agent_dag", **search_kw):
        """Phase-8 slice 8: drive the Stacked-DAG search with THIS agent's LLM as the
        experiment planner (controller, never scorer). The LLM proposes DAG
        combinations from the leaderboard; ``DAGSearch`` + the ``Evaluator`` score
        them. Works offline (heuristic proposer). Returns the best DAG found."""
        from ..search import DAGSearch
        from .planner import DAGPlanner
        search = DAGSearch(X, base_pool=base_pool, dataset_id=dataset_id, **search_kw)
        best = DAGPlanner(search, llm_chat=self.llm_chat).plan_and_search(
            rounds=rounds, n_per_round=n_per_round)
        skill = (best or {}).get("crps_skill")
        self.toolbox.remember(
            f"DAG search: best CRPS-skill={None if skill is None else round(float(skill), 4)} "
            f"({(best or {}).get('combiner')}), {search.lb.count()} nets tried",
            kind="dag_search")
        return best

    #: read-only tools the chat LLM may call — full file access + live RESULTS
    #: (leaderboard, capstone, state) so it reasons over code AND runtime; no mutate/web.
    _CHAT_TOOLS = ("list_files", "read_file", "search_files", "observe_state",
                   "retrieve_knowledge", "recall_memory", "read_leaderboard", "read_capstone")

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None,
             max_tool_calls: int = 3, max_seconds: float = 35.0) -> str:
        """Converse like a coding assistant — FULL READ ACCESS to the project files AND
        the live results (search leaderboard, capstone forward-test, bank state). A
        ReAct loop lets the LLM read/search before answering; each Q&A is cached to
        archival memory and recalled on similar questions. Offline → templated reply."""
        self.ensure_books()
        # operator mode (idea 2): if a write action is staged, a confirm/cancel reply
        # routes straight to execution — no LLM round-trip needed.
        if self._pending_action is not None:
            low = (message or "").strip().lower()
            if low in ("confirm", "yes", "y", "approve", "go", "go ahead", "do it", "run it"):
                return self.confirm_action(self._pending_action["id"], approve=True)["reply"]
            if low in ("cancel", "no", "n", "abort", "stop", "nevermind", "never mind"):
                return self.confirm_action(self._pending_action["id"], approve=False)["reply"]
        observe = self.toolbox.observe_state()
        kn = self.toolbox.retrieve_knowledge(message, k=3)
        recalls = self.toolbox.recall_memory(message, k=3)
        if self.llm_chat is None:
            return self._templated_reply(message, observe, kn, recalls)
        tools_help = (
            "You have FULL READ ACCESS to this project's files AND its live results. To use a tool, reply "
            "with ONLY a single JSON object on one line and NOTHING else:\n"
            '{"tool":"read_file","path":"pattern_brain/network.py"}\n'
            '{"tool":"search_files","query":"def run_capstone"}\n'
            '{"tool":"list_files","pattern":"**/*.py"}\n'
            '{"tool":"read_leaderboard"}   — which network combinations (mixture/stacked/gated) scored best\n'
            '{"tool":"read_capstone"}      — the latest freeze->forward-test verdict + metrics\n'
            '{"tool":"observe_state"}      — current bank/reputation state\n'
            "I will reply with the tool result; then use another tool or give your FINAL ANSWER as plain "
            "prose (no JSON). To answer 'why did the search pick X', READ the leaderboard first; to explain "
            "HOW something works, READ the real code first.\n"
            "You may also TRIGGER A REAL RUN (these are gated — they do NOT run until the human confirms):\n"
            '{"tool":"run_dag_search","symbol":"BTC/USD","timeframe":"1h"}  — search network combinations on a symbol\n'
            '{"tool":"run_capstone","symbol":"BTC/USD","timeframe":"1h"}    — run the freeze->forward-test capstone\n'
            "Emit a run tool ONLY when the user explicitly asks to run/execute something; otherwise stay read-only.")
        sys = (PERSONA + "\nYou are the ML Engineer for the Pattern Brain project (a network-of-models "
               "that forecasts return distributions). Answer concretely, citing files/nodes/numbers.\n"
               f"Live state: {json.dumps(observe)[:1000]}\n"
               f"Retrieved knowledge: {json.dumps(kn)[:600]}\n"
               f"Your past answers (reuse if they fit): {json.dumps(recalls)[:700]}\n" + tools_help)
        convo = [{"role": "system", "content": sys}] + (history or []) + \
                [{"role": "user", "content": message}]
        answer = None
        deadline = time.time() + max(5.0, max_seconds)    # hard wall-clock cap (dashboard responsiveness)
        for _ in range(max(1, max_tool_calls)):
            if time.time() > deadline:                    # stop looping; fall through to a final answer
                break
            reply = self.llm_chat(convo)
            action = self._parse_tool(reply)
            if action is None:
                answer = reply
                break
            name = action.get("tool")
            if name in self._CHAT_WRITE_TOOLS:           # operator mode: stage, don't run (idea 2)
                return self._stage_write_action(name, action)
            if name not in self._CHAT_TOOLS:
                result = {"error": f"{name!r} not available in chat (read-only): {list(self._CHAT_TOOLS)}"}
            else:
                try:
                    result = self.toolbox.call(name, **{k: v for k, v in action.items() if k != "tool"})
                except Exception as e:
                    result = {"error": str(e)}
            convo.append({"role": "assistant", "content": reply})
            convo.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{json.dumps(result)[:3000]}"})
        if answer is None:
            convo.append({"role": "user", "content": "Now give your final answer in plain prose (no tools)."})
            answer = self.llm_chat(convo)
        try:                                                 # cache the Q&A (builds project knowledge)
            self.toolbox.remember(f"Q: {message}\nA: {answer}", kind="chat_qa")
            self._maybe_consolidate()                        # periodic promotion to knowledge (idea 3)
        except Exception:
            pass
        return answer

    # ------------------------------------------------- operator mode (idea 2)
    #: chat may STAGE these write actions; they run only after human confirmation.
    _CHAT_WRITE_TOOLS = ("run_dag_search", "run_capstone")

    def _stage_write_action(self, name: str, action: Dict[str, Any]) -> str:
        """Park a write action as a pending, confirmation-gated request and return a
        human-readable proposal. Nothing executes until :meth:`confirm_action`."""
        args = self._sanitize_write(name, {k: v for k, v in action.items() if k != "tool"})
        aid = f"act-{int(time.time() * 1000)}"
        self._pending_action = {"id": aid, "tool": name, "args": args,
                                "desc": self._describe_write(name, args)}
        return ("⏸ That is a WRITE action — it runs real compute, so it needs your confirmation.\n"
                f"Proposed: {self._pending_action['desc']}\n"
                "Click Confirm (or reply 'confirm' to proceed / 'cancel' to abort).")

    def pending_action(self) -> Optional[Dict[str, Any]]:
        """The currently-staged write action awaiting confirmation, if any."""
        return self._pending_action

    def confirm_action(self, action_id: str, approve: bool = True) -> Dict[str, Any]:
        """Execute (or cancel) the staged write action. The single guarded boundary
        that turns the assistant from observer into operator."""
        pa = self._pending_action
        if pa is None or pa.get("id") != action_id:
            return {"ran": False, "reply": "No matching pending action to confirm "
                    "(it may have already run or been cancelled)."}
        self._pending_action = None
        if not approve:
            return {"ran": False, "reply": f"Cancelled: {pa['desc']}"}
        try:
            result = getattr(self, "_run_" + pa["tool"][4:])(**pa["args"]) \
                if pa["tool"].startswith("run_") else None
        except Exception as e:                               # surface failures honestly
            return {"ran": False, "error": str(e), "reply": f"Action failed: {e}"}
        reply = self._summarize_write(pa["tool"], result)
        try:
            self.toolbox.remember(f"Operator ran {pa['tool']}({pa['args']}): {reply}",
                                  kind="operator_action")
        except Exception:
            pass
        return {"ran": True, "reply": reply, "result": result}

    @staticmethod
    def _sanitize_write(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp operator-action arguments to safe bounds (Rule-16 safety rails)."""
        def _i(v, lo, hi, dflt):
            try:
                return int(max(lo, min(int(v), hi)))
            except Exception:
                return dflt
        out = {"symbol": str(args.get("symbol", "BTC/USD"))[:20],
               "timeframe": str(args.get("timeframe", "1h"))[:5],
               "exchange": str(args.get("exchange", "kraken"))[:20],
               "limit": _i(args.get("limit", 400), 120, 2000, 400),
               "prefer_live": bool(args.get("prefer_live", True))}
        if name == "run_dag_search":
            out["rounds"] = _i(args.get("rounds", 2), 1, 4, 2)
            out["n_per_round"] = _i(args.get("n_per_round", 4), 1, 6, 4)
        return out

    @staticmethod
    def _describe_write(name: str, a: Dict[str, Any]) -> str:
        if name == "run_dag_search":
            return (f"DAG search on {a['symbol']} {a['timeframe']} "
                    f"({a['rounds']}×{a['n_per_round']} proposals, {a['limit']} candles)")
        return f"freeze→forward-test capstone on {a['symbol']} {a['timeframe']} ({a['limit']} candles)"

    @staticmethod
    def _summarize_write(name: str, result: Optional[Dict[str, Any]]) -> str:
        r = result or {}
        if name == "run_dag_search":
            best = r.get("best") or {}
            sk = best.get("crps_skill")
            return (f"Ran DAG search on {r.get('symbol')} (data: {r.get('source')}, "
                    f"{r.get('n_candles')} candles). Best: combiner={best.get('combiner')}, "
                    f"CRPS-skill={None if sk is None else round(float(sk), 4)}. "
                    "Ask 'read the leaderboard' to see all combinations.")
        v = r.get("verdict")
        return (f"Ran the capstone on {r.get('symbol')} (data: {r.get('source')}). "
                f"Verdict={v}, forward-skill={r.get('forward_skill')}, "
                f"DSR={r.get('forward_dsr')}, calibrated={r.get('forward_calibrated')}.")

    def _run_dag_search(self, symbol="BTC/USD", timeframe="1h", limit=400,
                        exchange="kraken", prefer_live=True,
                        rounds=2, n_per_round=4) -> Dict[str, Any]:
        """Operator action: load a symbol and drive a real Stacked-DAG search,
        recording to the PERSISTENT leaderboard so the chat's ``read_leaderboard``
        sees it afterward."""
        from ..adapters.crypto import load_symbol
        from ..leaderboard import Leaderboard
        d = load_symbol(symbol, timeframe, limit, exchange, prefer_live=prefer_live)
        lb = Leaderboard()                                   # on-disk → read_leaderboard sees it
        try:
            best = self.plan_dag_search(d["matrix"], rounds=rounds, n_per_round=n_per_round,
                                        dataset_id=symbol, leaderboard=lb)
        finally:
            lb.close()
        return {"symbol": symbol, "timeframe": timeframe, "source": d["source"],
                "n_candles": d["n_candles"], "best": best}

    def _run_capstone(self, symbol="BTC/USD", timeframe="1h", limit=400,
                      exchange="kraken", prefer_live=True) -> Dict[str, Any]:
        """Operator action: run the freeze→forward-test capstone on a symbol and
        persist it so the chat's ``read_capstone`` sees the latest verdict."""
        from ..adapters.crypto import run_crypto_capstone
        rep = run_crypto_capstone(symbol, timeframe, limit, exchange, prefer_live=prefer_live,
                                  holdout_frac=0.3, strategy="random", budget=6,
                                  min_train=45, n_samples=80, seed=0, return_paths=False)
        self._persist_capstone(rep)
        return {"symbol": symbol, "timeframe": timeframe, "source": rep.get("source"),
                "verdict": rep.get("verdict"), "forward_skill": rep.get("forward_skill"),
                "forward_dsr": rep.get("forward_dsr"),
                "forward_calibrated": rep.get("forward_calibrated")}

    def _persist_capstone(self, rep: Dict[str, Any]) -> None:
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with open(os.path.join(self.state_dir, "last_capstone.json"), "w") as fh:
                json.dump({k: v for k, v in rep.items() if k != "forward_paths"}, fh)
        except Exception:
            pass

    # ------------------------------------------------ consolidation (idea 3)
    def consolidate(self, **kw) -> Dict[str, int]:
        """Promote good chat Q&A into first-class knowledge (idea 3)."""
        res = self.toolbox.consolidate_knowledge(**kw)
        self._qa_since_consolidate = 0
        return res

    def _maybe_consolidate(self) -> None:
        """Run consolidation every ``consolidate_every`` chat turns (best-effort)."""
        if self.consolidate_every <= 0:
            return
        self._qa_since_consolidate += 1
        if self._qa_since_consolidate >= self.consolidate_every:
            try:
                self.consolidate()
            except Exception:
                self._qa_since_consolidate = 0

    @staticmethod
    def _parse_tool(reply: str):
        """Extract a single {\"tool\": ...} JSON action from an LLM reply, if present."""
        import re
        m = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}', reply or "", re.DOTALL)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
            return d if isinstance(d, dict) and d.get("tool") else None
        except Exception:
            return None

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
