"""Ideation & Research Advisor (PLAN.md §10 / Rule 26, owner-mandated 2026-06-21).

The owner's premise, verbatim in spirit: "I'm an individual with limited knowledge —
I can only think of a function or a product I want; I can't think of the many
important concepts/ideas/architectures. You have access to vast data + the
internet + this project's files, and can identify what's working and what's not.
Replace me with you: read my goals/conversations/rules, look online + at your
memory, come up with advanced ideas, ask to search, then implement with my
approval."

This is the *in-code* half of that mechanism (the conversational half is Rule 26,
which makes Claude do the same proactively every turn). The Advisor:

1. **Gathers context** — reads the project's own RULES.md / PLAN.md / DISCUSSION_NOTES.md
   plus the live system state (bank by layer, pathway reputation, knowledge base),
   and works out **what's working and what's weak** (Rule 14 evidence).
2. **Generates ideas** — turns the owner's goals into advanced concepts/architectures:
   via the cloud LLM (the "vast data" proxy) grounded in the project context + the
   book KB, OR an offline heuristic that mines the project's own recorded gaps.
3. **Each idea is actionable + approval-gated** — it carries the *search queries*
   to run, the *proposed change*, the *expected benefit* and *risk*, and a status
   (proposed → approved → implemented/dismissed). The Advisor NEVER implements;
   it proposes, the owner approves, Claude implements (Rule 26).

Domain-agnostic (Rule 23); ideas + status persist under ``agent_state/`` (Rule 1).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..knowledge import PROJECT_ROOT
from ..registry import by_layer, layers
from .tools import Toolbox

DEFAULT_STATE_DIR = os.path.join(PROJECT_ROOT, "agent_state")

# Known headroom recorded in the project's own PLAN (§5 Block 42 + open items) —
# the offline heuristic draws grounded ideas from these instead of inventing.
KNOWN_GAPS = [
    ("deep reinforcement learning nodes (DQN/PPO/SAC)",
     "Catalog category 12 is in the plan but not in the bank; the rl layer is thin.",
     ["deep RL for time-series decision DQN PPO 2026", "stable-baselines3 lightweight PPO"]),
    ("true message-passing GNN nodes (torch_geometric)",
     "Only a k-NN graph-conv denoiser exists; real GNN routing is plan headroom.",
     ["graph neural network time series forecasting 2026", "torch_geometric temporal"]),
    ("foundation time-series models (Chronos/TimesFM/Moirai)",
     "Pretrained TS foundation models could be wrapped behind the Node interface.",
     ["time series foundation model Chronos TimesFM 2026 zero-shot"]),
    ("hybrid (dense+sparse/BM25) retrieval for the knowledge base",
     "2026 RAG standard; current KB is dense-only cosine.",
     ["hybrid search BM25 dense retrieval RAG 2026 best practice"]),
    ("a live web-search tool for the agent",
     "The agent has web_fetch (one URL) but no search; grounding ideas in current "
     "research needs a search API.",
     ["free web search API for LLM agents 2026 SearXNG Tavily Brave"]),
    ("a second data-domain adapter (prove §0 portability)",
     "The whole bank is domain-agnostic; a non-stock adapter would prove it.",
     ["domain-agnostic ML pipeline adapter pattern"]),
]


@dataclass
class Idea:
    id: str
    title: str
    concept: str
    why: str
    search_queries: List[str]
    proposed_change: str
    expected_benefit: str
    risk: str = ""
    status: str = "proposed"          # proposed | approved | implemented | dismissed
    source: str = "heuristic"         # heuristic | llm
    # effectiveness upgrades (Block 56 / AI Co-Scientist + Idea-Arena research):
    feasibility: float = 0.5          # critic pass: 0..1, is it grounded + buildable here
    novelty: float = 0.5              # RAG-grounded: 1 - max similarity to known corpus/ideas
    elo: float = 1000.0               # pairwise-tournament rank score
    critique: str = ""                # the self-critique (pitfalls / feasibility reasoning)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IdeationAdvisor:
    def __init__(self, toolbox: Optional[Toolbox] = None,
                 llm_chat: Optional[Any] = None,
                 state_dir: str = DEFAULT_STATE_DIR) -> None:
        self.toolbox = toolbox or Toolbox()
        self.llm_chat = llm_chat
        self.state_dir = state_dir
        self.ideas: List[Idea] = []
        self.load()

    # ----------------------------------------------------- context (the digging)
    def _read(self, name: str, tail: int = 0) -> str:
        path = os.path.join(PROJECT_ROOT, name)
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        return txt[-tail:] if tail else txt

    def gather_context(self) -> Dict[str, Any]:
        """Read the project's goals/rules/state and work out what's working vs weak."""
        by = {l: len(by_layer(l)) for l in layers()}
        weak_layers = sorted(by, key=lambda l: by[l])[:3]
        lb = self.toolbox.reputation.leaderboard(top=5)
        working = [{"pathway": s.to_dict()["pathway"], "score": s.to_dict()["score"]} for s in lb]
        kb = self.toolbox.knowledge.stats()
        # the project's own open items (Rule 14: read the record, don't guess)
        plan = self._read("PLAN.md")
        open_items = [ln.strip(" -") for ln in plan.splitlines()
                      if ("OPEN QUESTION" in ln or "❓" in ln or "still open" in ln.lower())][:6]
        return {
            "bank_by_layer": by, "weakest_layers": weak_layers,
            "working_pathways": working, "knowledge_passages": kb.get("passages", 0),
            "open_plan_items": open_items,
            "goal_excerpt": self._read("README.md")[:600],
        }

    # --------------------------------------------------------------- generation
    def generate(self, n: int = 5) -> List[Idea]:
        """Co-Scientist-style cycle: GENERATE candidates → CRITIQUE (feasibility +
        pitfalls) → NOVELTY (RAG) → pairwise-ELO RANK → keep the top n. Returns the
        survivors, best-first."""
        ctx = self.gather_context()
        pool = self._llm_ideas(ctx, max(n, 6)) if self.llm_chat is not None else []
        if not pool:
            pool = self._heuristic_ideas(ctx, max(n, 6))
        for idea in pool:           # critic pass + RAG novelty on every candidate
            self._critique(idea)
            self._novelty(idea)
        # bound the O(n^2) pairwise tournament: pre-filter by composite score to a
        # small finalist set, then run ELO only on the finalists (keeps LLM-judge
        # cost ~constant per cycle).
        pool.sort(key=lambda x: 0.5 * x.novelty + 0.5 * x.feasibility, reverse=True)
        finalists = pool[:max(n, 4)]
        self._elo_rank(finalists)   # pairwise, both orders (position-bias control)
        finalists.sort(key=lambda x: x.elo, reverse=True)
        ranked = finalists + pool[len(finalists):]
        added = self._merge(ranked[:n])
        self.save()
        return added

    # ----------------------------------------------------- effectiveness upgrades
    def _critique(self, idea: "Idea") -> None:
        """Attack the idea: feasibility 0..1 + pitfalls. Counters the documented
        'LLMs overstate success / skimp on feasibility' failure mode."""
        if self.llm_chat is not None:
            try:
                txt = self.llm_chat([
                    {"role": "system", "content": "You are a skeptical ML reviewer. Critique the "
                     "idea for feasibility, real pitfalls, and statistical rigor. Reply ONLY JSON "
                     '{"feasibility": 0..1, "critique": "<=2 sentences"}.'},
                    {"role": "user", "content": f"{idea.title}: {idea.concept} "
                     f"Proposed: {idea.proposed_change}"}])
                o = _extract_json_obj(txt)
                idea.feasibility = max(0.0, min(1.0, float(o.get("feasibility", 0.5))))
                idea.critique = str(o.get("critique", ""))[:400] or idea.critique
                if idea.critique:
                    return
            except Exception:
                pass
        # offline heuristic: feasibility from dependency/compute vs reuse signals
        text = (idea.concept + " " + idea.proposed_change + " " + idea.risk).lower()
        feas = 0.6
        for kw, d in (("new dependency", -0.18), ("torch", -0.08), ("compute", -0.08),
                      ("api key", -0.06), ("search api", -0.06),
                      ("wrap behind", 0.12), ("add node", 0.08), ("wire", 0.1),
                      ("optional", 0.1), ("existing", 0.08)):
            if kw in text:
                feas += d
        idea.feasibility = max(0.05, min(0.95, feas))
        idea.critique = ("Heuristic critique: feasibility inferred from dependency/compute vs "
                         "reuse signals; verify statistical rigor + real pitfalls before building.")

    def _novelty(self, idea: "Idea") -> None:
        """RAG-grounded novelty = 1 - max similarity to the known corpus + prior ideas
        (so we don't re-propose what the books already cover or we already proposed)."""
        try:
            kb = self.toolbox.knowledge
            v = kb.embedder.embed(idea.title + ". " + idea.concept)
            hits = kb.store.search(v, k=1)
            sim = float(hits[0].score) if hits else 0.0
            idea.novelty = max(0.0, min(1.0, 1.0 - sim))
        except Exception:
            idea.novelty = 0.6

    def _compare(self, a: "Idea", b: "Idea") -> float:
        """Return 1.0 if a is the better idea, else 0.0 (the Idea-Arena judge)."""
        if self.llm_chat is not None:
            try:
                t = self.llm_chat([
                    {"role": "system", "content": "Which idea is better overall "
                     "(novelty + feasibility + impact)? Reply ONLY 'A' or 'B'."},
                    {"role": "user", "content": f"A: {a.title} — {a.concept}\n"
                     f"B: {b.title} — {b.concept}"}]).strip().upper()
                if t.startswith("A"):
                    return 1.0
                if t.startswith("B"):
                    return 0.0
            except Exception:
                pass
        sa, sb = 0.5 * a.novelty + 0.5 * a.feasibility, 0.5 * b.novelty + 0.5 * b.feasibility
        return 1.0 if sa >= sb else 0.0

    def _elo_rank(self, ideas: List["Idea"], k: float = 24.0) -> None:
        """Pairwise round-robin ELO. permutations() runs each pair in BOTH orders —
        the documented fix for LLM-judge position bias."""
        import itertools
        for i in ideas:
            i.elo = 1000.0
        for a, b in itertools.permutations(ideas, 2):
            wa = self._compare(a, b)
            ea = 1.0 / (1.0 + 10 ** ((b.elo - a.elo) / 400.0))
            a.elo += k * (wa - ea)
            b.elo += k * ((1.0 - wa) - (1.0 - ea))

    def _llm_ideas(self, ctx: Dict[str, Any], n: int) -> List[Idea]:
        kn = self.toolbox.retrieve_knowledge(
            "advanced architectures and concepts to improve a self-evolving ML model bank", k=3)
        sys = (
            "You are the Ideation & Research Advisor for the Pattern Brain project. The "
            "owner has limited ML knowledge and relies on you to turn their goals into "
            "advanced concepts/architectures they couldn't think of. Use the project "
            "context + literature. Reply ONLY as a JSON array of "
            f"{n} objects with keys: title, concept, why, search_queries (array of 2-3 "
            "web queries to verify/deepen it), proposed_change (concrete, files/components), "
            "expected_benefit, risk. Prefer ideas grounded in real 2026 research; be specific.")
        user = (f"Project context: {json.dumps(ctx)[:2000]}\n"
                f"Literature snippets: {json.dumps(kn)[:1000]}")
        try:
            txt = self.llm_chat([{"role": "system", "content": sys},
                                 {"role": "user", "content": user}])
            arr = _extract_json_array(txt)
            out = []
            for o in arr[:n]:
                out.append(Idea(
                    id="", title=str(o.get("title", "untitled"))[:120],
                    concept=str(o.get("concept", ""))[:600],
                    why=str(o.get("why", ""))[:400],
                    search_queries=[str(q) for q in (o.get("search_queries") or [])][:4],
                    proposed_change=str(o.get("proposed_change", ""))[:600],
                    expected_benefit=str(o.get("expected_benefit", ""))[:300],
                    risk=str(o.get("risk", ""))[:300], source="llm"))
            return out
        except Exception:
            return []

    def _heuristic_ideas(self, ctx: Dict[str, Any], n: int) -> List[Idea]:
        out: List[Idea] = []
        weak = ctx["weakest_layers"][0] if ctx["weakest_layers"] else "probability"
        out.append(Idea(
            id="", title=f"Strengthen the '{weak}' layer (it's the thinnest)",
            concept=f"The {weak} layer has the fewest nodes "
                    f"({ctx['bank_by_layer'].get(weak)}). Add diverse algorithms there so the "
                    f"Connector + evolvers have real choice in that functional slot.",
            why="A thin layer is a structural bottleneck for pathway diversity (Block 18).",
            search_queries=[f"best {weak} models for time series 2026",
                            f"{weak} layer algorithms machine learning catalog"],
            proposed_change=f"Add 3-5 new {weak}-layer nodes in pattern_brain/nodes/, "
                            f"behind the generic Node interface (Rule 23), with tests.",
            expected_benefit="More pathway diversity; better Feature-3 evolution.",
            risk="Marginal compute; new nodes must conform to the interlingua.",
            source="heuristic"))
        for title, why, queries in KNOWN_GAPS:
            out.append(Idea(
                id="", title=title, concept=why, why=why, search_queries=list(queries),
                proposed_change="Scope after the search; wrap behind the Node/tool interface, "
                                "shadow-first, gated by the Evaluator before any live authority.",
                expected_benefit="Closes recorded plan headroom toward the full catalog.",
                risk="New dependency / compute; keep optional per §0b.",
                source="heuristic"))
        return out[:max(1, n)]

    # -------------------------------------------------------------- bookkeeping
    def _merge(self, ideas: List[Idea]) -> List[Idea]:
        existing = {i.title.lower() for i in self.ideas}
        added = []
        for idea in ideas:
            if idea.title.lower() in existing:
                continue
            idea.id = f"idea-{int(time.time()*1000)}-{len(self.ideas)}"
            self.ideas.append(idea)
            existing.add(idea.title.lower())
            added.append(idea)
            # the advisor records its own thinking into archival memory (Letta)
            self.toolbox.knowledge.remember(
                f"Proposed idea: {idea.title} — {idea.concept}", kind="idea",
                metadata={"id": idea.id})
        return added

    def set_status(self, idea_id: str, status: str) -> bool:
        if status not in ("proposed", "approved", "implemented", "dismissed"):
            raise ValueError(f"bad status {status!r}")
        for i in self.ideas:
            if i.id == idea_id:
                i.status = status
                self.save()
                return True
        return False

    def list_ideas(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self.ideas if status is None else [i for i in self.ideas if i.status == status]
        return [i.to_dict() for i in sorted(items, key=lambda x: x.ts, reverse=True)]

    # ------------------------------------------------------------- persistence
    def save(self) -> str:
        if not os.path.abspath(self.state_dir).startswith(PROJECT_ROOT + os.sep):
            raise ValueError(f"Rule 1: state dir must be inside {PROJECT_ROOT}")
        os.makedirs(self.state_dir, exist_ok=True)
        path = os.path.join(self.state_dir, "ideas.json")
        with open(path, "w") as fh:
            json.dump([i.to_dict() for i in self.ideas], fh, indent=2)
        return path

    def load(self) -> bool:
        path = os.path.join(self.state_dir, "ideas.json")
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            self.ideas = [Idea(**o) for o in json.load(fh)]
        return True


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    try:
        i, j = text.index("["), text.rindex("]")
        return json.loads(text[i:j + 1])
    except Exception:
        return []


def _extract_json_obj(text: str) -> Dict[str, Any]:
    try:
        i, j = text.index("{"), text.rindex("}")
        return json.loads(text[i:j + 1])
    except Exception:
        return {}
