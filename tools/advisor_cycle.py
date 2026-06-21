"""Daily Advisor cycle (PLAN.md §10 / Rule 26, Option 2 — owner-approved 2026-06-21).

Runs ONE full ideation cycle (generate → critique → novelty → pairwise-ELO rank)
and appends the top-N survivors to PLAN.md tagged 🟡 PROPOSED — the AUDITABLE,
NEVER-CROSSED boundary the autonomy research prescribes: it proposes only, it
never marks anything ✅ decided and never builds. Intended to be invoked by a
daily scheduled routine, which then opens a PR (it must NOT push to main).

Run manually:  set -a; . .env; set +a; python3 tools/advisor_cycle.py [N]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

import pattern_brain as pb
from pattern_brain.knowledge import PROJECT_ROOT, KnowledgeBase, _http_get
from pattern_brain.agent import Toolbox, IdeationAdvisor

SECTION = "## 🟡 Advisor proposals (auto-generated — NEVER decided/built; review + accept manually)"


def _load_dotenv() -> None:
    p = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _safe(fn, **kw):
    try:
        fn(**kw)
    except Exception:
        pass


def append_to_plan(ideas) -> None:
    plan = os.path.join(PROJECT_ROOT, "PLAN.md")
    with open(plan, encoding="utf-8") as fh:
        body = fh.read()
    block = [f"\n### {time.strftime('%Y-%m-%d')} — advisor run ({len(ideas)} proposals, ranked)\n"]
    for i in ideas:
        block.append(
            f"- 🟡 **{i.title}** — elo {i.elo:.0f} · feasibility {i.feasibility:.2f} · "
            f"novelty {i.novelty:.2f}\n"
            f"  - concept: {i.concept}\n"
            f"  - why: {i.why}\n"
            f"  - proposed: {i.proposed_change}\n"
            f"  - critique: {i.critique}\n"
            f"  - 🔎 search: {'; '.join(i.search_queries)}\n")
    addition = ("\n" + SECTION + "\n" if SECTION not in body else "") + "".join(block)
    with open(plan, "a", encoding="utf-8") as fh:
        fh.write(addition)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    _load_dotenv()
    chat, backend = pb.auto_text_completer()
    # short per-fetch timeout so book ingestion can never stall the cycle
    box = Toolbox(knowledge=KnowledgeBase(fetch=lambda u: _http_get(u, timeout=6)))
    adv = IdeationAdvisor(toolbox=box, llm_chat=chat)
    # ground novelty in the book corpus (best-effort, hard-capped at 40s)
    t = threading.Thread(target=lambda: _safe(box.knowledge.ingest_manifest, max_chars=30000),
                         daemon=True)
    t.start(); t.join(25)
    ideas = adv.generate(n=n)
    append_to_plan(ideas)
    print(f"[advisor_cycle] backend={backend or 'offline'} — appended {len(ideas)} 🟡 proposals to PLAN.md:")
    for i in ideas:
        print(f"  - {i.title}  (elo {i.elo:.0f}, feas {i.feasibility:.2f}, nov {i.novelty:.2f})")
    print("[advisor_cycle] NOTHING decided or built — review in PLAN.md / the dashboard Advisor panel.")


if __name__ == "__main__":
    main()
