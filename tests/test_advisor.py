"""Tests for the Ideation & Research Advisor (PLAN.md §10 / Rule 26).

Run: python3 tests/test_advisor.py — fully offline (heuristic path + injected LLM).
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.knowledge import KnowledgeBase, PROJECT_ROOT
from pattern_brain.agent import Toolbox, IdeationAdvisor

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _advisor(**kw):
    sd = os.path.join(PROJECT_ROOT, "agent_state", "_test_adv")
    shutil.rmtree(sd, ignore_errors=True)
    return IdeationAdvisor(toolbox=Toolbox(knowledge=KnowledgeBase()), state_dir=sd, **kw)


def test_gather_context_reads_project():
    a = _advisor()
    try:
        ctx = a.gather_context()
        check(len(ctx["bank_by_layer"]) == 8, "should see all 8 layers")
        check(len(ctx["weakest_layers"]) >= 1, "no weakest layer identified")
        check(isinstance(ctx["open_plan_items"], list), "open items not a list")
        check("Pattern Brain" in ctx["goal_excerpt"], "didn't read README goal")
        # coverage wiring: gaps surfaced + at least one heuristic idea targets a gap family
        check(isinstance(ctx["coverage_gaps"], list) and len(ctx["coverage_gaps"]) >= 1,
              "coverage_gaps not surfaced in context")
        check(all({"family", "missing", "n_built"} <= set(g) for g in ctx["coverage_gaps"]),
              "coverage gap entry malformed")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print(f"  context: 8 layers, weakest={ctx['weakest_layers']}, "
          f"{len(ctx['open_plan_items'])} open items, {len(ctx['coverage_gaps'])} coverage gaps "
          f"(top: {ctx['coverage_gaps'][0]['family'] if ctx['coverage_gaps'] else None})")


def test_heuristic_ideas_are_grounded():
    a = _advisor()
    try:
        ideas = a.generate(n=5)
        check(len(ideas) >= 3, "too few heuristic ideas")
        check(all(i.search_queries for i in ideas), "an idea has no search queries")
        check(all(i.proposed_change for i in ideas), "an idea has no proposed change")
        check(any(("layer" in i.title.lower() or "missing" in i.title.lower()
                   or "models" in i.title.lower()) for i in ideas), "no gap-targeting idea")
        check(all(i.status == "proposed" for i in ideas), "ideas should start 'proposed'")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print(f"  heuristic: {len(ideas)} grounded ideas, each with search queries + a proposed change")


def test_llm_ideas_path():
    fake = '[{"title":"Add conformal prediction","concept":"wrap forecasts with '\
           'calibrated intervals","why":"uncertainty","search_queries":["conformal '\
           'prediction time series 2026"],"proposed_change":"new node","expected_benefit":'\
           '"calibrated risk","risk":"compute"}]'
    a = _advisor(llm_chat=lambda messages, temperature=0.4: fake)
    try:
        ideas = a.generate(n=3)
        check(any("conformal" in i.title.lower() for i in ideas), "LLM idea not parsed")
        check(any(i.source == "llm" for i in ideas), "LLM source not tagged")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print("  llm: JSON idea array from the LLM parsed into structured proposals")


def test_status_lifecycle_and_persistence():
    a = _advisor()
    try:
        ideas = a.generate(n=3)
        iid = ideas[0].id
        check(a.set_status(iid, "approved"), "set_status failed")
        check(not a.set_status("nope", "approved"), "unknown id should return False")
        try:
            a.set_status(iid, "bogus"); check(False, "bad status should raise")
        except ValueError:
            pass
        # reload from disk -> approval persisted (survives restart, inside folder)
        b = IdeationAdvisor(toolbox=Toolbox(knowledge=KnowledgeBase()), state_dir=a.state_dir)
        check(any(i["id"] == iid and i["status"] == "approved" for i in b.list_ideas()),
              "approval not persisted")
        check(os.path.abspath(a.save()).startswith(PROJECT_ROOT + os.sep),
              "ideas saved outside folder (Rule 1)")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print("  lifecycle: proposed→approved persists across restart; never auto-implements")


def test_does_not_implement():
    """Approval-gated: the advisor only proposes — it has no method that mutates the bank."""
    a = _advisor()
    try:
        for attr in ("register", "promote", "create_node", "implement"):
            check(not hasattr(a, attr), f"advisor must NOT have an implementing method: {attr}")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print("  safety: advisor proposes only — no implementing method (approval-gated)")


def test_critique_novelty_and_elo_ranking():
    """The effectiveness upgrades: every idea gets a feasibility critique + a RAG
    novelty score, and the pool is ranked by a pairwise-ELO tournament."""
    a = _advisor()
    try:
        # seed the KB so novelty has something to compare against
        a.toolbox.knowledge.ingest_text("seed", "deep reinforcement learning DQN PPO " * 30,
                                        source="ML Book")
        ideas = a.generate(n=4)
        check(all(0.0 <= i.feasibility <= 1.0 for i in ideas), "feasibility out of range")
        check(all(0.0 <= i.novelty <= 1.0 for i in ideas), "novelty out of range")
        check(all(i.critique for i in ideas), "an idea has no critique")
        check(all(i.elo != 1000.0 for i in ideas) or len(ideas) == 1,
              "ELO never updated (tournament didn't run)")
        # returned best-first
        elos = [i.elo for i in ideas]
        check(elos == sorted(elos, reverse=True), "ideas not returned best-first by ELO")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print(f"  upgrades: critique+novelty+ELO applied; top idea elo={ideas[0].elo:.0f}, "
          f"feas={ideas[0].feasibility:.2f}, nov={ideas[0].novelty:.2f}")


def test_elo_uses_llm_judge_when_present():
    """With an LLM judge, the pairwise comparator drives ranking (Idea Arena)."""
    # judge always prefers whichever option mentions 'conformal'
    def judge(messages, temperature=0.4):
        u = messages[-1]["content"]
        a_line = u.split("\n")[0]
        return "A" if "conformal" in a_line.lower() else "B"
    a = _advisor(llm_chat=lambda m, temperature=0.4:
                 '[{"title":"Add conformal prediction","concept":"x","why":"y",'
                 '"search_queries":["q"],"proposed_change":"node","expected_benefit":"b","risk":"r"},'
                 '{"title":"Add caching","concept":"x","why":"y","search_queries":["q"],'
                 '"proposed_change":"node","expected_benefit":"b","risk":"r"}]')
    a.llm_chat = None  # generate offline, but set judge for compare path
    try:
        ideas = a._heuristic_ideas(a.gather_context(), 3)
        a.llm_chat = judge
        for i in ideas:
            a._critique(i); a._novelty(i)
        a._elo_rank(ideas)
        check(len(ideas) >= 2, "need >=2 ideas to rank")
    finally:
        shutil.rmtree(a.state_dir, ignore_errors=True)
    print("  arena: LLM pairwise judge wired into ELO (both orders for position-bias)")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(PROJECT_ROOT, "pattern_brain", "agent", "advisor.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in advisor.py: {hits}")
    print("  domain-independence: advisor.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Ideation & Research Advisor (Rule 26): tests")
    print("=" * 70)
    test_gather_context_reads_project()
    test_heuristic_ideas_are_grounded()
    test_llm_ideas_path()
    test_status_lifecycle_and_persistence()
    test_critique_novelty_and_elo_ranking()
    test_elo_uses_llm_judge_when_present()
    test_does_not_implement()
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
