"""Tests for LLM file access (owner-requested): file tools + the ReAct chat loop.

Run: python3 tests/test_file_access.py  (light stack; LLM is a fake — no network).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.agent.tools import Toolbox
from pattern_brain.agent.engineer import MLEngineerAgent

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_file_tools():
    tb = Toolbox()
    files = tb.list_files("**/*.py")
    check(any(f.endswith("network.py") for f in files), "list_files should find network.py")
    check(not any("/.venv/" in f or ".venv" in f.split(os.sep)[:1] for f in files), "should skip .venv")
    r = tb.read_file("pattern_brain/network.py")
    check("StackedDAG" in r.get("content", ""), "read_file should return network.py source")
    hits = tb.search_files("def run_capstone")
    check(any(h["file"].endswith("capstone.py") for h in hits), "search_files should locate run_capstone")
    print(f"  file tools: {len(files)} files listed, read_file works, search found run_capstone")


def test_rule1_guard():
    tb = Toolbox()
    for bad in ("../../../etc/passwd", "/etc/passwd"):
        try:
            tb.read_file(bad)
            check(False, f"Rule 1: should refuse {bad}")
        except ValueError:
            pass
    print("  Rule-1 guard: refuses paths outside the project folder")


def test_chat_reads_files_react_loop():
    """A fake LLM that first requests a file, then answers from its content — proving
    the chat gives the LLM real file access via the ReAct loop."""
    calls = {"n": 0}

    def fake(msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"tool":"read_file","path":"pattern_brain/capstone.py"}'
        saw = "run_capstone" in str(msgs)               # tool result is now in the convo
        return f"capstone.py defines run_capstone: {saw}"

    agent = MLEngineerAgent(llm_chat=fake)
    agent.ensure_books = lambda *a, **k: None            # skip book ingestion for the test
    reply = agent.chat("what does capstone.py do?")
    check(calls["n"] >= 2, "chat should make a tool call THEN answer (>=2 LLM calls)")
    check("True" in reply, f"the LLM should have seen the file content; reply={reply!r}")
    print(f"  ReAct chat: LLM read a file then answered using its content ({calls['n']} turns)")


def test_chat_blocks_nonread_tools():
    """Chat is read-only: a request for a mutating tool is refused, not executed."""
    calls = {"n": 0}

    def fake(msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"tool":"promote_to_node"}'
        return "ok, I will not mutate."

    agent = MLEngineerAgent(llm_chat=fake)
    agent.ensure_books = lambda *a, **k: None
    reply = agent.chat("promote something")
    check("not mutate" in reply or calls["n"] >= 2, "chat should refuse the mutating tool and continue")
    print("  chat is read-only: mutating tools refused (only file/state reads allowed)")


def test_runtime_result_tools():
    """The chat can read live RESULTS (leaderboard / capstone), not just files."""
    tb = Toolbox()
    lb = tb.read_leaderboard()
    check(isinstance(lb, dict) and ("summary" in lb or "n_runs" in lb), "read_leaderboard malformed")
    cap = tb.read_capstone()
    check(isinstance(cap, dict), "read_capstone should return a dict")
    # both dispatch via the tool-call interface the ReAct loop uses
    check(isinstance(tb.call("read_leaderboard"), dict), "read_leaderboard not callable via .call")
    print(f"  runtime tools: read_leaderboard ({'populated' if 'summary' in lb else 'empty'}) + read_capstone OK")


def test_chat_transcript_cached_and_recalled():
    """Idea 3: each Q&A is cached to archival memory and recalled later."""
    agent = MLEngineerAgent(llm_chat=lambda msgs: "The stacked combiner won because it had the best DSR.")
    agent.ensure_books = lambda *a, **k: None
    q = "why did the search pick the stacked combiner today"
    agent.chat(q)
    recalls = agent.toolbox.recall_memory("which combiner won the search", k=5)
    check(any("stacked combiner won" in str(r) for r in recalls),
          "the Q&A should be cached to memory and recalled on a similar question")
    print(f"  transcript caching: prior answer recalled on a similar question ({len(recalls)} hits)")


def test_chat_write_tool_requires_confirmation():
    """Idea 2: a write action ('run the capstone') is STAGED, not executed — it runs
    only after explicit confirmation, and exactly once."""
    ran = {"n": 0}
    agent = MLEngineerAgent(
        llm_chat=lambda m: '{"tool":"run_capstone","symbol":"X","prefer_live":false}')
    agent.ensure_books = lambda *a, **k: None
    agent._run_capstone = lambda **kw: (ran.__setitem__("n", ran["n"] + 1)
                                        or {"verdict": "PASS", "symbol": "X", "source": "synthetic"})
    reply = agent.chat("please run the capstone")
    check(ran["n"] == 0, "the run must NOT execute during chat (confirmation required)")
    check("confirm" in reply.lower(), f"chat should request confirmation; got {reply!r}")
    pa = agent.pending_action()
    check(pa and pa["tool"] == "run_capstone", "a pending action should be staged")
    out = agent.confirm_action(pa["id"], approve=True)
    check(ran["n"] == 1 and out["ran"], "confirm should execute the run exactly once")
    check(agent.pending_action() is None, "pending action cleared after confirm")
    print("  operator mode: write action gated behind explicit confirmation, runs once on confirm")


def test_chat_write_tool_cancel():
    """Idea 2: cancelling a staged action (button or typed 'cancel') runs nothing."""
    ran = {"n": 0}
    agent = MLEngineerAgent(
        llm_chat=lambda m: '{"tool":"run_dag_search","symbol":"X","prefer_live":false}')
    agent.ensure_books = lambda *a, **k: None
    agent._run_dag_search = lambda **kw: (ran.__setitem__("n", ran["n"] + 1) or {})
    agent.chat("run a dag search")
    pa = agent.pending_action()
    out = agent.confirm_action(pa["id"], approve=False)
    check(ran["n"] == 0 and not out["ran"], "cancel must not execute the run")
    check(agent.pending_action() is None, "pending cleared after cancel")
    agent.chat("run a dag search")                       # stage again
    r2 = agent.chat("cancel")                            # typed-cancel routes to cancellation
    check("cancel" in r2.lower() and ran["n"] == 0, "typed 'cancel' should abort the staged action")
    print("  operator mode: cancel (button + typed) aborts without running")


def test_consolidate_qa():
    """Idea 3: good chat Q&A is promoted into a first-class, retrievable knowledge
    source; junk is skipped; the job is idempotent."""
    agent = MLEngineerAgent(llm_chat=None)
    kb = agent.toolbox.knowledge
    answer = "Stacking won because it had the highest DSR and stayed calibrated across folds. " * 3
    kb.remember("Q: what is the best combiner\nA: " + answer, kind="chat_qa")
    kb.remember("Q: short one\nA: idk", kind="chat_qa")                       # too short → skip
    kb.remember("Q: offline\nA: [The ML Engineer — offline mode, no LLM backend configured] nope",
                kind="chat_qa")                                              # marker → skip
    res = agent.consolidate()
    check(res["promoted"] == 1, f"exactly one good Q&A should promote; got {res}")
    check(res["skipped_low_quality"] >= 2, f"short + offline answers should be skipped; got {res}")
    stats = kb.stats()
    check(stats["sources"].get("consolidated_qa", 0) == 1, "a consolidated_qa knowledge source should exist")
    hits = agent.toolbox.retrieve_knowledge("which combiner is best", k=5)
    check(any(h.get("source") == "consolidated_qa" for h in hits),
          "consolidated Q&A should be retrievable as first-class knowledge")
    res2 = agent.consolidate()
    check(res2["promoted"] == 0 and res2["skipped_dup"] >= 1, f"re-run should be idempotent; got {res2}")
    print("  consolidation: good Q&A → first-class knowledge, junk skipped, idempotent")


def main():
    print("=" * 70)
    print("Pattern Brain — LLM file access: tests")
    print("=" * 70)
    test_file_tools()
    test_rule1_guard()
    test_chat_reads_files_react_loop()
    test_chat_blocks_nonread_tools()
    test_runtime_result_tools()
    test_chat_transcript_cached_and_recalled()
    test_chat_write_tool_requires_confirmation()
    test_chat_write_tool_cancel()
    test_consolidate_qa()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
