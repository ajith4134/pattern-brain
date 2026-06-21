"""Tests for the ML Engineer Agent loop (PLAN.md §9, ENG-3).

Run: python3 tests/test_agent_loop.py
Fully offline: synthetic data, dependency-free KB, no LLM (templated chat) — plus
one test that injects a fake LLM text-completer to prove the LLM path.
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.knowledge import KnowledgeBase, PROJECT_ROOT
from pattern_brain.agent import Toolbox, MLEngineerAgent
from pattern_brain.registry import all_node_types

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _agent(**kw):
    sd = os.path.join(PROJECT_ROOT, "agent_state", "_test")
    shutil.rmtree(sd, ignore_errors=True)
    box = Toolbox(knowledge=KnowledgeBase(),
                  data_dir=os.path.join(PROJECT_ROOT, "data", "_test_loop"))
    return MLEngineerAgent(toolbox=box, state_dir=sd, population=5, generations=2, **kw)


def _cleanup(a):
    shutil.rmtree(a.state_dir, ignore_errors=True)
    shutil.rmtree(a.toolbox.data_dir, ignore_errors=True)


def test_single_step_ooda():
    a = _agent(seed=1)
    try:
        r = a.step()
        check(r.step == 1, "step counter wrong")
        check(r.feature in a.features, f"bad feature {r.feature}")
        check(r.mode in ("draft", "improve"), f"bad mode {r.mode}")
        check("tested" in r.act or "error" in r.act, "act summary malformed")
        check(isinstance(r.finding, str) and r.finding, "no finding text")
        check(len(a.feed) == 1, "feed not appended")
    finally:
        _cleanup(a)
    print(f"  step: OODA cycle ran -> {r.feature}[{r.mode}]: {r.finding[:60]}")


def test_run_promotes_into_bank():
    a = _agent(seed=4)
    try:
        before = len(all_node_types())
        a.run(6)
        check(a.steps_done == 6, f"expected 6 steps, got {a.steps_done}")
        check(len(all_node_types()) >= before, "bank shrank")
        # over several steps with admissions, at least track promotions coherently
        check(a.promotions == sum(1 for f in a.feed if f["promoted"]),
              "promotion counter disagrees with feed")
        print(f"  run: 6 steps, {a.promotions} node(s) promoted into the bank "
              f"({before} -> {len(all_node_types())})")
    finally:
        _cleanup(a)


def test_stop_flag_halts_daemon():
    a = _agent(seed=2)
    try:
        a.request_stop()
        ran = a.run_forever(max_steps=10)
        check(ran == 0, f"STOP flag should halt before any step, ran {ran}")
        a.clear_stop()
        ran2 = a.run_forever(max_steps=2)
        check(ran2 == 2, f"after clear, expected 2 steps, ran {ran2}")
    finally:
        _cleanup(a)
    print("  daemon: STOP flag halts immediately; clear resumes; max_steps budget honored")


def test_budget_cap_max_promotions():
    a = _agent(seed=3, max_promotions=1)
    try:
        a.run_forever(max_steps=20)
        check(a.promotions <= 1, f"max_promotions cap breached: {a.promotions}")
    finally:
        _cleanup(a)
    print(f"  budget: max_promotions cap respected ({a.promotions} <= 1)")


def test_state_persistence_inside_folder():
    a = _agent(seed=5)
    try:
        a.run(3)
        path = a.save_state()
        check(os.path.abspath(path).startswith(PROJECT_ROOT + os.sep),
              "state saved OUTSIDE the folder (Rule 1)")
        # build a fresh agent pointing at the SAME state dir, WITHOUT wiping it
        b = MLEngineerAgent(toolbox=Toolbox(knowledge=KnowledgeBase()),
                            state_dir=a.state_dir, population=5, generations=2, seed=5)
        check(b.load_state() and b.steps_done == 3, "state did not reload")
    finally:
        _cleanup(a)
    print("  persistence: loop state saved inside folder + reloaded (survives restart)")


def test_chat_offline_and_with_llm():
    a = _agent(seed=6)
    try:
        a.toolbox.knowledge.ingest_text("b", "cross validation prevents data leakage " * 30,
                                        source="ML Book")
        reply = a.chat("what is the state of the bank?")
        check(isinstance(reply, str) and "ML Engineer" in reply, "offline chat reply malformed")
        check("model nodes" in reply, "offline reply missing bank summary")
        # inject a fake LLM text-completer
        seen = {}

        def fake_llm(messages, temperature=0.4):
            seen["system"] = messages[0]["content"]
            return "Bank looks healthy; I'd run feature2 next."
        a.llm_chat = fake_llm
        reply2 = a.chat("what next?")
        check(reply2.startswith("Bank looks healthy"), "LLM chat path not used")
        check("The ML Engineer" in seen["system"], "persona not in system prompt")
    finally:
        _cleanup(a)
    print("  chat: offline templated reply works; injected LLM path used + persona grounded")


def test_llm_researcher_decision():
    a = _agent(seed=7)
    try:
        def fake_llm(messages, temperature=0.4):
            return '{"feature": "feature2_algorithm", "mode": "improve", "why": "test"}'
        a.llm_chat = fake_llm
        r = a.step()
        check(r.feature == "feature2_algorithm", f"LLM feature choice ignored: {r.feature}")
        check(r.orient == "test" or "test" in r.orient, "LLM rationale not used")
    finally:
        _cleanup(a)
    print("  researcher: LLM JSON decision drives feature/mode selection")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(PROJECT_ROOT, "pattern_brain", "agent", "engineer.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in engineer.py: {hits}")
    print("  domain-independence: engineer.py has no candle/ohlcv/orderbook coupling")


def main():
    print("=" * 70)
    print("Pattern Brain — ML Engineer Agent loop (ENG-3): tests")
    print("=" * 70)
    test_single_step_ooda()
    test_run_promotes_into_bank()
    test_stop_flag_halts_daemon()
    test_budget_cap_max_promotions()
    test_state_persistence_inside_folder()
    test_chat_offline_and_with_llm()
    test_llm_researcher_decision()
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
