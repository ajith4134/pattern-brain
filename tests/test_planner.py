"""Tests for Phase 8 slice 8 — the LLM experiment planner (PLAN.md §12).

Run: python3 tests/test_planner.py  (light stack; LLM is a fake — no network).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.search import DAGSearch
from pattern_brain.leaderboard import Leaderboard
from pattern_brain.network import DAGSpec, StackedDAG
from pattern_brain.agent.planner import DAGPlanner

FAILS = []
POOL = ["drift_forecast", "theta_forecast", "naive_mean_forecast"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _ar(n=130, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    return a.reshape(-1, 1)


def _search(seed=0):
    return DAGSearch(_ar(seed=seed), leaderboard=Leaderboard(":memory:"), base_pool=POOL,
                     max_base=3, min_train=35, n_samples=40, dataset_id="p", seed=seed)


def _json_llm(messages):
    return ('proposals:\n'
            '[{"layers": [["drift_forecast","theta_forecast"]], "combiner": "stacked"},'
            ' {"layers": [["naive_mean_forecast"]], "combiner": "mixture"},'
            ' {"layers": [["NOT_A_REAL_NODE","drift_forecast"]], "combiner": "weird"}]')


def test_llm_proposals_parsed_and_sanitized():
    s = _search()
    specs = DAGPlanner(s, llm_chat=_json_llm).propose(3)
    check(len(specs) == 3, f"should parse 3 proposals, got {len(specs)}")
    # invalid node dropped, invalid combiner repaired to a legal value
    third = specs[2]
    check(all(nd in POOL for layer in third.layers for nd in layer),
          f"out-of-pool node not dropped: {third.layers}")
    check(all(sp.combiner in ("mixture", "stacked") for sp in specs), "combiner not sanitized")
    print("  LLM JSON proposals parsed; out-of-pool node dropped + bad combiner repaired")


def test_garbage_llm_falls_back_to_heuristic():
    s = _search()
    s.random_search(2)                                  # seed the leaderboard
    specs = DAGPlanner(s, llm_chat=lambda m: "I'm sorry, I can't do that.").propose(4)
    check(len(specs) == 4 and all(DAGPlanner(s)._valid(sp) for sp in specs),
          "garbage LLM should fall back to valid heuristic proposals")
    print("  garbage LLM reply → clean heuristic fallback (4 valid specs)")


def test_heuristic_offline_no_llm():
    s = _search()
    s.random_search(2)
    specs = DAGPlanner(s, llm_chat=None).propose(5)
    check(len(specs) == 5, "offline heuristic should still propose n specs")
    check(all(DAGPlanner(s)._valid(sp) for sp in specs), "heuristic produced invalid spec")
    print("  offline (no LLM): heuristic proposes valid specs from the leaderboard best")


def test_llm_cannot_cheat_the_score():
    """The LLM only PROPOSES; the recorded score must equal an independent recompute
    by StackedDAG — the LLM can never inject an accuracy."""
    s = _search(seed=1)
    DAGPlanner(s, llm_chat=_json_llm).plan_and_search(rounds=1, n_per_round=3)
    row = s.lb.top(10)[0]
    spec = DAGSpec.from_dict(json.loads(row["spec_json"]))
    fresh = StackedDAG(spec, s.min_train, s.n_samples, s.seed).score(s.X)
    check(abs(row["crps_skill"] - fresh["crps_skill"]) < 1e-9,
          "recorded skill must equal an independent StackedDAG recompute (LLM cannot set it)")
    check(-1.0 <= row["crps_skill"] <= 1.0, "skill must be a real bounded score")
    print(f"  LLM cannot cheat: recorded skill {row['crps_skill']:.4f} == independent recompute")


def test_plan_and_search_records_and_returns_best():
    s = _search(seed=2)
    best = DAGPlanner(s, llm_chat=_json_llm).plan_and_search(rounds=2, n_per_round=3)
    check(s.lb.count() >= 4, f"plan_and_search should record multiple trials, got {s.lb.count()}")
    check(best is not None and np.isfinite(best["crps_skill"]), "best malformed")
    print(f"  plan_and_search: {s.lb.count()} nets tried, best skill={best['crps_skill']:.3f}")


def test_agent_integration_offline():
    from pattern_brain.agent.engineer import MLEngineerAgent
    agent = MLEngineerAgent(llm_chat=None)              # offline → heuristic planner
    best = agent.plan_dag_search(_ar(seed=3), rounds=1, n_per_round=2, base_pool=POOL,
                                 min_train=35, n_samples=40, dataset_id="agent_test")
    check(best is not None and np.isfinite(best["crps_skill"]), "agent DAG search failed")
    mems = agent.toolbox.recall_memory("DAG search", k=3)
    check(any("DAG search" in str(m) for m in mems), "agent should remember the DAG search")
    print(f"  agent.plan_dag_search (offline): best skill={best['crps_skill']:.3f}, remembered")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 8: LLM experiment planner: tests")
    print("=" * 70)
    test_llm_proposals_parsed_and_sanitized()
    test_garbage_llm_falls_back_to_heuristic()
    test_heuristic_offline_no_llm()
    test_llm_cannot_cheat_the_score()
    test_plan_and_search_records_and_returns_best()
    test_agent_integration_offline()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
