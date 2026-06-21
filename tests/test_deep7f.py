"""Tests for Phase 7f deep-tier nodes (PLAN.md §11, Block 57).

Run: .venv/bin/python tests/test_deep7f.py  (needs torch; skips cleanly without it).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.nodes import deep as _deep

FAILS = []
FORECASTERS = ["patchtst_forecaster", "mamba_forecaster", "nbeats_forecaster"]
DENOISERS = ["gat_denoise", "graphsage_denoise", "diffusion_denoise"]
RL = ["dqn_policy", "ppo_policy"]
ALL7F = FORECASTERS + DENOISERS + RL


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7f deep-tier nodes: tests")
    print("=" * 70)
    if not _deep.TORCH_AVAILABLE:
        print("  torch not installed -> 7f deep nodes skip cleanly (light bank unaffected)")
        print("ALL CHECKS PASSED (skipped)")
        return

    from pattern_brain.registry import all_node_types, create
    from pattern_brain.interlingua import validate_belief
    reg = all_node_types()
    x = np.cumsum(np.random.default_rng(2).normal(size=160)).reshape(-1, 1)

    for n in ALL7F:
        check(n in reg, f"{n} not registered (torch present)")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
    print(f"  all {len(ALL7F)} deep-7f nodes registered + conform")

    for n in FORECASTERS:
        nv = create(n).predict(x).payload["next_vector"]
        check(len(nv) == 1 and np.isfinite(nv[0]), f"{n} bad forecast {nv}")
    print(f"  forecasters: PatchTST/Mamba/N-BEATS produce finite one-step forecasts")

    for n in DENOISERS:
        p = create(n).predict(x).payload
        check(len(p["series"]) == len(x) and np.isfinite(p["residual"]),
              f"{n} bad denoised output")
    print(f"  denoisers: GAT/GraphSAGE/diffusion return a full cleaned series + finite residual")

    for n in RL:
        p = create(n).predict(x).payload
        check(p["action"] in (-1, 0, 1), f"{n} action out of range: {p['action']}")
        vals = p.get("q_values") or p.get("policy")
        check(len(vals) == 3, f"{n} should expose 3 action values")
    print(f"  deep-RL: DQN/PPO emit a valid action in short/flat/long + 3 action values")

    # light interpreter must be unaffected (deep nodes optional) — sanity on count
    check(len(reg) >= 150, f"torch bank unexpectedly small: {len(reg)}")
    print(f"  bank size with torch: {len(reg)} nodes")

    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
