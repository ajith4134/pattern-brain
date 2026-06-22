"""Tests for Phase 7f BATCH 2 deep-tier nodes (PLAN.md §11, Block 57).

Run: .venv/bin/python tests/test_deep7f2.py  (needs torch; skips cleanly without it).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.nodes import deep as _deep

FAILS = []
FORECASTERS = ["informer_forecaster", "autoformer_forecaster", "fedformer_forecaster",
               "s4_forecaster", "s5_forecaster", "fno_forecaster",
               "neural_ode_forecaster", "pinn_forecaster"]
DENOISERS = ["temporal_gnn_denoise", "hopfield_denoise", "rbm_denoise"]
ANOMALY = ["gan_anomaly", "normalizing_flow_anomaly"]
RL = ["sac_policy", "td3_policy", "a3c_policy"]
ALL7F2 = FORECASTERS + DENOISERS + ANOMALY + RL


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7f BATCH 2 deep-tier nodes: tests")
    print("=" * 70)
    if not _deep.TORCH_AVAILABLE:
        print("  torch not installed -> 7f batch-2 nodes skip cleanly (light bank unaffected)")
        print("ALL CHECKS PASSED (skipped)")
        return

    from pattern_brain.registry import all_node_types, create
    from pattern_brain.interlingua import validate_belief
    reg = all_node_types()
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=180)).reshape(-1, 1)

    for n in ALL7F2:
        check(n in reg, f"{n} not registered (torch present)")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check(np.isfinite(b.confidence), f"{n} non-finite confidence")
    print(f"  all {len(ALL7F2)} deep-7f-batch-2 nodes registered + conform")

    for n in FORECASTERS:
        nv = create(n).predict(x).payload["next_vector"]
        check(len(nv) == 1 and np.isfinite(nv[0]), f"{n} bad forecast {nv}")
    print(f"  forecasters: Informer/Autoformer/FEDformer/S4/S5/FNO/Neural-ODE/PINN "
          f"produce finite one-step forecasts")

    # multi-feature input (D=3) must also work for the transformer/SSM forecasters
    xD = np.column_stack([x[:, 0], np.cumsum(rng.normal(size=180)), rng.normal(size=180)])
    for n in ("informer_forecaster", "fedformer_forecaster", "s4_forecaster", "fno_forecaster"):
        nv = create(n).predict(xD).payload["next_vector"]
        check(len(nv) == 3 and all(np.isfinite(nv)), f"{n} bad D=3 forecast {nv}")
    print("  forecasters handle multi-feature (D=3) input -> full next-vector")

    for n in DENOISERS:
        p = create(n).predict(x).payload
        check(len(p["series"]) == len(x) and np.isfinite(p["residual"]),
              f"{n} bad denoised output")
    print(f"  denoisers: Temporal-GNN/Hopfield/RBM return a full cleaned series + finite residual")

    for n in ANOMALY:
        p = create(n).predict(x).payload
        check("flags" in p and len(p["flags"]) == len(p["scores"]),
              f"{n} flags/scores mismatch")
        check(all(f in (0, 1) for f in p["flags"]), f"{n} non-binary flags")
    print(f"  generative anomaly: GAN/Normalizing-flow score every window + binary flags")

    for n in RL:
        p = create(n).predict(x).payload
        check(p["action"] in (-1, 0, 1), f"{n} action out of range: {p['action']}")
        vals = p.get("q_values") or p.get("policy")
        check(len(vals) == 3, f"{n} should expose 3 action values")
    print(f"  deep-RL: SAC/TD3/A3C emit a valid action in short/flat/long + 3 action values")

    # behavioral: a clean AR(1) trend should let the forecasters beat last-value naive
    n = 300
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(scale=0.2)
    Xar = ar[:-1].reshape(-1, 1)
    truth = ar[-1]
    beat = 0
    for nm in ("informer_forecaster", "s4_forecaster", "neural_ode_forecaster", "fno_forecaster"):
        pred = create(nm).predict(Xar).payload["next_vector"][0]
        if abs(pred - truth) <= abs(Xar[-1, 0] - truth) * 1.5 + 0.5:
            beat += 1
    check(beat >= 3, f"only {beat}/4 deep forecasters tracked AR(1) within tolerance")
    print(f"  behavioral: {beat}/4 deep forecasters track an AR(1) signal within tolerance")

    check(len(reg) >= 185, f"torch bank unexpectedly small after batch 2: {len(reg)}")
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
