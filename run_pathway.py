"""Watch the first hardcoded pathway run, hop by hop (build-tracker step 2).

This is the walking-skeleton "watch it run" view in terminal form — the textual
precursor to the §8 dashboard's living-graph + belief-stream views, deliberately
committing **no** dashboard tech stack (that choice is still open in PLAN.md §8).
It runs the step-2 Connector over one synthetic (T, D) sequence and shows the
signal bus threading through the nodes while each emits a Belief into the stream.

Run: python3 run_pathway.py
"""
from __future__ import annotations

import json

import numpy as np

import pattern_brain as pb


def synthetic_sequence(T=240, D=3, seed=7):
    """A generic regime-switching multivariate sequence (no domain meaning)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = np.zeros((T, D))
    third = T // 3
    X[:third] = (0.05 * t[:third])[:, None] + rng.normal(0, 0.3, (third, D))
    X[third:2 * third] = rng.normal(0, 0.6, (third, D))
    X[2 * third:] = np.sin(0.35 * t[2 * third:])[:, None] + rng.normal(0, 0.2, (T - 2 * third, D))
    return X


_PAYLOAD_KEYS = ("action", "z", "current_state", "next_state", "n_clusters",
                 "n_noise", "mean_abs_change", "form", "dominant_freq")


def _summary(belief) -> str:
    parts = [f"{k}={belief.payload[k]}" for k in _PAYLOAD_KEYS if k in belief.payload]
    return ", ".join(parts[:3])


def main():
    X = synthetic_sequence()
    conn = pb.default_connector()
    print(f"Pattern Brain — Connector Intelligence v0 (walking skeleton)")
    print(f"signal bus in: (T={X.shape[0]}, D={X.shape[1]})  generic synthetic data\n")
    print(f"  pathway:  {'  ->  '.join(conn.pathway)}\n")

    res = conn.run(X)

    print(f"  {'#':<2} {'node':18} {'layer':10} {'belief':10} {'conf':>6}  bus  payload")
    print(f"  {'-'*2} {'-'*18} {'-'*10} {'-'*10} {'-'*6}  ---  {'-'*30}")
    for h in res.hops:
        bus = "rewr" if h.transformed else " tap"   # transformer rewrites vs. taps the bus
        print(f"  {h.step:<2} {h.node_type:18} {h.layer:10} {h.belief.type:10} "
              f"{h.belief.confidence:6.3f}  {bus}  {_summary(h.belief)}")

    print(f"\n  => terminal output: {res.output.type} "
          f"(conf={res.output.confidence:.3f})  {_summary(res.output)}")
    print(f"     ran in {res.elapsed_ms:.2f} ms across {len(res.hops)} hops\n")

    # The full auditable JSON trace — exactly what the §8 dashboard would consume.
    print("  belief-stream trace (JSON, dashboard-ready):")
    trace = res.to_dict()
    # print a compact view: drop bulky payload arrays for readability
    for hop in trace["hops"]:
        hop["belief"]["payload"] = {
            k: v for k, v in hop["belief"]["payload"].items()
            if not isinstance(v, list)
        }
    print(json.dumps(trace, indent=2)[:1400] + "\n  ...")


if __name__ == "__main__":
    main()
