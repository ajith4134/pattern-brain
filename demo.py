"""See the starter model bank working, in detail (build-tracker step 1).

Runs the whole bank on one synthetic regime-switching sequence and prints each
node's Belief, grouped by functional layer. This is the textual precursor to the
§8 dashboard's "Belief-space stream" and "Node Inspector" views.

Run: python3 demo.py
"""
from __future__ import annotations

import numpy as np

import pattern_brain as pb


def synthetic_sequence(T=240, D=3, seed=7):
    """A generic regime-switching multivariate sequence (no domain meaning)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = np.zeros((T, D))
    third = T // 3
    # three regimes: trend up, mean-reverting noise, oscillation
    X[:third] = (0.05 * t[:third])[:, None] + rng.normal(0, 0.3, (third, D))
    X[third:2 * third] = rng.normal(0, 0.6, (third, D))
    X[2 * third:] = np.sin(0.35 * t[2 * third:])[:, None] + rng.normal(0, 0.2, (T - 2 * third, D))
    return X


def main():
    X = synthetic_sequence()
    y = (np.arange(X.shape[0]) > X.shape[0] // 2).astype(int)  # labels for supervised nodes

    print(f"Pattern Brain — model bank demo on a synthetic (T={X.shape[0]}, D={X.shape[1]}) sequence")
    print(f"{len(pb.all_node_types())} nodes across {len(pb.layers())} functional layers\n")

    order = ["signal", "noise", "pattern", "sequence",
             "probability", "equation", "decision", "rl"]
    for layer in order:
        print(f"=== {layer.upper()} " + "=" * (62 - len(layer)))
        for nt in pb.by_layer(layer):
            node = pb.create(nt)
            belief = node.process(X, y) if node.requires_y else node.process(X)
            # one-line readable summary of the most informative payload fields
            keys = [k for k in ("action", "direction", "next_state", "current_state",
                                "form", "expression", "arm", "n_clusters", "n_anomalies",
                                "dominant_freq", "estimate") if k in belief.payload]
            summary = ", ".join(f"{k}={belief.payload[k]}" for k in keys[:2])
            print(f"  {nt:24} {belief.type:11} conf={belief.confidence:5.3f}  {summary}")
        print()


if __name__ == "__main__":
    main()
