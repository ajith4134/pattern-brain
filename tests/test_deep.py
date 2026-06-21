"""Contract + behavior tests for the deep / PyTorch nodes (build step 6 Phase 6b).

Run with the torch-equipped interpreter:  .venv/bin/python tests/test_deep.py
If torch is NOT installed this suite SKIPS (exit 0) — the deep nodes are an
optional dependency and the light bank must keep working without them (that
property itself is asserted in test_bank.py via the system interpreter).

Encodes the Phase-6b spec per Rule 25: deep nodes wrap behind the SAME generic
Node interface, run on a generic (T, D) sequence, and emit the SAME interlingua
belief types as the light nodes (no new belief type / no catalog change).
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.belief import Belief
from pattern_brain.nodes import deep as deep_mod

FAILS = []
EXPECTED = ["lstm_forecaster", "gru_forecaster", "transformer_forecaster",
            "tcn_forecaster", "deep_autoencoder", "autoencoder_anomaly"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def make_synth(T=160, D=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(0.2 * t)[:, None] + 0.02 * t[:, None]
    return base + rng.normal(0, 0.1, (T, D))


def main():
    print("=" * 70)
    print("Pattern Brain — deep / PyTorch nodes (step 6 Phase 6b): tests")
    print("=" * 70)

    if not deep_mod.TORCH_AVAILABLE:
        print("  torch NOT installed -> deep nodes correctly absent; SKIPPING.")
        check("lstm_forecaster" not in pb.all_node_types(),
              "deep nodes registered without torch (should be absent)")
        if FAILS:
            for f in FAILS:
                print("  - " + f)
            sys.exit(1)
        print("SKIPPED (no torch) — light bank unaffected, as required.")
        return

    print(f"  torch {deep_mod.torch.__version__} present.")
    registered = pb.all_node_types()
    for nt in EXPECTED:
        check(nt in registered, f"deep node {nt} not registered with torch present")

    X = make_synth()
    for nt in EXPECTED:
        node = pb.create(nt)
        try:
            b = node.predict(X)
        except Exception as e:
            check(False, f"{nt}: raised {type(e).__name__}: {e}")
            continue
        check(isinstance(b, Belief), f"{nt}: not a Belief")
        check(0.0 <= b.confidence <= 1.0, f"{nt}: confidence out of range")
        check(b.source == node.name, f"{nt}: bad source")
        # JSON-serializable (shared belief space)
        try:
            json.dumps(b.to_dict())
        except (TypeError, ValueError) as e:
            check(False, f"{nt}: belief not JSON-serializable: {e}")
        # interlingua conformance — deep nodes must speak the existing language
        viols = pb.validate_belief(b)
        check(not viols, f"{nt}: belief violates interlingua: {[v.to_dict() for v in viols]}")
        print(f"    [ok ] {nt:24} -> {b.type:9} conf={b.confidence:.3f}")

    # behavioral: forecasters produce a finite next_vector of width D; transformer
    # autoencoder reconstructs to the same shape.
    fc = pb.create("lstm_forecaster").predict(X)
    nv = np.asarray(fc.payload["next_vector"])
    check(nv.shape[0] == X.shape[1] and np.all(np.isfinite(nv)),
          f"lstm next_vector wrong shape/finite: {nv}")
    recon = pb.create("deep_autoencoder").transform(X)
    check(recon.shape == X.shape, f"autoencoder changed shape: {recon.shape} vs {X.shape}")

    # 1-D input must work (D=1)
    x1 = np.cumsum(np.random.default_rng(1).normal(size=120))
    for nt in ("gru_forecaster", "autoencoder_anomaly"):
        try:
            check(isinstance(pb.create(nt).predict(x1), Belief), f"{nt}: 1-D failed")
        except Exception as e:
            check(False, f"{nt}: 1-D raised {type(e).__name__}: {e}")

    print("\n  behavioral: forecasters + autoencoder shapes ok; 1-D input ok")
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
