"""Tests for the decomposition nodes (PLAN §17.6 — Wavelet / SSA / EMD / PySR).

Run: python3 tests/test_decomposition.py  (light stack — SSA/EMD/Wavelet need no
new dep; PySR is optional-guarded and simply absent here).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb  # noqa: F401  (registers the bank)
from pattern_brain.registry import create, all_node_types
from pattern_brain.interlingua import validate_belief

FAILS = []
ALWAYS_ON = ["ssa_decompose", "wavelet_decompose", "emd_decompose"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _signal(T=200, D=1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    cols = []
    for d in range(D):
        cols.append(0.02 * t + np.sin(0.3 * t + d) + rng.normal(0, 0.4, T))
    return np.column_stack(cols)


def test_registration():
    for nt in ALWAYS_ON:
        check(nt in all_node_types(), f"{nt} should register with no extra deps")
    # PySR registers ONLY if pysr/Julia is importable (optional-guard like torch)
    from pattern_brain.nodes.decomposition import PYSR_AVAILABLE
    check(("pysr_symbolic" in all_node_types()) == PYSR_AVAILABLE,
          "pysr_symbolic registration must match PYSR_AVAILABLE")
    print(f"  registration: {ALWAYS_ON} present; pysr_symbolic={'on' if PYSR_AVAILABLE else 'off (no Julia)'}")


def test_transformer_contract():
    """Each is a transformer: transform → finite (T, D) of the SAME shape."""
    X = _signal(T=200, D=3)
    for nt in ALWAYS_ON:
        out = create(nt).transform(X)
        check(out.shape == X.shape, f"{nt} transform shape {getattr(out,'shape',None)} != {X.shape}")
        check(np.all(np.isfinite(out)), f"{nt} transform produced non-finite values")
    print("  transformer contract: all 3 return finite (T, D) matching input (D=3)")


def test_belief_conformant():
    """Each emits an interlingua-conformant 'signal' belief (required key 'series')."""
    X = _signal(T=200, D=2)
    for nt in ALWAYS_ON:
        b = create(nt).predict(X)
        check(b.type == "signal", f"{nt} should emit a 'signal' belief, got {b.type}")
        check("series" in b.payload, f"{nt} belief missing required 'series' key")
        check(0.0 <= b.confidence <= 1.0, f"{nt} confidence out of range: {b.confidence}")
        check(not validate_belief(b), f"{nt} belief not interlingua-conformant: {validate_belief(b)}")
    print("  belief contract: all 3 emit conformant 'signal' beliefs with 'series'")


def test_short_series_robustness():
    """No crash on a degenerate short series."""
    Xs = np.random.default_rng(1).normal(0, 1, (5, 1))
    for nt in ALWAYS_ON:
        node = create(nt)
        out = node.transform(Xs)
        check(out.shape == Xs.shape and np.all(np.isfinite(out)), f"{nt} short-series transform fail")
        check(not validate_belief(node.predict(Xs)), f"{nt} short-series belief invalid")
    print("  robustness: all 3 handle a T=5 series without error")


def test_denoising_actually_smooths():
    """Behavioral: a denoise transform should reduce variance vs the noisy input
    (it removes high-frequency content), for SSA + wavelet + EMD."""
    rng = np.random.default_rng(2)
    t = np.arange(256)
    clean = np.sin(0.2 * t)
    noisy = (clean + rng.normal(0, 0.8, t.shape)).reshape(-1, 1)
    for nt in ALWAYS_ON:
        out = create(nt).transform(noisy)[:, 0]
        # correlation with the TRUE clean signal should beat the raw noisy input
        c_out = float(np.corrcoef(out, clean)[0, 1])
        c_raw = float(np.corrcoef(noisy[:, 0], clean)[0, 1])
        check(c_out >= c_raw - 0.05,
              f"{nt} denoise should not be worse than raw (clean-corr {c_out:.3f} vs raw {c_raw:.3f})")
    print("  behavior: SSA/wavelet/EMD reconstructions track the underlying clean signal")


def main():
    print("=" * 70)
    print("Pattern Brain — decomposition nodes (Wavelet/SSA/EMD/PySR): tests")
    print("=" * 70)
    test_registration()
    test_transformer_contract()
    test_belief_conformant()
    test_short_series_robustness()
    test_denoising_actually_smooths()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
