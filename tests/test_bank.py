"""Contract + behavior tests for the starter model bank (build-tracker step 1).

Run: python3 tests/test_bank.py
Exits non-zero if any check fails. Doubles as visible evidence that every node
runs on a generic (T, D) sequence and emits a valid Belief (Rule 25 self-check).
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.belief import Belief

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


def make_synth(T=200, D=3, seed=0):
    """A regime-switching multivariate sequence + per-row regime labels.
    Generic numeric data — no domain meaning."""
    rng = np.random.default_rng(seed)
    half = T // 2
    # regime 0: rising ramp; regime 1: falling + oscillation
    t = np.arange(T)
    base = np.zeros((T, D))
    y = np.zeros(T, dtype=int)
    base[:half] = (0.03 * t[:half])[:, None] + rng.normal(0, 0.3, (half, D))
    y[:half] = 0
    base[half:] = (2.0 - 0.02 * (t[half:] - half))[:, None] \
        + 0.5 * np.sin(0.3 * t[half:])[:, None] + rng.normal(0, 0.3, (T - half, D))
    y[half:] = 1
    return base, y


def test_registry():
    layers = pb.layers()
    expected = {"signal", "noise", "pattern", "sequence",
                "probability", "equation", "decision", "rl"}
    check(set(layers) == expected, f"layers mismatch: got {layers}")
    n = len(pb.all_node_types())
    check(n >= 55, f"expected the step-6-expanded bank (>=55), got {n}")
    print(f"  registry: {n} node types across {len(layers)} layers: {layers}")
    for layer in sorted(layers):
        print(f"    [{layer:11}] {pb.by_layer(layer)}")


def test_contract():
    X, y = make_synth()
    print("\n  per-node beliefs (T=200, D=3):")
    for nt in pb.all_node_types():
        node = pb.create(nt)
        try:
            if node.requires_y:
                node.fit(X, y)
                belief = node.predict(X)
            else:
                belief = node.predict(X)
        except Exception as e:  # a node that throws fails the contract
            check(False, f"{nt}: raised {type(e).__name__}: {e}")
            continue

        ok = True
        ok &= check(isinstance(belief, Belief), f"{nt}: did not return a Belief")
        ok &= check(isinstance(belief.type, str) and belief.type,
                    f"{nt}: empty belief.type")
        ok &= check(0.0 <= belief.confidence <= 1.0,
                    f"{nt}: confidence {belief.confidence} out of [0,1]")
        ok &= check(isinstance(belief.payload, dict), f"{nt}: payload not a dict")
        ok &= check(belief.source == node.name,
                    f"{nt}: source {belief.source!r} != name {node.name!r}")
        # Belief must be JSON-serializable (shared belief space requirement)
        try:
            json.dumps(belief.to_dict())
        except (TypeError, ValueError) as e:
            check(False, f"{nt}: belief not JSON-serializable: {e}")
        # metadata schema
        md = node.metadata()
        for key in ("node_type", "layer", "name", "requires_y", "is_transformer"):
            check(key in md, f"{nt}: metadata missing {key}")
        # transformer nodes must produce a (T, D') array of correct length
        if node.is_transformer:
            out = node.transform(X)
            check(out.shape[0] == X.shape[0],
                  f"{nt}: transform changed T {out.shape} vs {X.shape}")
        flag = "ok " if ok else "FAIL"
        print(f"    [{flag}] {nt:24} -> {belief.type:11} conf={belief.confidence:.3f}")


def test_1d_input():
    """Nodes must accept a 1-D series (coerced to (T, 1))."""
    x = np.cumsum(np.random.default_rng(1).normal(size=120))
    for nt in pb.all_node_types():
        node = pb.create(nt)
        if node.requires_y:
            continue
        try:
            b = node.predict(x)
            check(isinstance(b, Belief), f"{nt}: 1-D input did not yield a Belief")
        except Exception as e:
            check(False, f"{nt}: failed on 1-D input: {type(e).__name__}: {e}")


def test_bad_input():
    """Validation must reject 3-D and non-finite input."""
    node = pb.create("fft")
    for bad, label in [(np.zeros((2, 2, 2)), "3-D"),
                       (np.array([[1.0, np.nan]]), "NaN")]:
        try:
            node.predict(bad)
            check(False, f"validation accepted {label} input")
        except ValueError:
            pass


def test_behaviors():
    """A few real behavioral checks beyond the contract."""
    rng = np.random.default_rng(2)
    # 1) KMeans separates two well-separated blobs
    blob = np.vstack([rng.normal(-5, 0.2, (50, 2)), rng.normal(5, 0.2, (50, 2))])
    b = pb.create("kmeans", n_clusters=2).predict(blob)
    check(b.payload["n_clusters"] == 2, "kmeans did not find 2 clusters on bimodal data")
    check(b.confidence > 0.7, f"kmeans confidence low on clean blobs: {b.confidence}")

    # 2) AR forecast continues a linear ramp roughly in range
    ramp = np.arange(100).reshape(-1, 1).astype(float)
    f = pb.create("autoregressive").predict(ramp).payload["next_vector"][0]
    check(95 <= f <= 110, f"AR forecast off a linear ramp: {f}")

    # 3) Symbolic regression recovers a sinusoid as 'sin'
    t = np.arange(1, 200)
    sine = np.sin(0.2 * t).reshape(-1, 1)
    b = pb.create("symbolic_regression").predict(sine)
    check(b.payload["form"] == "sin",
          f"symbolic_regression picked {b.payload['form']} for a sine")

    # 4) Gaussian HMM recovers ~2 distinct states on a 2-regime sequence
    X, _ = make_synth()
    b = pb.create("gaussian_hmm", n_states=2).predict(X)
    check(0 <= b.payload["current_state"] <= 1, "HMM produced an invalid state index")
    check(b.confidence > 0.5, f"HMM posterior confidence low: {b.confidence}")
    print("\n  behavioral checks: kmeans/AR/symbolic/HMM all asserted")


def test_domain_independence():
    """Rule 23: no CORE node couples to a stock-specific data shape. We inspect
    code *identifiers* only (via tokenize), not docstrings/comments — the prose
    legitimately mentions candles/order books to explain what the code avoids.

    The ``adapters/`` subpackage is excluded: per Rule 23 / PLAN.md §0 it is the
    ONE sanctioned home for domain code (candles live there on purpose). Its own
    separation is enforced separately in test_adapter.py — generic numeric tools
    stay domain-free, and the core never imports the adapter."""
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain")
    hits = []
    for dirpath, dirnames, files in os.walk(root):
        if "adapters" in dirnames:
            dirnames.remove("adapters")  # sanctioned domain home — scanned by test_adapter.py
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with tokenize.open(os.path.join(dirpath, fn)) as fh:
                for tok in tokenize.generate_tokens(fh.readline):
                    if tok.type in (tokenize.NAME, tokenize.STRING):
                        low = tok.string.lower()
                        for bad in forbidden:
                            if bad in low and tok.type == tokenize.NAME:
                                hits.append(f"{fn}:{tok.string}")
    check(not hits, f"domain-coupling identifiers found in core: {hits}")
    print("  domain-independence: no candle/ohlcv/orderbook coupling in code (Rule 23 ok)")


def main():
    print("=" * 70)
    print("Pattern Brain — starter model bank: contract + behavior tests")
    print("=" * 70)
    test_registry()
    test_contract()
    test_1d_input()
    test_bad_input()
    test_behaviors()
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
