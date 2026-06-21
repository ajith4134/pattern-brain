"""Tests for Phase 7c probabilistic/sequence nodes (PLAN.md §11, Block 57). Light stack."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
NODES = ["higher_order_markov", "semi_markov", "hierarchical_hmm", "crf_tagger",
         "bayesian_network", "dynamic_bayesian_network"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _series(seed=4, n=300):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (0.03 * t + 2 * np.sin(2 * np.pi * t / 20) + np.cumsum(rng.normal(scale=0.4, size=n)))


def test_registered_and_conform():
    x = _series().reshape(-1, 1)
    xD = np.column_stack([x[:, 0], np.roll(x[:, 0], 1)])
    for n in NODES:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(xD if n == "dynamic_bayesian_network" else x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
    print(f"  all {len(NODES)} nodes registered + conform")


def test_markov_distributions_valid():
    x = _series(2).reshape(-1, 1)
    for n in ("higher_order_markov", "semi_markov", "dynamic_bayesian_network"):
        d = create(n).predict(np.column_stack([x[:, 0], x[:, 0]]) if n == "dynamic_bayesian_network" else x).payload
        dist = d["distribution"]
        check(abs(sum(dist) - 1.0) < 1e-6, f"{n} distribution not normalized: {sum(dist)}")
        check(d["next_state"] == int(np.argmax(dist)), f"{n} next_state != argmax(dist)")
    print("  markov: higher-order/semi/DBN emit normalized next-state distributions")


def test_hierarchical_hmm_regime():
    x = _series(3).reshape(-1, 1)
    p = create("hierarchical_hmm").predict(x).payload
    check(abs(sum(p["next_distribution"]) - 1.0) < 1e-6, "regime next_distribution not normalized")
    check(len(p["state_posterior"]) == p["n_regimes"], "posterior length != n_regimes")
    check(np.isfinite(p["log_likelihood"]), "non-finite log-likelihood")
    print(f"  hhmm: {p['n_regimes']} coarse regimes, normalized transition, finite loglik")


def test_crf_learns_direction():
    """The CRF should label a strong up-trend mostly 'up' (in-sample structured fit)."""
    from pattern_brain.nodes.probabilistic import _direction_labels
    up = np.cumsum(np.abs(np.random.default_rng(1).normal(size=300)) + 0.1)   # near-monotone up
    p = create("crf_tagger").predict(up.reshape(-1, 1)).payload
    check(p["action"] in (-1, 0, 1), f"crf action out of range: {p['action']}")
    check(all(lab in (0, 1, 2) for lab in p["predictions"]), "crf predicted invalid labels")
    # most true labels on an up-trend are 'up' (2); the CRF's recent predictions should lean up
    check(p["vote"] >= 0, f"crf on an up-trend should not vote 'down', got {p['vote']}")
    print(f"  crf: linear-chain CRF decodes valid labels; up-trend vote={p['vote']} (>=0)")


def test_bayesian_network_structure():
    x = _series(6).reshape(-1, 1)
    p = create("bayesian_network").predict(x).payload
    check(np.isfinite(p["estimate"]), "BN estimate non-finite")
    check(isinstance(p["edges"], dict) and len(p["edges"]) >= 1, "Chow-Liu edges missing")
    check(p["bn_root_parent"] in p["edges"], "chosen parent not among edges")
    print(f"  bayes-net: Chow-Liu picked parent {p['bn_root_parent']} (MI {p['mutual_information']})")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "probabilistic.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in probabilistic.py: {hits}")
    print("  domain-independence: probabilistic.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7c probabilistic/sequence nodes: tests")
    print("=" * 70)
    test_registered_and_conform()
    test_markov_distributions_valid()
    test_hierarchical_hmm_regime()
    test_crf_learns_direction()
    test_bayesian_network_structure()
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
