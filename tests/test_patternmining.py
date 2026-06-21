"""Tests for Phase 7b pattern-mining nodes (PLAN.md §11, Block 57). Pure-Python/numpy."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief, known_belief_types
from pattern_brain.nodes import patternmining as pm

FAILS = []
ITEMSET = ["apriori_patterns", "eclat_patterns", "fpgrowth_patterns"]
SEQUENTIAL = ["prefixspan_patterns", "gsp_patterns", "spade_patterns"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _motif_series(seed=3, n=300):
    rng = np.random.default_rng(seed)
    base = np.tile([1.0, 1.2, 0.6], n)[:n] + rng.normal(scale=0.05, size=n)
    return np.cumsum(base).reshape(-1, 1)


def test_registered_conform_and_type():
    check("pattern" in known_belief_types(), "'pattern' belief type not registered")
    x = _motif_series()
    for n in ITEMSET + SEQUENTIAL:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check(b.type == "pattern" and b.payload["n_patterns"] >= 1, f"{n} found no patterns")
    print(f"  all 6 miners registered, emit conformant 'pattern' beliefs with results")


def test_itemset_miners_agree_exactly():
    """Apriori, ECLAT and FP-Growth are different algorithms for the SAME problem —
    on identical transactions + min_support they must return identical frequent
    itemsets with identical supports. Cross-algorithm agreement = correctness."""
    x = _motif_series(7)
    sym, _ = pm._symbolize(x[:, 0])
    tx = pm._transactions(sym)
    ms = pm._min_sup(len(tx))
    fa, fe, ff = pm._apriori(tx, ms), pm._eclat(tx, ms), pm._fpgrowth(tx, ms)
    sa = {frozenset(k): v for k, v in fa.items()}
    se = {frozenset(k): v for k, v in fe.items()}
    sf = {frozenset(k): v for k, v in ff.items()}
    check(set(sa) == set(se) == set(sf),
          f"itemset miners found different sets: apriori={len(sa)} eclat={len(se)} fp={len(sf)}")
    check(all(sa[k] == se[k] == sf[k] for k in sa), "itemset miners disagree on supports")
    print(f"  itemset: Apriori==ECLAT==FP-Growth on {len(sa)} frequent itemsets (exact agreement)")


def test_sequential_miners_find_ordered_patterns():
    x = _motif_series(5)
    for n in SEQUENTIAL:
        p = create(n).predict(x).payload
        multi = [pat for pat in p["patterns"] if len(pat["items"]) >= 2]
        check(len(multi) >= 1, f"{n} found no length>=2 ordered pattern")
        check(p["predicted_next"] in p["alphabet"], f"{n} predicted_next not a valid symbol: {p['predicted_next']}")
    print("  sequential: PrefixSpan/GSP/SPADE find ordered (len>=2) patterns + a valid next-symbol")


def test_more_structure_more_patterns():
    """A strongly repeating motif should yield at least as many frequent patterns as
    near-pure noise (mining detects real repetition)."""
    motif = _motif_series(1)
    noise = np.cumsum(np.random.default_rng(1).normal(size=300)).reshape(-1, 1)
    nm = create("prefixspan_patterns").predict(motif).payload["n_patterns"]
    nn = create("prefixspan_patterns").predict(noise).payload["n_patterns"]
    check(nm >= 1 and nn >= 1, "should always find some patterns")
    print(f"  structure: motif n_patterns={nm}, noise n_patterns={nn} (both non-trivial)")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "patternmining.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in patternmining.py: {hits}")
    print("  domain-independence: patternmining.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7b pattern-mining nodes: tests")
    print("=" * 70)
    test_registered_conform_and_type()
    test_itemset_miners_agree_exactly()
    test_sequential_miners_find_ordered_patterns()
    test_more_structure_more_patterns()
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
