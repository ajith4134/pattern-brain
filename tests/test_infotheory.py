"""Tests for Phase 7d information-theory nodes (PLAN.md §11, Block 57). Light stack."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
NODES = ["mutual_information", "transfer_entropy", "conditional_entropy",
         "cross_entropy", "kolmogorov_complexity", "mdl_complexity"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_registered_and_conform():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=300))
    xD = np.column_stack([x, np.roll(x, 1)])
    for n in NODES:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(xD if n == "transfer_entropy" else x.reshape(-1, 1))
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check(np.isfinite(b.confidence), f"{n} non-finite conf")
    print(f"  all {len(NODES)} info-theory nodes registered + conform")


def test_mi_higher_for_structured():
    rng = np.random.default_rng(1)
    structured = np.tile([0.0, 1.0, 2.0, 1.0], 80).reshape(-1, 1)         # repeating
    noise = rng.normal(size=320).reshape(-1, 1)
    mi_s = create("mutual_information").predict(structured).payload["mutual_information"]
    mi_n = create("mutual_information").predict(noise).payload["mutual_information"]
    check(mi_s > mi_n, f"MI should be higher for structured ({mi_s:.2f}) vs noise ({mi_n:.2f})")
    print(f"  MI: structured={mi_s:.2f} > noise={mi_n:.2f}")


def test_transfer_entropy_direction():
    """If Y is X's lagged copy, X drives Y -> TE(X->Y) should exceed TE(Y->X)."""
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=400))
    y = np.roll(x, 1) + rng.normal(scale=0.1, size=400)        # y[t] ~ x[t-1]
    p = create("transfer_entropy").predict(np.column_stack([x, y])).payload
    check(p["te_x_to_y"] > p["te_y_to_x"], f"TE direction wrong: x->y {p['te_x_to_y']:.3f} vs y->x {p['te_y_to_x']:.3f}")
    check(p["direction"] == "X->Y", f"expected X->Y, got {p['direction']}")
    # single channel -> 0
    z = create("transfer_entropy").predict(x.reshape(-1, 1)).payload
    check(z["transfer_entropy"] == 0.0, "single-channel TE should be 0")
    print(f"  transfer entropy: X->Y {p['te_x_to_y']:.3f} > Y->X {p['te_y_to_x']:.3f} (direction detected)")


def test_kolmogorov_structured_more_compressible():
    t = np.arange(400)
    sine = np.sin(2 * np.pi * t / 16).reshape(-1, 1)
    noise = np.random.default_rng(2).normal(size=400).reshape(-1, 1)
    r_s = create("kolmogorov_complexity").predict(sine).payload["kolmogorov_complexity"]
    r_n = create("kolmogorov_complexity").predict(noise).payload["kolmogorov_complexity"]
    check(r_s < r_n, f"sine should compress better (lower ratio) than noise: {r_s:.2f} vs {r_n:.2f}")
    print(f"  kolmogorov: sine ratio={r_s:.2f} < noise ratio={r_n:.2f} (structure compresses)")


def test_mdl_order_and_finite():
    x = np.cumsum(np.random.default_rng(3).normal(size=300)).reshape(-1, 1)
    p = create("mdl_complexity").predict(x).payload
    check(p["mdl_order"] in (0, 1, 2), f"mdl order out of range: {p['mdl_order']}")
    check(np.isfinite(p["description_length"]) and p["description_length"] > 0, "bad MDL length")
    ce = create("conditional_entropy").predict(x).payload["conditional_entropy"]
    cx = create("cross_entropy").predict(x).payload["cross_entropy"]
    check(np.isfinite(ce) and np.isfinite(cx), "conditional/cross entropy non-finite")
    print(f"  mdl: selected order {p['mdl_order']}, length {p['description_length']:.0f}; cond/cross entropy finite")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "infotheory.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in infotheory.py: {hits}")
    print("  domain-independence: infotheory.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7d information-theory nodes: tests")
    print("=" * 70)
    test_registered_and_conform()
    test_mi_higher_for_structured()
    test_transfer_entropy_direction()
    test_kolmogorov_structured_more_compressible()
    test_mdl_order_and_finite()
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
