"""Tests for Phase 7g equation/symbolic-discovery nodes (PLAN.md §11, Block 57).

Run: python3 tests/test_equation_discovery.py  (light stack only — numpy/sklearn).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
NODES = ["sindy_regression", "genetic_symbolic_regression"]


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def test_registered_and_conform():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200)).reshape(-1, 1)
    for n in NODES:
        check(n in all_node_types(), f"{n} not registered")
        b = create(n).predict(x)
        check(not validate_belief(b), f"{n} off-contract: {validate_belief(b)}")
        check("coef" in b.payload and "r2" in b.payload, f"{n} missing coef/r2")
        check(np.isfinite(b.confidence), f"{n} non-finite confidence")
    print(f"  both 7g nodes registered, run, and conform to the 'equation' contract")


def test_sindy_recovers_linear_decay():
    """A series obeying dx/dt = -0.3 x (exponential decay) -> SINDy should put a
    sizable NEGATIVE coefficient on the 'x' term, stay sparse, and fit well; a
    random walk -> low derivative-fit r2."""
    n = 300
    x = np.zeros(n); x[0] = 5.0
    for t in range(1, n):
        x[t] = x[t - 1] + 1.0 * (-0.3 * x[t - 1])          # dx/dt = -0.3 x
    p = create("sindy_regression").predict(x.reshape(-1, 1)).payload
    coef_x = p["coef"][1]                                   # terms = [1, x, x^2, x^3, sin, cos]
    check(coef_x < -0.05, f"SINDy should find a negative 'x' coefficient, got {coef_x:.3f}")
    check(p["n_active"] <= 3, f"SINDy should be sparse on a 1-term law, got {p['n_active']} active")
    check(p["r2"] > 0.7, f"SINDy derivative fit r2 should be high on a clean ODE, got {p['r2']:.2f}")

    noise = np.random.default_rng(1).normal(size=n).cumsum().reshape(-1, 1)
    r2_noise = create("sindy_regression").predict(noise).payload["r2"]
    check(p["r2"] > r2_noise, f"clean ODE r2 ({p['r2']:.2f}) should exceed random-walk ({r2_noise:.2f})")
    print(f"  sindy: recovered dx/dt≈{coef_x:.2f}*x (active={p['n_active']}, r2={p['r2']:.2f}) "
          f"> random-walk r2={r2_noise:.2f}")


def test_gp_fits_structure_better_than_noise():
    """GP symbolic regression should explain a structured (quadratic) trend far
    better than pure noise, and return a closed form that uses the time variable."""
    T = 200
    t_idx = np.linspace(0, 1, T)
    structured = (5.0 * t_idx ** 2 - 2.0 * t_idx).reshape(-1, 1)
    noise = np.random.default_rng(3).normal(size=T).reshape(-1, 1)
    ps = create("genetic_symbolic_regression").predict(structured).payload
    pn = create("genetic_symbolic_regression").predict(noise).payload
    check(ps["r2"] > 0.5, f"GP should explain a quadratic trend (r2>0.5), got {ps['r2']:.2f}")
    check(ps["r2"] > pn["r2"], f"structured r2 ({ps['r2']:.2f}) should beat noise ({pn['r2']:.2f})")
    check(len(ps["coef"]) == 2, f"GP coef should be [a, b], got {ps['coef']}")
    check("t" in ps["tree"], f"discovered expression should use t: {ps['tree']}")
    check(0.0 <= ps["r2"] <= 1.0, "r2 must be normalized")
    print(f"  gp: structured r2={ps['r2']:.2f} (tree={ps['tree']}) > noise r2={pn['r2']:.2f}")


def test_gp_deterministic():
    """Same seed -> same discovered expression (reproducible science)."""
    x = np.cumsum(np.random.default_rng(7).normal(size=150)).reshape(-1, 1)
    a = create("genetic_symbolic_regression").predict(x).payload["tree"]
    b = create("genetic_symbolic_regression").predict(x).payload["tree"]
    check(a == b, f"GP not deterministic for a fixed seed: {a!r} vs {b!r}")
    print(f"  gp: deterministic under a fixed seed (tree={a})")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "equation.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in equation.py: {hits}")
    print("  domain-independence: equation.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 7g equation/symbolic-discovery nodes: tests")
    print("=" * 70)
    test_registered_and_conform()
    test_sindy_recovers_linear_decay()
    test_gp_fits_structure_better_than_noise()
    test_gp_deterministic()
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
