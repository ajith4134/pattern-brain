"""Oracle/acceptance tests for the Tier-3 kinematics & pricing-foundation nodes.

Written FIRST (Code-Generation Contract, Phase 4 before Phase 3): the closed-form
oracles are the spec these three nodes must satisfy.

* ``momentum_kinematics`` — must recover the EXACT next value on a constant-
  acceleration synthetic series (x_t = x0 + v0*t + 0.5*a0*t^2), since its one-step
  constant-acceleration kinematic step x_{t+1}=x_t+v+0.5a is exact there.
* ``almgren_exec`` — its liquidation holdings schedule must match the Almgren-Chriss
  sinh closed form x_k = X*sinh(kappa(N-k))/sinh(kappa N), and reduce to the linear
  (TWAP) schedule as the risk-aversion kappa -> 0.
* ``bs_pricing`` — the Black-Scholes utility must satisfy put-call parity
  (C - P = S - K e^{-rT}) to <1e-8, correct Greek signs (call delta in [0,1],
  vega > 0), and be monotonically increasing in volatility.

Domain-agnostic (Rule 23): the Nodes consume a generic (T, D) array; the closed-form
helpers are tested directly as pricing/scheduling utilities.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief
from pattern_brain.nodes import tier3_pricing as t3

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


# ----------------------------------------------------------------- registration
def test_registered_and_conform():
    for nt in ("momentum_kinematics", "almgren_exec", "bs_pricing"):
        check(nt in all_node_types(), f"{nt} not registered")
    x = np.cumsum(np.random.default_rng(0).normal(size=(200, 1)), axis=0) + 100.0
    types = {"momentum_kinematics": "forecast", "almgren_exec": "decision",
             "bs_pricing": "signal"}
    for nt, want in types.items():
        b = create(nt).predict(x)
        check(b.type == want, f"{nt} emitted {b.type!r}, want {want!r}")
        check(not validate_belief(b), f"{nt} off-contract: {validate_belief(b)}")
    print(f"  registration: 3 nodes registered, correct belief types, conform")


# ------------------------------------------------ momentum_kinematics oracle
def test_momentum_recovers_constant_acceleration():
    # x_t = x0 + v0*t + 0.5*a0*t^2  ->  the constant-acceleration step is EXACT.
    x0, v0, a0 = 3.0, 1.5, 0.4
    t = np.arange(60, dtype=float)
    x = x0 + v0 * t + 0.5 * a0 * t * t
    # direct helper: predict x[k+1] from history x[:k+1]
    for k in (3, 10, 40, 58):
        pred = t3.kinematic_next_step(x[:k + 1])
        check(abs(pred - x[k + 1]) < 1e-7,
              f"kinematic step off at k={k}: pred {pred} vs true {x[k + 1]}")
    # through the Node
    b = create("momentum_kinematics").predict(x.reshape(-1, 1))
    pred_node = b.payload["next_vector"][0]
    true_next = x0 + v0 * len(x) + 0.5 * a0 * len(x) ** 2
    check(abs(pred_node - true_next) < 1e-6,
          f"node forecast off: {pred_node} vs {true_next}")
    print(f"  momentum: exact next on constant-accel series (Node err "
          f"{abs(pred_node - true_next):.2e})")


def test_momentum_short_series_falls_back_to_persistence():
    for x in (np.array([[5.0]]), np.array([[5.0], [7.0]])):
        b = create("momentum_kinematics").predict(x)
        check(abs(b.payload["next_vector"][0] - x[-1, 0]) < 1e-12,
              "short series must fall back to persistence (last value)")
    print("  momentum: T<3 falls back to persistence cleanly")


# --------------------------------------------------- almgren_exec oracle
def test_almgren_matches_sinh_closed_form():
    X, N, kappa = 1000.0, 10, 0.6
    sched = t3.almgren_holdings(X, N, kappa)
    check(len(sched) == N + 1, f"schedule should have N+1={N+1} points, got {len(sched)}")
    # closed form x_k = X * sinh(kappa(N-k)) / sinh(kappa N)
    for k in range(N + 1):
        ref = X * np.sinh(kappa * (N - k)) / np.sinh(kappa * N)
        check(abs(sched[k] - ref) < 1e-9, f"holdings[{k}] {sched[k]} != closed form {ref}")
    check(abs(sched[0] - X) < 1e-9, "schedule must start at full inventory X")
    check(abs(sched[-1]) < 1e-9, "schedule must liquidate to 0 at the horizon")
    # holdings strictly decreasing (a valid sell-down trajectory)
    check(np.all(np.diff(sched) < 0), "holdings must be monotonically decreasing")
    print("  almgren: holdings match the sinh closed form, start=X, end=0, decreasing")


def test_almgren_reduces_to_twap_as_kappa_zero():
    X, N = 1000.0, 8
    sched = t3.almgren_holdings(X, N, 1e-7)            # kappa -> 0
    linear = X * (1.0 - np.arange(N + 1) / N)          # TWAP holdings
    check(np.allclose(sched, linear, atol=1e-3),
          f"kappa->0 must give the linear/TWAP schedule; max diff "
          f"{np.max(np.abs(sched - linear)):.2e}")
    # and the next-step trade fraction -> 1/N (equal slices)
    frac = (sched[0] - sched[1]) / X
    check(abs(frac - 1.0 / N) < 1e-3, f"TWAP next-step fraction should be 1/N={1/N}, got {frac}")
    print(f"  almgren: kappa->0 reduces to TWAP (next fraction {frac:.4f} ~ 1/N={1/N:.4f})")


def test_almgren_node_emits_decision_fraction():
    x = np.cumsum(np.random.default_rng(1).normal(size=(120, 1)), axis=0) + 50.0
    b = create("almgren_exec").predict(x)
    frac = b.payload["execution_fraction"]
    check(0.0 < frac <= 1.0, f"execution fraction must be in (0,1], got {frac}")
    check("action" in b.payload, "decision belief should carry an 'action'")
    print(f"  almgren: Node emits a decision with execution_fraction={frac:.4f}")


# ----------------------------------------------------- bs_pricing oracle
def test_bs_put_call_parity():
    S, K, r, T, sigma = 100.0, 95.0, 0.03, 0.75, 0.25
    c = t3.bs_price(S, K, r, T, sigma, call=True)
    p = t3.bs_price(S, K, r, T, sigma, call=False)
    parity = (c - p) - (S - K * np.exp(-r * T))
    check(abs(parity) < 1e-8, f"put-call parity violated by {parity:.2e}")
    print(f"  bs: put-call parity holds to {abs(parity):.2e}")


def test_bs_greek_signs():
    S, K, r, T, sigma = 100.0, 100.0, 0.02, 0.5, 0.3
    g_call = t3.bs_greeks(S, K, r, T, sigma, call=True)
    g_put = t3.bs_greeks(S, K, r, T, sigma, call=False)
    check(0.0 <= g_call["delta"] <= 1.0, f"call delta out of [0,1]: {g_call['delta']}")
    check(-1.0 <= g_put["delta"] <= 0.0, f"put delta out of [-1,0]: {g_put['delta']}")
    check(g_call["vega"] > 0, f"vega must be positive: {g_call['vega']}")
    check(g_call["gamma"] > 0, f"gamma must be positive: {g_call['gamma']}")
    # delta from finite difference matches the closed form
    h = 1e-4
    fd = (t3.bs_price(S + h, K, r, T, sigma) - t3.bs_price(S - h, K, r, T, sigma)) / (2 * h)
    check(abs(fd - g_call["delta"]) < 1e-4, f"delta FD {fd} != closed form {g_call['delta']}")
    print(f"  bs: greek signs correct; delta FD={fd:.4f} ~ closed={g_call['delta']:.4f}")


def test_bs_monotonic_in_vol():
    S, K, r, T = 100.0, 100.0, 0.02, 0.5
    vols = np.linspace(0.05, 0.9, 30)
    prices = np.array([t3.bs_price(S, K, r, T, v) for v in vols])
    check(np.all(np.diff(prices) > 0), "BS call price must be strictly increasing in vol")
    print(f"  bs: call price strictly increasing in vol over [{vols[0]},{vols[-1]}]")


def test_bs_node_emits_finite_signal_series():
    x = np.abs(np.cumsum(np.random.default_rng(2).normal(size=(200, 1)), axis=0)) + 100.0
    b = create("bs_pricing").predict(x)
    series = np.asarray(b.payload["series"], dtype=float)
    check(series.ndim == 2 and series.shape[0] == x.shape[0],
          f"signal series must be (T,D'); got {series.shape}")
    check(np.all(np.isfinite(series)), "signal series must be finite (Rule 23)")
    print(f"  bs: Node emits a finite generic signal series {series.shape}")


# ---------------------------------------------------------------- domain purity
def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "tier3_pricing.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in tier3_pricing.py: {hits}")
    print("  domain-independence: tier3_pricing.py clean")


def test_zzz_no_accumulated_failures():
    """Defined LAST so pytest runs it after every other test: assert the shared
    FAILS accumulator is empty so pytest actually fails if any check() failed."""
    assert not FAILS, "accumulated check failures:\n" + "\n".join("  - " + f for f in FAILS)


def main():
    print("=" * 70)
    print("Pattern Brain — Tier-3 kinematics & pricing nodes: oracle tests")
    print("=" * 70)
    test_registered_and_conform()
    test_momentum_recovers_constant_acceleration()
    test_momentum_short_series_falls_back_to_persistence()
    test_almgren_matches_sinh_closed_form()
    test_almgren_reduces_to_twap_as_kappa_zero()
    test_almgren_node_emits_decision_fraction()
    test_bs_put_call_parity()
    test_bs_greek_signs()
    test_bs_monotonic_in_vol()
    test_bs_node_emits_finite_signal_series()
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
