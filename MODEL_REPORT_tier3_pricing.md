# Model Report — Tier-3 Kinematics & Pricing Foundations

**File:** `pattern_brain/nodes/tier3_pricing.py` · **Tests:** `tests/test_tier3_pricing.py`
**Date:** 2026-06-24 · **Author:** ML Engineer (Code-Generation Contract, all 5 phases)
**Scope:** 3 nodes — `momentum_kinematics`, `almgren_exec`, `bs_pricing`. Each is a
classical-physics / quant-finance foundation behind the generic `Node` interface
(Rule 23: a `(T, D)` array in, a `Belief` out — no candle/order-book knowledge inside).

The oracle/acceptance tests were written **before** the implementation (Contract Phase 4
before Phase 3); the closed forms are the spec the code was driven to satisfy.

---

## 1. `momentum_kinematics` — Momentum/Inertia Forecaster (layer `sequence`)

**Computes.** Treats the series (feature 0) as a 1-D *position* sampled at unit time
step (Newton/Leibniz kinematics): velocity `v = x_t − x_{t−1}`, acceleration
`a = x_t − 2x_{t−1} + x_{t−2}`, then the constant-acceleration extrapolation
`x_{t+1} = x_t + v + a` (= `3x_t − 3x_{t−1} + x_{t−2}`, the exact forward extrapolation
of a quadratic from 3 points — the backward velocity lags by ½a, so the half-step
`v + ½a` plus the remaining ½a gives the full-step discrete recovery). Falls back to
persistence for `T < 3`. **Belief type: `forecast`** (`next_vector`, `velocity`,
`acceleration`).

**Oracle (Phase 4, exact).** On a constant-acceleration series
`x_t = 3 + 1.5·t + 0.5·0.4·t²`, the kinematic step recovers the next value **exactly**:
Node forecast error = **0.00e+00** (true 813.0, predicted 813.0). Persistence fallback
on `T<3` verified.

**Real-data verdict: REJECT (as a forecaster).** Walk-forward one-step vs persistence +
permutation null (`harness.benchmark`, seed 0) on the 10-series panel:

| target | beats persistence | typical skill | typical p | dir-acc |
|---|---|---|---|---|
| `close` (level)   | **0/10** | ≈ −5.2 | 0.003 | ≈ 0.49 |
| `logreturn`       | **0/10** | ≈ −9.0 | ≈ 0.5–0.9 | ≈ 0.38 |

The constant-acceleration assumption adds momentum noise on near-random-walk crypto:
strongly negative skill everywhere, never beats the persistence baseline. (The negative
`close` p-values mean it is *significantly worse* than chance-aligned, not better — a
permutation that breaks its pred↔actual link does *better*.) This matches the project's
standing lesson (exotic/structured forecasters lose to the simple baseline). It still
emits a conforming, projectable `forecast` belief and may carry value as a *velocity/
acceleration regime feature* downstream, but it does **not** earn a forecasting slot.

---

## 2. `almgren_exec` — Almgren-Chriss Optimal Executor (layer `decision`)

**Computes.** The Almgren-Chriss optimal-liquidation schedule (the Pontryagin-maximum /
discrete-Euler-Lagrange solution of the impact+risk cost functional). Optimal holdings
`x_k = X·sinh(κ(N−k))/sinh(κN)`, `k = 0..N`, with urgency
`κ = (1/τ)·arccosh(1 + ½·λσ²τ²/η)`. The node estimates σ from the window's one-step
changes and outputs the **next-step optimal trade fraction** `(x_0 − x_1)/X`.
**Belief type: `decision`** (`action`, `execution_fraction`, `kappa`, `horizon`).

**Oracle (Phase 4).** Judged by design-job correctness, not forecasting:
- Holdings match the sinh closed form to **< 1e-9** for all k; schedule starts at `X`,
  liquidates to `0`, strictly decreasing. ✓
- As **κ → 0** it reduces to the linear/**TWAP** schedule (`max diff < 1e-3`) and the
  next-step fraction → `1/N` (measured **0.1250 = 1/8**). ✓
- Node emits a `decision` with `execution_fraction ∈ (0, 1]` on real data
  (BTCUSDT: fraction 0.2675, κ 0.3095). ✓

**Verdict: KEEP-as-utility.** This is an optimal-control decision utility, correctly
implemented and reducing to TWAP in the risk-neutral limit. It is not a series
forecaster and is not run through the persistence harness (that would be a category
error). Earns its place as a control/decision node.

---

## 3. `bs_pricing` — Black-Scholes Fair-Value-Vol (layer `signal`)

**Computes.** The Black-Scholes-Merton European-option closed form
`C = S·N(d1) − K·e^{−rT}·N(d2)` (+ put, + first-order Greeks), with explicit limits at
`T→0` / `σ→0` (discounted intrinsic value). As a **Node / transformer** it turns the
window into a generic finite signal series: per-feature **rolling realized volatility**
("fair-value vol"), plus an at-the-money BS reference price and Greeks computed from it.
**Belief type: `signal`** (`series`, `mean_abs_change`, `fair_value_vol`,
`atm_call_price`, `delta`, `vega`).

**Oracle (Phase 4).** A correctness-validated *pricing identity*, not a predictor:
- **Put-call parity** `C − P = S − K·e^{−rT}` holds to **7.11e-15** (< 1e-8). ✓
- **Greek signs:** call delta ∈ [0,1], put delta ∈ [−1,0], vega > 0, gamma > 0;
  finite-difference delta matches the closed form (**0.5609 ≈ 0.5609**). ✓
- **Monotonic in vol:** call price strictly increasing over σ ∈ [0.05, 0.9]. ✓
- Node emits a **finite** generic `(T, D)` signal series (Rule 23). ✓

**Verdict: KEEP-as-utility (NOT a baseline-beating predictor).** Black-Scholes is a
pricing identity; it is honest to certify it by parity/Greeks/monotonicity, not by an
OOS skill score against persistence. The Node form yields a valid, finite, generic
fair-value-vol signal that projects cleanly into the belief space.

---

## Integration & rigor checks
- **Registration:** all 3 register via `@register`; `momentum_kinematics`→`forecast`,
  `almgren_exec`→`decision`, `bs_pricing`→`signal` (reused interlingua types, none invented).
- **Interlingua conformance:** all 3 beliefs conform on synthetic and real data
  (`validate_belief` empty); all project to finite `belief_to_vector` / `belief_to_point`.
- **Rule 21 (breaks nothing):** runs in the real load→predict→project flow on BTCUSDT
  without error; finite vectors; consumable downstream.
- **Domain purity (Rule 23):** `tier3_pricing.py` token-scanned clean of
  candle/ohlcv/orderbook identifiers.
- **Tests:** `pytest tests/test_tier3_pricing.py -q` → **12 passed**.

## Summary verdicts
| node_type | layer | belief | verdict |
|---|---|---|---|
| `momentum_kinematics` | sequence | forecast | **REJECT** (0/10 vs persistence; loses on its own home turf — price level too) |
| `almgren_exec` | decision | decision | **KEEP-as-utility** (sinh schedule exact; → TWAP as κ→0) |
| `bs_pricing` | signal | signal | **KEEP-as-utility** (parity 7e-15, Greek signs, vol-monotone) |
