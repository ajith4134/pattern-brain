# MODEL REPORT — TIER-3 decision/control foundations (`tier3_control`)

**Date:** 2026-06-24 · **Engineer:** The ML Engineer (Pattern Brain)
**Files:** `pattern_brain/nodes/tier3_control.py`, `tests/test_tier3_control.py`, this report.
**Method:** Code-Generation Contract (CLAUDE.md, 5 phases) — oracle/acceptance tests
written BEFORE the implementation, then iterated to green.

These three nodes are the *mathematical ancestors* of the bank's sizing/allocation/exit
machinery (Kelly, Markowitz, optimal-stopping). They are **CONTROL UTILITIES, not 1-step
forecasters** — judged by DESIGN-JOB correctness (does the optimum match the known closed
form / does the sizing reduce variance), exactly as RMT and signatures are judged
(Rule 34). All emit the generic interlingua **`decision`** belief; none invents a new type.
Domain-agnostic (Rule 23): generic `(T, D)` input, no candle/orderbook knowledge.

---

## Node 1 — `log_utility` (LogUtilitySizingNode), layer `decision`

**Computes:** Bernoulli expected/log-utility position sizing — the ancestor of Kelly.
From a window's return estimates `mu, sigma^2`, the growth-optimal fraction is

> `f* = mu / sigma^2`  (maximiser of `E[log(1+f·r)] ≈ f·mu − ½f²σ²`), clipped to `[-1, 1]`.

**Belief:** `decision` — `action` ∈ {long, flat, short} by sign of `f*`, `z` = `f*` (the sizing
fraction), confidence = `|f*|`.

**Oracle result:** node `f*` vs the independent closed-form `mu/var(ddof=1)` oracle —
**max-abs-diff = 0.0** (exact). Sign logic verified on synthetic `mu>0` → long, `mu<0` →
short, exactly-zero-mean → `f*=0`, flat. Clipping verified (low-variance high-mean → ±1).

**Design-job verdict: KEEP (as a sizing utility).** Evidence — walk-forward sizing across
the 10-asset panel (200-bar windows, OOS half):
- sized-P&L vol **below** full-exposure vol (vol cut ≈ 1%, and ≤ full-exposure by construction
  since `|f|≤1`); proxy-Sharpe `mean/std` **flips +0.0165 vs full-exposure −0.0339** — the
  sizing scales down when the edge/variance ratio is weak, the criterion's intended behaviour.
- It is a *sizing* tool, not an alpha source; it earns its place by correctly implementing
  growth-optimal sizing (oracle-exact), not by predicting direction.

---

## Node 2 — `lagrange_opt` (LagrangeMeanVarianceNode), layer `decision`

**Computes:** Lagrangian constrained mean-variance allocation (the math under Markowitz):

> `min wᵀΣw − λ·wᵀμ  s.t.  1ᵀw = 1`.
> Closed form via stationarity `2Σw − λμ − 2γ1 = 0`:
> `a = Σ⁻¹1`, `b = Σ⁻¹μ` (by **`np.linalg.solve`**, never `inv`),
> `γ = (1 − ½λ·1ᵀb) / 1ᵀa`,  `w = ½λ·b + γ·a`.  `λ=0` ⇒ global minimum-variance.

**Belief:** `decision` — `weights` (the allocation), `action` by the expected-portfolio-drift
sign, `z` = portfolio drift. 1-D input degrades to a single asset (`w = [1]`).

**Oracle result:** node weights vs an independent `inv`-based closed-form oracle on a known
SPD `(Σ, μ)` — **max-abs-diff = 5.6e-17** (machine precision); `Σwᵢ = 1` exactly. `λ=0`
matches the global min-variance oracle `Σ⁻¹1/(1ᵀΣ⁻¹1)` to 1e-10.

**Design-job verdict: KEEP (as an allocation utility).** Evidence — train/test split of the
10-asset panel log-returns: the `λ=0` minimum-variance allocation (estimated on TRAIN)
delivers **OOS portfolio variance 7.94e-06 vs 4.07e-05 for equal-weight — an 80.5%
variance reduction**, the textbook minimum-variance guarantee, realised out-of-sample.

---

## Node 3 — `hjb_control` (HJBControlNode), layer `decision`, node_type `hjb_control`

**Computes:** Bellman dynamic-programming / discretized-HJB optimal hold-vs-exit policy.
On a z-score state grid, value iteration solves the discrete HJB optimal-stopping equation

> `V(s) = max( exit_reward,  hold_reward(s) + γ·V(f(s)) )`, transition `f(s) = (1−κ)s`
> (mean-reversion toward 0). The Bellman operator is a γ-contraction → unique fixed point.

**Belief:** `decision` — `action` ∈ {hold, exit} (optimal for the current z-state), `z` = current
z-score, plus `value` and `bellman_residual` for audit. Covers BOTH Bellman (TIER-3) and
HJB (D10).

**Oracle result:** value iteration **converges to Bellman residual = 0.0** (≤ 1e-6 asserted).
Toy reward `−z²` + mean-reversion: center state → `hold`, extreme states → `exit` (the
obviously-optimal structure: stay near equilibrium, stop when the holding penalty dominates;
60/61 grid points exit, only the equilibrium holds).

**Design-job verdict: KEEP (as a control/exit utility).** Evidence — the value-iteration is a
provably-convergent γ-contraction (residual → 0 exact) and recovers the correct
stay-near-equilibrium / stop-at-extremes optimal-stopping policy. It is a *control law*, not a
forecaster; judged on optimum-correctness in a setting where the optimum is known (Rule 34).

---

## Verification (Phase 4 — actual command + output)

```
$ PYTHONPATH=. python3 -m pytest tests/test_tier3_control.py -q
...............                                                          [100%]
15 passed in 0.69s
```

Registration:
```
$ PYTHONPATH=. python3 -c "from pattern_brain.nodes import tier3_control; from pattern_brain.registry import _REGISTRY; print(sorted(k for k in _REGISTRY if k in ('log_utility','lagrange_opt','hjb_control')))"
['hjb_control', 'lagrange_opt', 'log_utility']
```

Interlingua conformance: all three beliefs pass `validate_belief` against the `decision`
schema (test `test_beliefs_conform_to_interlingua_decision_type`).

## Honesty / scope notes
- These are **decision utilities, not alpha** — verdicts are DESIGN-JOB correctness, not OOS
  forecasting skill (the right standard for sizing/allocation/control, per Rule 34).
- The panel is all-crypto (what's on the VPS); the math is asset-agnostic. The min-variance
  and Kelly results are the standard textbook guarantees realised on real OOS data.
- Numerical policy honoured: float64, `np.linalg.solve` for every linear system (no `inv` in
  the implementation; `inv` appears only in the *oracle* as an independent reference), tiny
  ridge guards a near-singular sample covariance, `LinAlgError` falls back to `pinv`.
- File isolation: only the three permitted files were created; no `data/` files were deleted
  or modified; the scratch dir `data/_test_tier3_ctrl/` is created and cleaned up by the test.
