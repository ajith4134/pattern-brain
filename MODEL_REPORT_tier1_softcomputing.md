# MODEL REPORT — TIER-1 Soft-Computing Nodes

Oracle-first build (CLAUDE.md Code-Generation Contract: acceptance check written
BEFORE implementation). Three interpretable / data-poor / nonlinear soft-computing
nodes. Light stack only (numpy/scipy). All emit v0.1-conformant beliefs
(`validate_belief == []`). Tested in the regime where each method's theory is meant
to apply (the "should-work" oracle) + a control, then on the DESIGN-INPUT panel data
(`data/panel/*.npz`, close = col 3) against persistence.

**Test command / result:**
`PYTHONPATH=. python3 -m pytest tests/test_tier1_softcomputing.py -q` → **10 passed**.

PROJECT LESSON honored: standalone level/return forecasters usually lose to
persistence on near-random-walk crypto closes — and they do here. These nodes earn
their place as interpretable / data-poor / nonlinear TOOLS, proven on synthetic
oracles, NOT as alpha forecasters. Verdicts are honest.

| node_type | layer | belief | verdict | headline evidence |
|-----------|-------|--------|---------|-------------------|
| `fuzzy_ts` | sequence | forecast | **SHADOW** | ORACLE PASS: tracks a smooth level series corr=0.996, MAE 17.5 (<8% of range); flat-series control collapses to the level (500.00). DESIGN DATA: beats persistence on **0/10** panel symbols → SHADOW (interpretable level tracker, no edge on RW closes). |
| `grey_gm11` | sequence | forecast | **KEEP-as-utility** | ORACLE PASS (data-poor): from only 6 points of `100·e^{0.10t}` recovers −a=0.0999≈0.10, 1-step err 0.24 vs persistence 17.34, conf 0.95; conf collapses to 0.05 on noisy/oscillatory data. DESIGN DATA: 0/10 on the panel close → not alpha, but KEEP-as-utility for genuine smooth/short-horizon extrapolation where its theory holds. |
| `anfis` | equation | forecast | **KEEP (utility) / SHADOW on returns** | ORACLE PASS: on `y=sin(x1)+x2²` test RMSE 0.099 vs ordinary linear 1.169 (>10× lower → captures nonlinearity); reduces to RMSE≈0.000 ≈ linear on a linear target. DESIGN DATA (lagged returns+vol → next return): beats zero-persistence on **0/10** → SHADOW as a return forecaster; KEEP-able as a general nonlinear regressor proven on the oracle. |

## Method notes (Phase-1 specs realized)

**1. `fuzzy_ts` — Chen (1996) Fuzzy Time Series.** Universe of discourse
`[min−d, max+d]` split into k equal-width fuzzy sets (k = √n, clipped 5..30); fuzzify
each point to its set; mine first-order FLRs Aᵢ→Aⱼ grouped by LHS; defuzzify the
successor(s) of the last set (no successor → its own midpoint; one → that midpoint;
many → mean of midpoints). Interpretable, parameter-light.

**2. `grey_gm11` — Grey System GM(1,1) (Deng 1982), DATA-POOR.** Accumulated
Generating Operation `x¹ₖ=Σx⁰`, background mean `zₖ=½(x¹ₖ+x¹ₖ₋₁)`, grey ODE
`x⁰ₖ+a·zₖ=b` stacked and solved by **lstsq** (not normal equations). Time-response
forecast `x̂¹ₖ₊₁=(x⁰₀−b/a)e^{−ak}+b/a`, inverse-AGO to original scale. Non-negativity
handled by a shift when a window dips ≤0. Confidence = `1−20·rel_fit_error`
(low ⇒ poor fit flagged, as the control proves). Works from ~4-6 points.

**3. `anfis` — Sugeno ANFIS (Jang 1993), HYBRID learning.** Gaussian input MFs +
first-order linear consequents. Per epoch: **least-squares** solve for the consequent
parameters given fixed premises (the linear half), then a finite-difference gradient
step on the MF centres/widths (the nonlinear half); widths floored positive.
`requires_y=True` supervised regressor. Numerically stable (lstsq, float64, no `inv`).

## Honest failures / caveats
- **No return-forecasting edge.** None of the three beats persistence on the panel
  (0/10 each). Stated plainly: that is the expected outcome for standalone
  forecasters on near-random-walk closes — not a bug.
- `fuzzy_ts` and `grey_gm11` carry the trend on smooth series but lag at turning
  points (a one-step-behind partition/extrapolation), so they offer no advantage on
  the noisy panel.
- `anfis` finite-difference premise gradient is O(R·D) MF re-evaluations per epoch —
  fine for the small rule counts used (≤6), but it is not a scalable trainer.
