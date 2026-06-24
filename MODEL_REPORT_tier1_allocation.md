# MODEL REPORT — TIER-1 Allocation / Control nodes

Oracle-test-first build (CLAUDE.md 5-phase contract). 4 ALLOCATION/CONTROL nodes,
light-stack only (numpy/scipy/sklearn). Allocators/sizers/controllers are judged by
**correctness + risk reduction (KEEP-as-utility)** — like RMT min-var — not standalone Sharpe.

Test command (only this file):
`PYTHONPATH=. python3 -m pytest tests/test_tier1_allocation.py -q` → **20 passed**.

## Verdict table

| node_type | layer | belief | verdict | headline evidence |
|---|---|---|---|---|
| `hrp_alloc` | decision | decision{weights} | **KEEP-as-utility** | OOS portfolio variance on universe (60 assets, walk-forward): **HRP 2.12e-5 vs equal-weight 6.27e-5 vs inverse-var 4.50e-5** — HRP cuts risk to ~1/3 of equal-weight and beats inverse-variance. Multi-regime daily span: HRP 9.20e-4 vs equal 1.08e-3. No inversion ⇒ stays finite & non-degenerate under a rank-deficient (duplicated-asset) covariance. |
| `cvar_opt` | decision | decision{weights} | **KEEP-as-utility** | OOS tail-loss (mean of worst-5% portfolio returns) on universe: **CVaR −6.25e-3 vs equal-weight −1.74e-2** — ~64% smaller tail loss. Oracle: with a crash-prone fat-tailed asset it under-weights it vs min-variance; recovers ~equal weights on i.i.d. symmetric assets. |
| `merton_alloc` | decision | decision{fraction,fractions,weights,mu,sigma,gamma} | **KEEP-as-utility** | Closed-form recovery exact to 1e-12: returns (μ−r)/(γσ²); monotone ↑μ, ↓σ², ↓γ. Sane signed sizing on panel (BTC/ETH/SOL μ<0 over span → negative fractions). |
| `mpc_position` | decision | decision{positions,turnover} | **KEEP-as-utility** | Tracks a mean-reverting signal (corr>0.5) while smoothing it: turnover < naive sign-following; ablation — turnover **monotonically decreases** as cost weight rises (0.1→20). On real BTC returns: MPC turnover 190.6 vs naive 965.8 (~5x less churn). |

## What each node is (Phase-1 spec, condensed)

1. **`hrp_alloc`** — Hierarchical Risk Parity (Lopez de Prado 2016). corr→distance
   `d=sqrt(0.5(1-ρ))` → single-linkage clustering → quasi-diagonalization (leaf order)
   → recursive bisection allocating risk between halves inversely to cluster variance.
   Long-only, sums to 1, **no matrix inversion** (only diagonal/inverse-variance terms).
2. **`cvar_opt`** — Rockafellar-Uryasev (1999) CVaR LP via `scipy.optimize.linprog`
   (HiGHS). Vars [w, VaR, u_t]; minimizes `VaR + 1/((1-α)T)Σu_t` s.t. tail-excess
   constraints, simplex w≥0 Σw=1. Optimum objective IS the minimized CVaR_α.
3. **`merton_alloc`** — Merton (1969) `w*=(μ−r)/(γσ²)` per asset from sample μ,σ.
   Reports scalar `fraction` (D==1) + per-asset `fractions` + normalized long-only tilt.
4. **`mpc_position`** — MPC/LQG QP: minimize `Σ[-s_t·p_t + risk·p_t²] + Σ cost·(Δp_t)²`.
   ∇=0 ⇒ tridiagonal `H p = s` solved with `np.linalg.solve` (no inv). Larger cost ⇒
   smoother path (less turnover).

## Oracle coverage (true-positive + control, written first)

- HRP: simplex+positive; diversifies across blocks (15%<each<60%); no blow-up on
  near-singular cov (adversarial control); inverse-variance within an uncorrelated pair.
- CVaR: under-weights fat-tailed asset vs mean-variance (true-pos); recovers equal
  weights on i.i.d. symmetric (control).
- Merton: closed-form exact to 1e-12 (true-pos); monotone in μ/σ²/γ (control);
  node recovers its own estimated fraction.
- MPC: tracks-but-smooths (true-pos); turnover monotone-decreasing in cost (control/ablation).
- Interlingua: all 4 beliefs pass `validate_belief == []`; weights sum to 1.

## Honest notes / limitations

- Verdicts are **KEEP-as-utility**, not standalone-edge KEEP: these are risk/sizing
  tools. The strong universe numbers (HRP variance, CVaR tail) are genuine OOS risk
  reductions, but none claims a directional return edge.
- `merton_alloc` and `mpc_position` are judged by **correctness/turnover**, not OOS
  PnL — appropriate for sizing/control utilities. Merton fractions are unbounded by
  design (raw `(μ−r)/(γσ²)`); the consumer is responsible for leverage capping.
- CVaR solves an LP per call (T+D+1 vars). Bounded fine for the universe T~999, but
  it is the heaviest of the four (cost="med").
- MPC tail-truncates signals >2000 points to bound the dense T×T solve.
