# MODEL_REPORT — TIER-1 DUAL-DOMAIN verification (Rule 36 clause 5)

**Date:** 2026-06-24 · **Harness:** `tools/eval_crossdomain_tier1.py`
**Rule:** every GENERAL-PURPOSE node must earn a verdict on a **real NON-trading dataset of the
type its theory consumes**, not only on trading data. (Trading-data verdicts: the per-family
MODEL_REPORTs.) Real non-financial datasets (`data/crossdomain/`): **sunspots** (monthly, ~11yr
cycle), **airline** (passengers, trend+seasonal), **temps** (daily min temperature, seasonal
heteroskedastic), **diabetes** (UCI regression 442×10), **digits** (UCI handwriting 1797×64).

## Results (each node on its design-appropriate non-trading input + a baseline/control)
| node | non-trading test | result | dual-domain verdict |
|---|---|---|---|
| `koopman_dmd` | sunspots 1-step (cyclic dynamics — its home turf) | **BEATS** persistence, skill +0.157 | ✅ CONFIRMED: KEEP on dynamical/cyclic, REJECT on returns — the home-turf claim holds on real non-trading dynamics |
| `heston_vol` | temps realized-variance 1-step | **BEATS** persistence (135 vs 246) | ✅ CONFIRMED KEEP: vol forecasting generalizes to non-financial heteroskedasticity |
| `har_rv` | temps realized-variance 1-step (fed the LEVEL; computes RV internally) | **BEATS** persistence (135 vs 246) | ✅ CONFIRMED KEEP (batch 2): generalizes to non-financial RV |
| `sv_particle` | temps: corr(filtered vol, rolling \|dev\|) | **+0.57** | ✅ CONFIRMED KEEP-as-utility: tracks non-financial volatility |
| `infogeom_distance` | sunspots low→high-activity regime splice | spike **6.06** vs shuffled control **0.68** | ✅ CONFIRMED KEEP-as-utility: regime-distance fires on a real non-financial shift |
| `jump_diffusion` | temps daily changes vs Gaussian-matched control | **12** jumps vs **1** | ✅ CONFIRMED KEEP-as-utility: detects real non-financial jumps |
| `percolation_risk` | UCI digits 64-pixel correlation network vs column-shuffled | giant-frac **0.41** vs **0.02** | ✅ CONFIRMED KEEP-as-utility: detects real non-financial correlation structure |
| `diffusion_map` | UCI pendigits (done in build) | kNN **0.80** vs 0.10 chance | ✅ CONFIRMED KEEP-as-utility (cross-domain in the batch report) |
| `hrp_alloc` | UCI digits covariance (correctness) | weights sum 1, ≥0, valid | ✅ correct on a non-financial covariance (note: concentrates, eff-bets 1.0 on this degenerate pixel cov) |
| `bsts_forecast` | log-airline 1-step | **loses** (skill −1.05) | ⚠️ DOWNGRADE → SHADOW as a forecaster (it is a trend/level FILTER; no seasonal term ⇒ loses to persistence on seasonal data, same as on returns) |
| `fuzzy_ts` | airline 1-step | **loses** | ✅ consistent SHADOW: interpretable tracker, no forecast edge in either domain |
| `grey_gm11` | airline 6-pt windows | **loses** (seasonality breaks the smooth-growth assumption) | ⚠️ SCOPE-NARROWED: KEEP-as-utility **only** for genuinely smooth/monotone short series (its exponential oracle); not for seasonal/noisy data |
| `anfis` | UCI diabetes regression vs linear | **loses** (RMSE 174 vs 53) | ⚠️ DOWNGRADE → SHADOW: captures low-dim synthetic nonlinearity (oracle) but does NOT generalize to real 10-feature regression — would have been a false KEEP without this test |

**Exempt from clause 5** (the rule targets general-purpose methods): `ofi`, `vpin`, `kyle_impact`
are trading-INTRINSIC (their input *is* order flow). `merton_alloc`, `mpc_position`, `cvar_opt`
are domain-agnostic decision/control MATH, validated by closed-form oracle (the formula is
identical regardless of domain).

## What the dual-domain rule bought us (its whole point)
- It **confirmed** 8 verdicts on a genuinely independent real domain (koopman/heston/har/sv/
  infogeom/jump/percolation/diffusion) — these aren't trading-data flukes.
- It **caught two over-generous verdicts**: `anfis` does not generalize beyond low-dim synthetic
  nonlinearity (real-regression RMSE 3× worse than linear), and `grey_gm11`'s utility is narrower
  than the build claimed (smooth-monotone only). `bsts_forecast` is confirmed a filter, not a
  forecaster. These are now honestly SHADOW/scope-narrowed.
- It exposed (and fixed) a **harness bug**, not a model bug: `har_rv` consumes a *level* series and
  computes RV internally; feeding it pre-computed RV double-differenced it and produced a spurious
  33× blow-up. Fixed the harness (feed levels); confirmed `har_rv` is unchanged and beats
  persistence on both crypto (10/10) and temps RV. (A momentary ridge "fix" was reverted once the
  real cause — the harness — was found; Rule 14: evidence before root cause.)
