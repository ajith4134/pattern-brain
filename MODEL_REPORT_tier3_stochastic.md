# Tier-3 Stochastic-Process Foundations — Model Report

**Date:** 2026-06-24 · **Author:** Claude (senior ML engineer, Rule 30)
**Module:** `pattern_brain/nodes/tier3_stochastic.py` · **Tests:** `tests/test_tier3_stochastic.py`
**Contract:** CLAUDE.md 5-phase (SPECIFY→GROUND→IMPLEMENT→VERIFY→REFINE), oracle-first.
**Belief types reused (no new types invented):** `forecast`, `density` (interlingua v0.1).

Three canonical SDE-engine nodes — the *honest baselines and density evolvers* that
the bank's exotic models must beat. Assume the simple baseline wins until proven
otherwise (Rule 30 / our own history). It did.

---

## 1. The three nodes

| node_type | layer | belief | what it computes |
|---|---|---|---|
| `gbm_baseline` | `sequence` | `forecast` | Geometric/Bachelier BM: μ,σ from log-returns; one-step level forecast = lognormal mean `x_t·exp(μ+σ²/2)` + σ band. THE random-walk-with-drift baseline. |
| `langevin_sampler` | `sequence` | `forecast` | Langevin/OU `dX=−γ(X−μ)dt+σdW`; fit γ,μ,σ via AR(1) `X_{t+1}=a+bX_t` (b=e^{−γ}); one-step conditional mean = `a+bX_t`. |
| `fokker_planck` | `probability` | `density` | Fokker-Planck forward equation: evolves the next-value DENSITY one step from the fitted OU/Bachelier generator → exact Gaussian transition `N(m,v)`, discretized on a ±4σ grid that integrates to 1. |

### Honest overlap (stated, not hidden)
`langevin_sampler` is the **SDE-engine framing of the existing `ou_mean_reversion`
node** — the discrete Langevin one-step conditional mean `E[X_{t+1}|X_t]=a+bX_t` is
*identical* to AR(1) and to `ou_mean_reversion`. The oracle test
`test_langevin_conditional_mean_equals_ar1_identity` pins the equality to **<1e-8**,
and `test_langevin_matches_ou_node` confirms the two nodes produce the same point
forecast to <1e-8. The value `langevin_sampler` adds over `ou_mean_reversion` is the
**explicit SDE parameterisation** (γ, σ in continuous-time units, half-life) and its
role as the generator that `fokker_planck` evolves — not a different point forecast.

---

## 2. Oracle results (Phase-4 verify gate — written BEFORE the code)

`PYTHONPATH=. python3 -m pytest tests/test_tier3_stochastic.py -q` → **12 passed**.

| Oracle check | Ground truth | Result |
|---|---|---|
| GBM drift/vol recovery (8 seeds, n=4000) | μ=0.0005, σ=0.02 | μ̂, σ̂ within tol (`|σ̂−σ|<0.002`, `|μ̂−μ|<0.0015`) ✓ |
| GBM forecast = lognormal mean | `x_t·exp(μ̂+σ̂²/2)` | matches to **<1e-8** ✓ |
| GBM Bachelier fallback (zero-crossing series) | `x_t+mean(Δx)` | matches to <1e-8, flags `Bachelier` ✓ |
| Langevin γ,μ recovery (8 seeds, n=6000) | γ=0.05, μ=5.0 | within tol (`|γ̂−γ|<0.01`, `|μ̂−μ|<0.2`) ✓ |
| **Langevin cond.mean ≡ AR(1)** | `a+bX_t` (independent OLS) | equal to **<1e-8** ✓ |
| Langevin ≡ ou_mean_reversion | same point forecast | equal to <1e-8 ✓ |
| Fokker-Planck density ∫=1 | trapezoid over grid | mass = 1.000 (`<1e-3`) ✓ |
| Fokker-Planck moments = OU transition | `m=μ+(x−μ)e^{−γ}`, `v=σ²/(2γ)(1−e^{−2γ})` | analytic & grid moments match ✓ |
| Fokker-Planck on GBM-like series | normalized, non-negative pdf, valid mode | ✓ |
| domain-independence (Rule 23) | no candle/ohlcv/orderbook tokens | clean ✓ |

The closed-form identities (GBM lognormal mean, Langevin≡AR(1), OU transition
moments) are the reference oracles — these are *provably* correct, not "looks right".

---

## 3. Real-data evaluation (Rule 32 panel + Rule 34 design-appropriate data)

Walk-forward one-step vs **persistence** + permutation null + bootstrap CI
(`pattern_brain/harness.py`, 5 splits, embargo 0.02, 200 perms, 300 boots). KEEP bar
= beat persistence OOS with significance (skill>0, p<0.05, CI-low>0) on ≥50% of datasets.

### 3a. Panel — log-returns (10 crypto series)

| node | beats persistence | honest read |
|---|---|---|
| `gbm_baseline` | **0/10** | skill ≈ −0.002 on every series — it *is* persistence (random-walk-with-drift). Exactly as predicted. |
| `langevin_sampler` | **1/10** (DOGE only, p=0.010) | +0.52 "skill" everywhere is the **AR(1)-on-returns artifact** (returns mean-revert toward ~0, so persistence-of-last-return is a weak baseline); γ̂=0 (diffusive limit). Only 1/10 clears significance → that one is multiple-comparison luck, not edge. |
| `fokker_planck` | **0/10** | the harness reduces a *density* to its mean point; large negative point-skill is expected and IS NOT the right judge (see §3c). |

### 3b. Design-appropriate data — mean-reverting spread LEVELS (5 cointegrated spreads, Rule 34)

This is `langevin_sampler`'s home turf (OU on a genuinely mean-reverting level).

| node | beats persistence | honest read |
|---|---|---|
| `gbm_baseline` | **0/5** | skill ≈ −0.002; ties persistence (as designed). |
| `langevin_sampler` | **0/5** | skill ≈ 0 (−0.014…+0.003). The spreads are **near unit-root** (b≈1), so OU collapses to persistence here too — no marginal edge at the 1-step horizon. |
| `fokker_planck` | **0/5** | point-skill judge inapplicable; density correctness verified separately (integral=1.000000, mean/var match the OU generator on the real spread). |

### 3c. Why fokker_planck is judged by CORRECTNESS, not PnL (per the spec)

The task explicitly says: *"Judge by correctness (the density integrates to 1,
mean/var match the SDE), not PnL."* On a **real ETH/BTC spread** the emitted density
gives `∫pdf=1.000000`, non-negative everywhere, with mean/variance equal to the
closed-form OU transition moments. That is its passing verdict — the harness skill
column (which collapses the distribution to a point) is the wrong instrument for a
density node and is reported only for completeness.

---

## 4. Verdicts (KEEP / SHADOW / REJECT — honest)

| node | verdict | rationale |
|---|---|---|
| `gbm_baseline` | **KEEP-as-utility (baseline)** | Does not beat persistence (it *is* the random-walk-with-drift baseline) — and it was never meant to. Its job is to be the honest yardstick + emit a calibrated σ band. Correct by construction (oracle-proven). Not harmful. |
| `langevin_sampler` | **SHADOW** | Point forecast ≡ AR(1)/`ou_mean_reversion` (proven). No marginal 1-step edge over persistence on returns (artifact-driven 1/10) **or** on its design-appropriate near-unit-root spreads (0/5). Keep in shadow for its SDE diagnostics (γ, half-life, z_dev) and as the Fokker-Planck generator; do not promote on point forecast. |
| `fokker_planck` | **KEEP-as-utility (density)** | Passes its real test — a correct, normalized OU/Bachelier transition density (∫=1, moments match the SDE) on simulated AND real data. Useful as a calibrated next-value distribution, not as a point forecaster. Not judged by PnL (per spec). |

**Bottom line (Rule 10 — forced conclusion):** all three are *correct* (oracle-proven)
and none beats the simple baseline on point forecast — precisely the project's
recurring lesson. `gbm_baseline` and `fokker_planck` earn their place as the
honest baseline and the calibrated density; `langevin_sampler` is an explicit-SDE
re-skin of an existing node and stays shadow.

---

## 5. Adversarial / robustness note (ML_ENGINEERING_PRACTICES §F)

- **Near-unit-root / non-stationary** (the spreads, b≈1): `langevin_sampler` and
  `_ou_params_from_ar1` guard the explosive/unit-root case (clamp b∈[−0.999999,
  0.999999], fall to the diffusive γ→0 limit with `v=σ²dt`) — no blow-ups, finite
  forecasts throughout (0 non-finite over 15 datasets × walk-forward).
- **Zero-crossing series** (returns centered at 0): `gbm_baseline` detects the
  non-positive level and switches to arithmetic Bachelier BM instead of taking
  log(≤0) — oracle-tested.
- **Tiny windows** (<8 obs): all three return a safe persistence/near-delta fallback.
- Numerical stability: AR(1) fit via `np.linalg.lstsq` (QR-based, not normal-equation
  inversion); float64 throughout; densities renormalized over the finite grid.

---

## 6. File isolation (Step 4 compliance)

Created ONLY these 3 files — touched nothing shared (not `nodes/__init__.py`, not the
registry, not other docs). The module imports cleanly standalone
(`from pattern_brain.nodes import tier3_stochastic`); the orchestrator wires it into
`nodes/__init__.py` and consolidates.

- `pattern_brain/nodes/tier3_stochastic.py`
- `tests/test_tier3_stochastic.py`
- `MODEL_REPORT_tier3_stochastic.md`
