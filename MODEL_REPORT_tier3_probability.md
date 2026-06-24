# Model Report — Tier-3 Probability Foundations

**Module:** `pattern_brain/nodes/tier3_probability.py`
**Tests:** `tests/test_tier3_probability.py`
**Date:** 2026-06-24 · **Engineer:** Claude (senior ML engineer persona, Code-Generation Contract)
**Layer:** `probability` (all three) · **Belief types reused (no new type invented):** `forecast`, `density`

Built per the 5-phase contract (SPECIFY → GROUND → IMPLEMENT → VERIFY → REFINE), oracle-test-first.
Project lesson honored: *assume the simple baseline wins until proven otherwise* — and it did, for the
forecaster.

---

## Oracle test result (Phase 4)

```
PYTHONPATH=. python3 -m pytest tests/test_tier3_probability.py -q
13 passed in 0.66s
```

Key numeric oracles (all PASS):
- **bayes_update** — internal `normal_normal_posterior` matches the textbook precision-weighted
  closed form to **< 1e-10**; with prior N(0,1), σ²=1 it equals `(Σx)/(n+1)` to < 1e-10; posterior
  mean provably lies strictly between prior mean and data mean.
- **maxent_dist** — Gaussian-MaxEnt of N(0,1) data recovers **μ≈0 (|μ|<0.05), σ≈1 (|σ−1|<0.05)**;
  of N(7, 2.5²) recovers **μ≈7, σ≈2.5** (tol 0.1); returned pdf grid integrates (trapezoid) to
  **1.0 ± 1e-2**.
- **chebyshev_bound** — interval half-width is exactly `k·σ`; on a heavy-tailed Student-t(4) sample
  of 100k points, empirical coverage at **k=2,3,4 each ≥ the 1−1/k² guarantee** (distribution-free
  property holds where a Normal band would fail).

Registration confirmed:
```
['bayes_update', 'chebyshev_bound', 'maxent_dist']
```
Belief-conformance to the interlingua: all three emit conforming beliefs (`forecast`/`density`).

---

## Node 1 — `bayes_update` (Bayesian Posterior Updater)

**Computes:** conjugate **Normal-Normal** posterior of the next value's mean. Empirical-Bayes prior
(window mean & variance) updated by the window's observations; posterior mean is the one-step point
forecast, predictive std = √(posterior_var + obs_var). **Belief: `forecast`.**

**Real-data verdict: REJECT as a forecaster (panel).**
Walk-forward 1-step on log-returns of close, 10 panels:

| metric | result |
|---|---|
| MSE-skill vs persistence | +0.49 to +0.56 (looks strong) |
| **permutation p-value** | **0.64 – 0.997 (NOT significant)** |
| beats_baseline (skill>0 ∧ p<0.05 ∧ ci_low>0) | **0/10** |

The apparent +0.5 skill is **hollow**: on (near-uncorrelated) log-returns the persistence baseline
("next return = last return") is terrible, so any shrink-to-mean predictor beats it on raw MSE — but
the permutation null shows the pred↔actual alignment carries **no real information** (p≈0.9). The
posterior mean is ≈ the window mean ≈ 0 on returns (confirmed: posterior_mean = window_mean to 5 dp),
i.e. it is a mean-reverting/shrinkage predictor, not a directional one. This is exactly the project's
documented failure mode (skill without significance). **Verdict: REJECT/SHADOW for forecasting.** It
remains a correct, useful *uncertainty quantifier* (the posterior_std / predictive std are honest), so
its value is as a calibration/conviction feature feeding other nodes, never as a standalone point model.

---

## Node 2 — `maxent_dist` (Maximum-Entropy Distribution)

**Computes:** the least-biased density p(x) ∝ exp(−Σλᵢfᵢ(x)) matching the window's moments. Mean+variance
constraints → closed-form **Gaussian MaxEnt** N(μ,σ²); an optional 4-moment (skew/kurtosis) numeric
refinement (Newton on the log-partition gradient, **log-sum-exp** stabilized, **`np.linalg.solve`** for
the step) with **explicit ill-conditioning flagging** (Jacobian cond > 1e10). **Belief: `density`** (the
required `bic`/`labels`/`max_proba` keys are populated honestly; the real payload is `mu`,`sigma`,`grid`,
`pdf`,`lambdas`,`ill_conditioned`,`jacobian_cond`).

**Verdict: KEEP-as-utility (judged by its design job — distribution recovery, not PnL).**
On all 10 panels (log-returns of close):
- recovered **μ̂ = empirical mean** and **σ̂ = empirical std to < 1e-6** (closed-form correctness),
- returned **pdf integrates to 1.0000** on every panel,
- the 4-moment refinement **correctly FLAGS ill-conditioning** (cond ≈ 1e11–1e14) on these heavy-tailed
  return distributions — it does NOT silently return a wrong high-moment fit; the trustworthy Gaussian-MaxEnt
  closed form is unaffected and is the primary output. This is the designed, honest behavior (flag, don't
  fake), per the brief.

This node is not a forecaster and is not judged by the persistence harness; it does its job (recovers the
right distribution, normalized, with ill-conditioning surfaced) on every real series tested.

---

## Node 3 — `chebyshev_bound` (Chebyshev Tail Bound)

**Computes:** the **distribution-free** interval [μ̂−kσ̂, μ̂+kσ̂] around a point forecast (persistence
anchor), with the guarantee P(|X−μ|≥kσ) ≤ 1/k² ⇒ coverage ≥ 1−1/k². k defaults to 3 (≈88.9%).
**Belief: `forecast`** (carries `interval`, `k`, `coverage_guarantee`).

**Verdict: KEEP-as-utility (calibrated interval; judged by coverage, not point-skill).**
Walk-forward k=3 band coverage on log-returns, 10 panels:

| | coverage (k=3) | guarantee | holds? |
|---|---|---|---|
| all 10 panels | **0.923 – 0.946** | ≥ 0.889 | **10/10 ✅** |

Empirical coverage exceeds the Chebyshev guarantee on every panel (and exceeds it comfortably, because
Chebyshev is a worst-case bound — it is *conservative* on near-symmetric return data, as expected). The
point forecast equals persistence by design (skill ≈ 0 — it is an *interval* model, not a point model;
"beats persistence as a forecaster: 0/10" is the expected, honest result). Its value is the
**distribution-free uncertainty band**: a conformal-style interval that needs no distributional
assumption — useful as a risk/abstention gate for other nodes.

---

## Adversarial / robustness note (§F)

- `chebyshev_bound` was stressed on a **heavy-tailed Student-t(4)** regime in the oracle test (where a
  Gaussian band under-covers) — coverage still met the guarantee at k=2,3,4. Distribution-free by design.
- `maxent_dist`'s numeric high-moment solver hits genuine **ill-conditioning** on heavy-tailed data and
  **flags it** rather than diverging silently (cond surfaced in the payload). The closed-form path is the
  stable fallback and is always returned.

## Honest summary

| node | belief | design job | verdict |
|---|---|---|---|
| `bayes_update` | forecast | 1-step point forecast | **REJECT** as forecaster (skill not significant, p≈0.9, 0/10); keep only as an uncertainty feature |
| `maxent_dist` | density | recover the distribution | **KEEP-as-utility** — exact moment recovery, pdf mass 1.0, ill-conditioning flagged (10/10) |
| `chebyshev_bound` | forecast | calibrated distribution-free interval | **KEEP-as-utility** — coverage ≥ guarantee on 10/10 panels, robust on heavy tails |

None of the three earns a forecasting bank slot on point-prediction significance (the simple baseline /
permutation null wins, as expected). Two earn their keep as **correct, honest probability utilities**
(distribution estimation, distribution-free intervals); the Bayesian updater is a sound uncertainty
quantifier whose point forecast is not significant on the panel.

---

## File isolation

Touched **only** these 3 files (per the brief): `pattern_brain/nodes/tier3_probability.py`,
`tests/test_tier3_probability.py`, `MODEL_REPORT_tier3_probability.md`. No edits to
`pattern_brain/nodes/__init__.py`, `registry.py`, or any shared doc.
