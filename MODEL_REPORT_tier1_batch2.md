# MODEL_REPORT — TIER-1 · batch 2 (`johansen_coint`, `har_rv`, `copula_dependence`, `bocpd_break`)

**Date:** 2026-06-24 · **Files:** `pattern_brain/nodes/tier1_classics.py` · `tests/test_tier1_classics.py`
Built under the Code-Generation Contract (oracle-test-first, light numpy/scipy stack — no
statsmodels; the ADF/cointegration math is implemented and oracle-checked here). 18/18 tier-1
oracles pass; full bank still conforms (46 passed in the regression subset).

Verdicts: **`har_rv` = KEEP (genuine forecaster, beats baseline).** `johansen_coint`,
`copula_dependence`, `bocpd_break` = **KEEP-as-utility** (correctness-validated risk/regime/
stat tools, same category as EVT/RMT/conformal — judged by correctness, not point-PnL).

---

## 1) `johansen_coint` — cointegration / stat-arb spread  ·  KEEP-as-utility
**Computes:** Engle-Granger two-step on a (T, D≥2) array — OLS hedge ratio of col 0 on the
rest, then an **Augmented Dickey-Fuller** test (own implementation, with constant) on the
residual spread; reports hedge ratio, ADF stat, cointegration verdict, and mean-reversion
half-life. Emits a z-scored spread as a `signal`. D==1 ⇒ tests the level itself.
**Oracle (written first) — PASS:** ADF gives `t>-2.86` on a random walk (unit root) and
`t<-5` on AR(0.2) (mean-reverting); Engle-Granger recovers a cointegrated pair (hedge ratio
2.0±0.2, residual stationary) and rejects two independent random walks.
**Real data (design-appropriate, Rule 34 — `data/meanrev/` spreads + reconstructed panel pairs):**
| input | adf | verdict | half-life |
|---|---|---|---|
| 5 curated mean-reversion spreads | −1.7 … −3.0 | 1/5 clear the strict bar | 35–104 |
| 5 real crypto pairs (ETH/BTC, …) | −1.7 … −3.2 | 0/5 cointegrated | 34–105 |

**Honest reading:** the node is **oracle-correct**; the real result is a true negative — crypto
large-caps are **near-unit-root / only weakly & unstably cointegrated** in this 1000-bar corpus
(half-lives of 35–100+ bars; a well-documented property, and consistent with our OU/EVT
findings). It correctly ranks LTC/BTC as the most stationary spread. As a utility it still
yields a useful hedge ratio + half-life + z-spread feature; it just (correctly) fires few live
cointegration signals here. Not promoted as a standalone signal until tested on a corpus with
genuinely cointegrated assets (Rule 36 breadth).

## 2) `har_rv` — Corsi (2009) Heterogeneous-AR realized-volatility forecaster  ·  **KEEP**
**Computes:** RVₜ₊₁ = c + β_d·RVₜ + β_w·RV̄⁽⁵⁾ + β_m·RV̄⁽²²⁾ (daily/weekly/monthly realized
variance), fit by OLS on a train split. Column 0's squared first-difference is the RV proxy.
Emits a `forecast` of next-step RV (and √RV vol).
**Oracle — PASS:** on GARCH(1,1)-simulated vol-clustering data HAR's out-of-sample MSE beats
the persistence baseline RVₜ₊₁=RVₜ.
**Real data (Rule 32 panel, 10 crypto symbols, walk-forward 60/40):**
| | beats persistence | skill (1−MSE/MSE_naive) |
|---|---|---|
| panel result | **10 / 10 (100%)** | **+0.20 … +0.48** (median ≈ +0.46) |

Point-wise Wilcoxon on squared-error deltas is significant on a few symbols (ADA p=.037,
TRX p=.0006) and noisy on the spiky ones, but the **direction is unanimous (10/10, all positive
skill, 20–48% MSE reduction)** — strong, consistent evidence well past the Rule-32 ≥50% bar.
**Reading:** the first TIER-1 node that genuinely **beats a baseline**, because it forecasts
**volatility (predictable)** rather than **returns (not)** — exactly the project lesson. KEEP.

## 3) `copula_dependence` — dependence structure & tail co-movement  ·  KEEP-as-utility
**Computes:** Kendall's τ, Spearman ρ, Gaussian-copula ρ=sin(πτ/2), and **empirical upper/lower
tail dependence** (P of joint extremes — the joint-crash risk a correlation misses). Emits a
per-point co-extremeness `signal`.
**Oracle — PASS:** a t-copula(ν=3) sample shows materially higher upper-tail dependence than a
Gaussian copula at the same ρ (Gaussian copula has asymptotic tail dependence 0).
**Real data:** ETH/BTC τ=**0.71**, tail-dep ≈ **0.72–0.76**; SOL/ETH τ=0.70, tail-dep 0.68;
independent control ≈ **0.06–0.08** (≈ the q=0.95 floor). Correct, and a genuine risk insight:
crypto large-caps crash **together** (high tail dependence), so a Gaussian-correlation view
understates joint-drawdown risk.

## 4) `bocpd_break` — Bayesian Online Change-Point Detection  ·  KEEP-as-utility
**Computes:** Adams-MacKay (2007) BOCPD, Gaussian (known-variance, Normal-prior-on-mean)
observation model + constant hazard. **Key correctness note:** P(run length=0) is identically
the hazard after normalisation (a known property) — useless as a detector; the real signal is
the **collapse of the run-length posterior**, so the node reports the **fractional drop in the
expected run length E[rₜ]** (∈[0,1]) as the per-step change-point score, emitted as an `anomaly`.
**Oracle — PASS:** a clean mean shift at t=100 is localized to within ±8.
**Real data:** spliced two genuinely different real return regimes (BTC bars then 3× DOGE bars)
with the break at t=300 → detector peak at **t=305 (±5)**, max score 0.98. Correctly localizes
a real structural break. KEEP-as-utility (regime tool).

---
### Build note (Rule 14/31)
The only mid-build wall was BOCPD: the textbook "changepoint probability" P(rₜ=0) came out flat
at exactly the hazard rate. Rather than call it broken, traced it to the BOCPD normalisation
identity and switched the detector to the run-length-posterior collapse (expected-run-length
drop) — the standard practical BOCPD detector — which localizes correctly. Logged for reuse.
