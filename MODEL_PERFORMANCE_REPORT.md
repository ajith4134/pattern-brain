# Model Performance Report — Pattern Brain
_Auto-appended by `pattern_brain.reporting` (temp Rule 32). Every invented ML model is tested
across the full data-variety panel; only majority-beaters are kept in the bank. Review at the end._

**Verdict rule:** KEEP if `beats_baseline` (skill>0 AND permutation p<0.05 AND bootstrap CI-low>0)
on **≥ 50% of datasets**; else REJECT/SHADOW. All metrics are out-of-sample walk-forward.


## Persistence (control)  —  **REJECT/SHADOW**
- node_type: `persistence` · category: baseline
- beat baseline on **0/12 datasets (0.0%)** · mean skill 0.0 · median skill 0.0 · mean dir-acc 0.0

| dataset | skill_vs_persist | dir_acc | p_value | CI_low | beats? |
|---|---|---|---|---|---|
| 1000PEPEUSDT_1h | +0.0000 | 0.000 | 0.013 | +0.0000 | ✗ |
| BNBUSDT_1h | +0.0000 | 0.000 | 0.073 | +0.0000 | ✗ |
| BTCUSDT_1d | +0.0000 | 0.000 | 0.927 | +0.0000 | ✗ |
| BTCUSDT_1h | +0.0000 | 0.000 | 0.305 | +0.0000 | ✗ |
| BTCUSDT_5m | +0.0000 | 0.000 | 1.000 | +0.0000 | ✗ |
| DOGEUSDT_1h | +0.0000 | 0.000 | 0.483 | +0.0000 | ✗ |
| ETHUSDT_1h | +0.0000 | 0.000 | 0.013 | +0.0000 | ✗ |
| ETHUSDT_5m | +0.0000 | 0.000 | 0.993 | +0.0000 | ✗ |
| LINKUSDT_1h | +0.0000 | 0.000 | 0.040 | +0.0000 | ✗ |
| SOLUSDT_1h | +0.0000 | 0.000 | 0.106 | +0.0000 | ✗ |
| USDCUSDT_1h | +0.0000 | 0.000 | 0.066 | +0.0000 | ✗ |
| XRPUSDT_1h | +0.0000 | 0.000 | 0.046 | +0.0000 | ✗ |

## AR(1) shrinkage (example)  —  **REJECT/SHADOW**
- node_type: `ar1` · category: D1 statistical
- beat baseline on **2/12 datasets (16.7%)** · mean skill 0.4963 · median skill 0.4913 · mean dir-acc 0.7405

| dataset | skill_vs_persist | dir_acc | p_value | CI_low | beats? |
|---|---|---|---|---|---|
| 1000PEPEUSDT_1h | +0.4856 | 0.753 | 0.781 | +0.4318 | ✗ |
| BNBUSDT_1h | +0.4910 | 0.757 | 0.974 | +0.4188 | ✗ |
| BTCUSDT_1d | +0.5182 | 0.750 | 1.000 | +0.4630 | ✗ |
| BTCUSDT_1h | +0.4966 | 0.750 | 1.000 | +0.4703 | ✗ |
| BTCUSDT_5m | +0.5186 | 0.759 | 0.007 | +0.4782 | ✅ |
| DOGEUSDT_1h | +0.5002 | 0.752 | 1.000 | +0.4358 | ✗ |
| ETHUSDT_1h | +0.4869 | 0.753 | 0.828 | +0.4482 | ✗ |
| ETHUSDT_5m | +0.5173 | 0.751 | 0.245 | +0.4453 | ✗ |
| LINKUSDT_1h | +0.4886 | 0.756 | 1.000 | +0.4384 | ✗ |
| SOLUSDT_1h | +0.4915 | 0.745 | 0.960 | +0.4645 | ✗ |
| USDCUSDT_1h | +0.4702 | 0.617 | 0.046 | +0.2978 | ✅ |
| XRPUSDT_1h | +0.4903 | 0.743 | 1.000 | +0.4507 | ✗ |

## Ornstein-Uhlenbeck Mean-Reverter  —  **REJECT/SHADOW**
- node_type: `ou_mean_reversion` · category: econometrics
- beat baseline on **1/10 datasets (10.0%)** · mean skill 0.5247 · median skill 0.5249 · mean dir-acc 0.7556

| dataset | skill_vs_persist | dir_acc | p_value | CI_low | beats? |
|---|---|---|---|---|---|
| ADAUSDT | +0.5261 | 0.770 | 0.761 | +0.4388 | ✗ |
| BNBUSDT | +0.5163 | 0.741 | 0.601 | +0.4255 | ✗ |
| BTCUSDT | +0.5097 | 0.754 | 0.990 | +0.4138 | ✗ |
| DOGEUSDT | +0.5607 | 0.766 | 0.007 | +0.4684 | ✅ |
| ETHUSDT | +0.5143 | 0.745 | 0.744 | +0.4155 | ✗ |
| LINKUSDT | +0.5359 | 0.764 | 0.595 | +0.4582 | ✗ |
| LTCUSDT | +0.5398 | 0.761 | 0.292 | +0.4344 | ✗ |
| SOLUSDT | +0.5332 | 0.750 | 0.412 | +0.4475 | ✗ |
| TRXUSDT | +0.4876 | 0.750 | 0.814 | +0.3916 | ✗ |
| XRPUSDT | +0.5237 | 0.755 | 0.316 | +0.4377 | ✗ |

#### Rule-34 design-appropriate test — mean-reverting spread LEVELS (OU's home turf)
Real cointegrated log-spreads (data/meanrev/, lag-1 autocorr ~0.98 = genuinely but slowly mean-reverting). OU one-step verified ≡ AR(1) on every spread (skills identical), so this isolates whether OU beats persistence on the data it was BUILT for.

| spread | OU skill_vs_persist | dir_acc | p | beats? | AR(1) skill |
|---|---|---|---|---|---|
| ADA_XRP | +0.0008 | 0.525 | 0.003 | ✗ | +0.0008 |
| BNB_BTC | −0.0145 | 0.522 | 0.003 | ✗ | −0.0145 |
| ETH_BTC | −0.0005 | 0.513 | 0.003 | ✗ | −0.0005 |
| LTC_BTC | +0.0026 | 0.523 | 0.003 | ✗ | +0.0026 |
| SOL_ETH | −0.0080 | 0.500 | 0.003 | ✗ | −0.0080 |

→ OU beats persistence on **0/5** design-appropriate spreads (they are near-unit-root at 1h, so 1-step slow reversion ≈ persistence).

**FINAL VERDICT (Rule 30 + Rule 34): REJECT / SHADOW.** Tested on BOTH the standard returns panel (1/10 beat, the rest fail the permutation null) AND its design-appropriate mean-reverting spreads (0/5 beat). OU's one-step forecast provably equals AR(1) (confirmed numerically), so it offers **no point-forecast edge over the linear baseline** on the available real data. Honest outcome consistent with the project lesson (simple baseline wins). Pipeline value: the full invention loop (build → oracle-verify → dual-regime walk-forward → permutation null → honest reject) ran end-to-end with no promotion on a lucky result. Possible future re-test: faster-reverting series or a multi-step / z_dev-signal evaluation (the diagnostics, not the point forecast).

#### Rule-34 CORRECTION — horizon-correct design-appropriate test (supersedes the 1-step verdict)
The earlier verdict tested OU only at the 1-step horizon, where the OU conditional mean reduces to AR(1) and slow reversion ≈ persistence — an INVALID test for a mean-reverter (owner-flagged; the reason Rule 34/35 exist). Re-tested at the **reversion horizon k ≈ estimated half-life** (`tools/eval_ou_horizon.py`), walk-forward + causal, permutation null:

| spread | k | n | OU skill_vs_persist | p | rev_dir_acc | beats? |
|---|---|---|---|---|---|---|
| ETH_BTC | 40 | 831 | **+0.1358** | 0.000 | 0.597 | ✅ |
| ADA_XRP | 40 | 840 | **+0.0811** | 0.000 | 0.599 | ✅ |
| LTC_BTC | 40 | 840 | **+0.0608** | 0.000 | 0.568 | ✅ |
| BNB_BTC | 40 | 796 | −0.0342 | 0.000 | 0.592 | ✗ |
| SOL_ETH | 27 | 853 | −0.1810 | 0.000 | 0.511 | ✗ |

→ OU beats persistence at its design horizon on **3/5 (60% ≥ 50%)** mean-reverting spreads, statistically significant.

**REVISED VERDICT (Rule 30/34): KEEP (as a mean-reversion expert) — conditional.** On its DESIGN-APPROPRIATE data at the CORRECT horizon, OU earns a real, significant edge over persistence (3/5 ≥ the 50% bar). Honest caveats: (1) OU's k-step forecast still equals AR(1)'s, so it ties — does not *beat* — the linear baseline; its distinct value is the interpretable mean-reversion framing + diagnostics (μ, half-life, z_dev) and the horizon-aware reversion signal; (2) it has NO edge on raw returns (1-step) — use it only in mean-reverting regimes (spreads/vol), gated by half-life. The earlier "REJECT 0/5" is RETRACTED as a wrong-horizon artifact. **Lesson banked: Rule 34/35 caught a false-reject — always test at the model's design data AND design horizon before judgment.**

### 2026-06-23 — IDEA-026 · RMT Covariance Cleaning (eigenvalue clipping) — KEEP ✅
- module: `pattern_brain/rmt.py` (utility, not a forecasting node) · category: risk/allocation
- Code-correct: oracle test (`tests/test_rmt.py`) — MP edge λ+=σ²(1+√q)² exact, trace-preserving clip, PSD, planted-spike recovery — all pass.
- Design-appropriate task (Rule 34/35): OOS minimum-variance portfolio realized variance on a high-dim universe (60 crypto assets, `data/universe/`), swept over train-window sizes (Rule 36 varied conditions). Lower = better.

| T_train | q=N/T | sample | ledoit-wolf | rmt_clip | rmt<sample | rmt<LW | verdict |
|---|---|---|---|---|---|---|---|
| 60 | 1.00 | 1.34e-4 | **8.25e-6** | 1.34e-4 | 0.00 | 0.00 | — (falls back to sample at q≥1) |
| 90 | 0.67 | 1.28e-5 | 7.43e-6 | **4.99e-6** | 0.98 | 0.89 | KEEP |
| 120 | 0.50 | 8.47e-6 | 6.67e-6 | **5.25e-6** | 0.84 | 0.74 | KEEP |
| 180 | 0.33 | 6.27e-6 | 5.91e-6 | 6.06e-6 | 0.60 | 0.62 | KEEP |
| 250 | 0.24 | **6.11e-6** | 5.93e-6 | 6.78e-6 | 0.65 | 0.57 | — |

**VERDICT: KEEP.** RMT cleaning delivers a real, large OOS risk reduction over the sample covariance in its design regime (high q≈0.5–0.67: 84–98% of windows, also beating Ledoit-Wolf 74–89%), tested across varied conditions + the high-dim adversarial regime (Rule 36). This is the **first bank model to genuinely BEAT its baseline** (OU only tied AR(1)). Honest caveats: (1) at q≥1 the clip is ill-posed and falls back to sample cov, where **Ledoit-Wolf dominates** — a fixable limitation (fall back to LW, not sample); (2) at low q (≤0.33) sample cov is already good, so RMT's edge fades. Use in the high-dimensional regime (N comparable to T). Not a forecasting Node — a covariance estimator for risk/allocation, evaluated by its own portfolio benchmark (NOT the 1-step return harness — that would be the Rule-34 wrong-task mistake).

### 2026-06-23 — IDEA-024 · Hawkes Self-Excitation (order-flow intensity) — SHADOW/REJECT ⚠️
- module: `pattern_brain/hawkes.py` (utility; operates on event times, not bars) · category: point-process/microstructure
- Code-correct: oracle test (`tests/test_hawkes.py`) — O(N) Ogata loglik == O(N²) brute force; MLE recovers the branching ratio on a simulated path; Poisson stream → n≈0 (no fabricated excitation) — all pass.
- Design-appropriate data (Rule 34/35): REAL Binance aggTrades tick streams (4 symbols × 40k trades, `data/ticks/`). Task = held-out point-process log-likelihood/event, walk-forward (18 windows/symbol). Baseline = homogeneous Poisson (no-self-excitation null). **Adversarial control (§F): shuffle inter-arrivals (destroys clustering); a genuine self-excitation edge MUST collapse.**

| symbol | hawkes_ll | poisson_ll | Δ | win% | n=α/β | real Δ vs shuffled Δ |
|---|---|---|---|---|---|---|
| BTC | 9.20 | 1.22 | +7.98 | 1.00 | 0.62 | +7.98 vs +7.25 → **LEAK** |
| ETH | 7.17 | 1.06 | +6.11 | 1.00 | 0.72 | +6.11 vs +6.78 → **LEAK** |
| DOGE | −0.05 | −1.03 | +0.99 | 1.00 | 0.33 | +0.99 vs +0.92 → **LEAK** |
| SOL | +0.17 | −0.50 | +0.67 | 1.00 | 0.23 | +0.67 vs +0.60 → **LEAK** |

**VERDICT: SHADOW / REJECT (self-excitation claim not supported).** Hawkes beats the Poisson null on 4/4 symbols, but the edge **survives inter-arrival shuffling on 4/4** (genuine-clustering increment Δ_real−Δ_shuffled ≈ 0, even negative for ETH). So the advantage comes from fitting the **over-dispersed inter-arrival marginal** (many short gaps), NOT temporal self-excitation. The Poisson baseline was too weak; against the proper **renewal** control (shuffled) Hawkes shows no genuine clustering edge. 🔬 The §F adversarial control **prevented a false KEEP** — the mirror image of the OU case where horizon testing prevented a false REJECT. Follow-up before any KEEP (IDEA-055): score Hawkes vs an explicit renewal-process baseline and use cross-window history in the held-out intensity.

### 2026-06-23 — IDEA-001/005 · Chaos/Lyapunov "structure vs noise" — REJECT as forecaster / SHADOW as feature ⚠️
- module: `pattern_brain/chaos.py` (utility; Takens embedding + Rosenstein LLE + nearest-neighbor analog forecast) · category: dynamical-systems/chaos (Axis-D gap)
- Code-correct: oracle test (`tests/test_chaos.py`) — embedding shape; logistic map predictable (skill>0.9) + LLE>0.3; noise has NO genuine structure (real−shuffled<0.1); logistic genuine structure>0.5 — all pass.
- Honest metric (REFINE-phase fix): raw analog-skill-vs-persistence is CONFOUNDED on zero-mean data (predicting the mean beats persistence with no structure), so the verdict uses **genuine structure = skill(real) − skill(shuffled)** (the §F shuffle control built in).

| series | skill_real | skill_shuf | GENUINE | LLE | verdict |
|---|---|---|---|---|---|
| logistic_map | 1.000 | 0.391 | **0.609** | 0.689 (≈ln2) | STRUCTURE ✓ |
| lorenz_x (subsampled) | 0.967 | 0.383 | **0.584** | 0.176 | STRUCTURE ✓ |
| real returns (10 panel) | ~0.46 | ~0.38 | **0/10 > 0.2** (mean +0.08) | — | noise/none |

**VERDICT: detector VALIDATED; REJECT as a return forecaster, SHADOW as a regime/predictability FEATURE.** The detector correctly recovers structure + the Lyapunov exponent on both canonical chaotic systems (logistic LLE≈ln2; Lorenz), and correctly finds **no exploitable chaos in real returns (0/10)** — confirming the well-established result that markets are not low-dimensional deterministic chaos. So no return-forecasting edge → not a bank forecaster; the LLE/predictability estimate is a legitimate regime-gating feature (shadow). Two reusable lessons reinforced: (1) the HORIZON/SAMPLING trap recurred — finely-sampled Lorenz made 1-step persistence trivially perfect (cf. OU 1-step), fixed by subsampling; (2) the mean-regression confound on zero-mean data, fixed by the shuffle-control genuine-structure metric (cf. Hawkes). Bank so far: OU=KEEP(cond), RMT=KEEP, Hawkes=SHADOW, Chaos=SHADOW(feature).

### 2026-06-23 — IDEA-023 · Path Signatures (rough-path universal features) — KEEP (as a feature multiplier) ✅
- module: `pattern_brain/signatures.py` (truncated signature via Chen's identity; numpy) · category: sequence-feature transform (multiplier/enabler, not a forecaster)
- Code-correct: oracle test (`tests/test_signatures.py`) — level-1 = total increment; level-2 Lévy area = 0.5 on the L-triangle; Chen's identity sig(full)=chen(halves); reparametrization invariance; dim formula — all pass.
- Design-appropriate capability test (Rule 34/35 — its job is ORDER): label = sign of the path's Lévy area (order-dependent). Linear model on **signatures 0.995** vs **order-invariant summary features 0.553** → signatures capture order that mean/std/min/max/increment cannot. PASS.
- Real-returns add-on (honest): next-return SIGN, signatures vs raw lags across the 10-panel — both ~chance (0.50–0.53), signatures beat raw on 1/10 (mean Δ +0.002). No standalone edge on efficient returns (expected).

**VERDICT: KEEP as a feature transform / MULTIPLIER, not a standalone forecaster.** Signatures provably do their design job (linearize order-dependent path functionals: 0.99 vs 0.55) — validated on the controlled task. As expected they give no standalone return-prediction edge (efficient markets), so their value is as a universal feature FEEDING the bank's experts (the IDEA-057 pivot: build multipliers, not more standalone predictors). First KEEP earned as an enabler rather than a predictor. Bank so far: OU=KEEP(cond), RMT=KEEP, Hawkes=SHADOW, Chaos=SHADOW(feature), Signatures=KEEP(feature/multiplier).

#### Rule-34 cross-domain validation — signatures on REAL non-trading data (UCI pendigits handwriting)
Tested on a COMPLETELY DIFFERENT real domain (pen-stroke trajectories, the canonical signature benchmark), not just synthetic paths + trading returns. Same linear classifier, real train/test split (7494/3498), 10 classes:

| features | dim | test acc |
|---|---|---|
| order-invariant (control) | 8 | 0.507 |
| signatures L2 | 6 | 0.786 |
| signatures L3 | 14 | 0.847 |
| raw flattened trajectory | 16 | 0.916 |

→ Signatures genuinely encode real handwriting trajectories (0.85 vs 0.51 order-blind, vs 0.10 chance) — **validated as a real sequence-feature representation on a non-trading domain**. Honest nuance: on these SHORT fixed-length (8-point) paths, raw flattening (0.92) still wins; signatures' advantage is for long / variable-length / irregular / multi-channel paths and as a compact universal feature. Confirms the KEEP (feature multiplier) verdict with cross-domain real evidence.

### 2026-06-23 — IDEA-060 · Integration layer (consume the validated features) — IMPLEMENTED ✅
- module: `pattern_brain/integrated.py` — turns shelf parts into consumed capabilities.
- Code-correct: oracle test (`tests/test_integrated.py`) — min_variance_weights matches analytic Σ⁻¹1/(1ᵀΣ⁻¹1); rmt_allocator returns valid weights; integrated_features has the right length + finite; regime predictability separates logistic-chaos from noise (after REFINE-fix: predictability uses NONLINEAR genuine structure, since the logistic map has ~0 linear autocorrelation — consuming the chaos finding properly).
- **Consumes:** RMT→`rmt_allocator` (sizing); signatures+chaos+OU→`integrated_features` (unified order-aware + regime expert input); chaos genuine-structure + OU half-life/z_dev→`regime_diagnostics` (gate features).
- Integration earns-its-keep (Rule 30/36) — OOS min-variance realized variance on the 60-asset universe, swept windows:

| T_train | q=N/T | equal-weight | sample-cov | RMT allocator | rmt<sample | rmt<equal |
|---|---|---|---|---|---|---|
| 90 | 0.67 | 4.97e-5 | 1.28e-5 | **4.99e-6** | 0.98 | 1.00 |
| 120 | 0.50 | 5.09e-5 | 8.47e-6 | **5.25e-6** | 0.84 | 1.00 |
| 180 | 0.33 | 5.27e-5 | 6.27e-6 | **6.06e-6** | 0.60 | 1.00 |
| 250 | 0.24 | 5.44e-5 | 6.11e-6 | 6.78e-6 | 0.65 | 1.00 |

**VERDICT: IMPLEMENTED / KEEP.** The RMT allocator beats BOTH sample-cov min-var and equal-weight on 3/4 conditions, beating equal-weight on 100% of windows with a ~10× OOS risk reduction (~5e-6 vs ~5e-5). The one clear model edge (RMT) is now a CONSUMED sizing decision, and signatures + chaos + OU are unified into one expert-input/gate feature vector. Turned the build sweep's validated parts into a working capability instead of shelfware.

### 2026-06-23 — IDEA-061 · Cost-aware backtest of the RMT allocator — RMT CONFIRMED cost-robust (as a risk overlay)
- `tools/eval_rmt_backtest.py` (+ `long_only_min_variance_weights`, `portfolio_turnover` in integrated.py). Walk-forward rebalancing on the 60-asset universe (1h bars), net of turnover cost, fee swept 0/5/10 bps.

| fee | strategy | net Sharpe | ann Vol | ann Ret | maxDD | avg turnover |
|---|---|---|---|---|---|---|
| 0bps | equal | −2.82 | 0.663 | −1.87 | −0.28 | 0.044 |
| 0bps | sample_mv | −5.36 | 0.274 | −1.47 | −0.21 | **2.461** |
| 0bps | rmt_mv | −5.40 | **0.217** | −1.17 | −0.15 | **0.371** |
| 0bps | rmt_long_only | −3.83 | 0.240 | −0.92 | −0.14 | 0.245 |
| 10bps | sample_mv | **−9.18** | 0.278 | −2.55 | −0.26 | 2.461 |
| 10bps | rmt_mv | −6.15 | 0.217 | −1.34 | −0.16 | 0.371 |
| 10bps | rmt_long_only | −4.28 | 0.240 | −1.03 | −0.15 | 0.245 |

**VERDICT: RMT's edge is REAL and COST-ROBUST — as a risk overlay, not an alpha.** Net of cost RMT delivers (1) the **lowest realized volatility** (0.217 vs sample 0.274 vs equal 0.663), (2) **6.6× lower turnover** than sample min-var (0.37 vs 2.46) — cleaning stabilizes weights, so (3) it **survives transaction costs far better** (sample_mv Sharpe collapses −5.4→−9.2 with fees; RMT degrades gently −5.4→−6.2), and (4) the **smallest drawdown** (−0.15). All Sharpes are NEGATIVE because the ~41-day window was a falling market (equal-weight −28% DD) and minimum-variance minimizes RISK, not return — it loses least, it doesn't generate alpha. Honest caveats: short single-regime window; weights fixed intra-holding-period; no risk-free rate. **Conclusion:** RMT is a usable, cost-robust SIZING/RISK overlay (lowest vol, turnover, drawdown) that must be paired with a return signal to make money; the long-only variant is the realistic tradable form and the most cost-robust. Confirms IDEA-026/060 net of real costs.

### 2026-06-23 — IDEA-064 · RMT risk overlay + momentum return signal — PROMISING (first positive net Sharpe) ✅⚠️
- `tools/eval_rmt_momentum.py` (+ `mean_variance_weights`, `cross_sectional_momentum` in integrated.py). Markowitz tilt w ∝ Σ⁻¹μ with μ = cross-sectional momentum; Σ from sample vs RMT-cleaned. Walk-forward, net of turnover, all strategies L1-normalised to gross=1.

| @10bps | net Sharpe | annVol | maxDD | avgTurn |
|---|---|---|---|---|
| ew_momentum (signal, no cov) | −2.04 | 0.338 | −0.14 | 0.571 |
| mv_sample_mom (Markowitz sample-cov) | +0.32 | 0.073 | −0.03 | 0.792 |
| **mv_rmt_mom (Markowitz RMT-cov)** | **+3.94** | 0.125 | −0.04 | 0.602 |
| rmt_minvar (pure risk, no signal) | −5.78 | 0.118 | −0.09 | 0.200 |

**Robustness (net Sharpe @10bps by lookback L): mv_rmt_mom positive on ALL windows** — L=90:4.57, 120:3.94, 150:3.53, 180:1.62, 250:3.59 (sample-cov erratic −1.56..4.28; RMT decisively wins in the high-q short-window regime). **Signal-shuffle permutation null (20 shuffles): real +3.94 vs null mean −8.10 (max −4.38), p=0.000** → the edge is genuinely the momentum signal aligned with assets, not covariance/artifact.

**VERDICT: PROMISING (provisional KEEP).** First positive, robust, cost-aware, significance-tested net-Sharpe strategy of the whole sweep. All four claims hold: signal adds return (+3.94 vs −5.78 pure-risk), covariance overlay beats naive momentum (+3.94 vs −2.04), **RMT stabilisation is the difference between viable and cost-wiped (+3.94 vs sample +0.32)**, and the edge passes the shuffle null (p=0.000). HONEST CAVEATS (Rule 30/36): single ~41-day crypto regime — the null proves the signal-edge is real WITHIN this window but NOT that it survives regime change (momentum crashes in reversals); absolute Sharpe magnitude is statistically thin as a forward estimate; no risk-free rate; weights fixed intra-period. **Provisional, NOT deployable alpha** until confirmed on multi-regime / longer out-of-sample data (IDEA-063). This is the synthesis the sweep was building toward: validated RMT risk tool + a return signal = the first thing that makes money net of cost, in-sample-window-significant.

### 2026-06-23 — IDEA-065 · Multi-regime OOS confirmation — the +3.94 did NOT survive (regime artifact) ⚠️→ downgrades IDEA-064
- `tools/eval_multiregime.py` on multi-year DAILY data (`data/universe_daily/`, 890 days × 42 assets, 2021-09..2024-02 = 2022 bear + 2023 bull + 2024 chop). Net of 10 bps.

| full-period strategy | net Sharpe (2.5yr) |
|---|---|
| ew_momentum | −0.27 |
| mv_sample_mom | −0.25 |
| mv_rmt_mom (the IDEA-064 "win") | **+0.30** |
| rmt_minvar (pure RMT risk overlay) | **+1.29** |

- mv_rmt_mom per-year: 2022 bear −0.11, 2023 bull +0.59, 2024 chop +1.69 → regime-dependent.
- Nested OOS (pick L on 1st half → confirm on 2nd half): best L=60 (1st-half Sharpe −0.04) → 2nd-half OOS +1.07 (weak/uninformative selection).
- **Signal-shuffle permutation null (full period): real +0.30 vs null mean −0.23 (max +0.79), p=0.200 — NOT significant.**

**VERDICT: IDEA-064 DOWNGRADED PROMISING → REJECT (momentum-alpha claim).** The +3.94 single-window net Sharpe was a REGIME ARTIFACT: over 2.5 multi-regime years the RMT-momentum edge is modest (+0.30) and fails the permutation null (p=0.20). Momentum works in 2023-24 (bull/chop) but not the 2022 bear — not a regime-robust alpha. **The durable, multi-regime-confirmed edge is RMT as a RISK OVERLAY: rmt_minvar has the BEST full-period Sharpe (+1.29)** across bull/bear/chop — consistent with the whole sweep (RMT's value is robust risk reduction, not return). 🔬 This is the rigor framework (Rules 34/36 + permutation null + nested OOS) doing its most important job yet: PREVENTING a false alpha (the exciting single-window result) from being shipped. Caveat: rmt_minvar's +1.29 itself warrants its own null/significance + cost-of-low-vol-tilt check before deployment (IDEA-067). Bank truth after the full arc: RMT-risk-overlay = the one durable, multi-regime edge; standalone return-prediction (incl. momentum) does not survive honest OOS.

### 2026-06-23 — IDEA-067 · Audit of rmt_minvar's +1.29 — cleaning is REAL, Sharpe NOT yet significant
- `tools/audit_rmt_minvar.py` (+ `inverse_vol_weights` in integrated.py). Daily multi-regime universe, net 10 bps.

| test | result | reading |
|---|---|---|
| full-period Sharpe: equal / inverse-vol / sample-mv / **rmt-mv** | −0.71 / −0.64 / 0.85 / **1.29** | edge is NOT the low-vol anomaly (inverse-vol LOSES) |
| time-shuffle null (kills cross-asset correlation) | real 1.29 vs null −2.84, **p=0.000** | correlation-cleaning genuinely adds value |
| effective N / low-vol tilt / top-5 share | 16.4/42 / corr +0.84 / 0.46 | somewhat concentrated + low-vol-tilted |
| drop top-5 weighted assets | Sharpe 0.78 (was 1.29) | edge survives but weakens |
| **block-bootstrap 95% CI on Sharpe** | **[−0.06, 2.52]** | **NOT significant (includes 0)** |

**VERDICT: RMT covariance-CLEANING is validated; the +1.29 Sharpe is NOT statistically significant.** Two honest, reconcilable findings: (1) the edge is genuine correlation-structure cleaning — it beats inverse-vol (which loses) and collapses to −2.84 when correlations are destroyed (p=0.000), so it is NOT merely the low-volatility anomaly; (2) BUT the realized portfolio Sharpe over 2.5 years is not reliably > 0 (bootstrap CI includes 0), and the weights are concentrated (top-5=46%) with a strong low-vol tilt (+0.84). 🔬 The bootstrap CI PREVENTED overclaiming +1.29 as proven alpha. **Precise conclusion:** RMT is a validated risk-reduction MECHANISM (lower OOS variance + turnover + uses real correlation structure), but a min-variance portfolio's positive return is statistically UNCONFIRMED on the available history — it needs longer data + concentration/exposure constraints before being treated as a deployable edge. RMT = a real risk tool, not (yet) a proven money-maker.

### 2026-06-23 — IDEA-069 · RMT as a vol-targeting risk overlay — overlay WORKS, but RMT≈sample for FORECASTING (precise scoping)
- `tools/eval_risk_overlay.py` (+ `forecast_vol`, `risk_overlay` in integrated.py). Scale a signal's weights so RMT/sample/diagonal-forecast vol = 15% target; multi-regime daily; judged by realised-vol tracking (not Sharpe).

| signal | forecaster | realized vol | \|vol−target\| | track err |
|---|---|---|---|---|
| equal | diagonal (no corr) | 0.771 | 0.621 | 0.628 |
| equal | sample | 0.152 | 0.002 | 0.058 |
| equal | rmt | 0.151 | 0.001 | 0.057 |
| momentum | diagonal | 0.126 | 0.024 | 0.056 |
| momentum | sample | 0.146 | 0.004 | 0.056 |
| momentum | rmt | 0.179 | 0.029 | 0.069 |

**VERDICT: the vol-targeting overlay works (correlations essential — diagonal blows up to 77% vol), but RMT provides NO advantage over sample-cov for vol FORECASTING** (ties on equal-weight, slightly WORSE on momentum). Precise scientific scoping of where RMT helps: **RMT's value is in min-variance WEIGHT CONSTRUCTION (where sample-cov INVERSION amplifies noise — proven in IDEA-026/061/067), NOT in vol FORECASTING (computing wᵀΣw needs no inversion, so sample-cov already suffices).** So the deployable risk overlay = use **rmt_minvar for the WEIGHTS** (the proven inversion-sensitive step) and sample-or-RMT for the vol-target scaling (RMT not required there). Another honest negative that sharpened the picture: RMT is not a universal "better covariance" — it is specifically a better *inverse*-covariance for portfolio construction.

### 2026-06-23 — IDEA-071 · RMT for Kelly/max-Sharpe sizing — CONFIRMS inversion-sensitivity thesis (synthetic), signal-limited (real)
- `tools/eval_rmt_kelly.py` (+ `kelly_weights` Σ⁻¹μ in integrated.py).

**(A) Controlled synthetic (isolates the Σ-inversion question; known μ,Σ; q=N/T=0.67):**
| task | optimal (true Σ) | sample-Σ⁻¹ | RMT-Σ⁻¹ | RMT advantage |
|---|---|---|---|---|
| max-Sharpe | 0.136 | 0.079 (58% of opt) | **0.116 (85% of opt)** | **+0.037** |
| min-var | ~0* | ~0 | ~0 | ~0 |
*(min-var Sharpe ≈0 here as μ is factor-aligned, not a fair Sharpe comparison; min-var's RMT gain is in VARIANCE — IDEA-026.)

→ RMT recovers **85% of the optimal max-Sharpe vs sample's 58%** — a large inversion-robustness gain; advantage is larger for max-Sharpe than min-var, **confirming the thesis that RMT's cleaned inverse-covariance matters MOST for the most inversion-noise-sensitive sizing (Kelly/max-Sharpe)**.

**(B) Real multi-regime (momentum μ, net @10bps):** sample −0.25 (CI [−1.61, 1.11]), rmt +0.30 (CI [−1.19, 1.82]) — **both NOT significant**. RMT cannot rescue a non-generalizing return signal (consistent with IDEA-065).

**VERDICT: thesis CONFIRMED in the controlled setting; real-data gated by signal quality.** This completes the precise map of where RMT helps, ordered by inversion-sensitivity: **vol-forecasting (no inverse) → no edge (IDEA-069); min-var (inverse, moderate) → real modest edge (IDEA-026/067); max-Sharpe/Kelly (inverse, high) → largest edge (here, 85% vs 58% recovery).** But on REAL data RMT's benefit is realisable only with a return signal that itself generalises OOS — and none of the tested signals do. So RMT is a validated *inverse-covariance estimator* whose value grows with inversion-sensitivity, deployable as a sizing improver ONCE a regime-robust signal exists.

### 2026-06-23 — WAVE-2 batch (3 parallel agents, orchestrator-consolidated) — IDEA-006, IDEA-028, IDEA-029
Built by 3 concurrent agents, each bound to the Code-Generation Contract (oracle-first) + Rules 30/34/35/36, strict file isolation (own files only); orchestrator verified + consolidated. Full suite 326 green (the one dashboard-route failure an agent saw was an environmental concurrency flake — passes alone).

- **IDEA-006 · Causal-Inference Expert** (`pattern_brain/causal.py`: Granger F-test via QR-lstsq + transfer-entropy). Oracle 6/6 (synthetic X→Y: Granger/TE directional, Y→X null; independent series: neither). **Verdict: REJECT on daily crypto** — correlations high (0.63–0.86) but 0/15 directed pairs survive the time-shuffle surrogate null → cleanly shows causality ≠ correlation; no exploitable daily lead-lag. Estimators validated correct; needs intraday/tick or a non-financial directed-causality benchmark for a complete Rule-36 verdict.
- **IDEA-028 · No-Regret Meta-Combiner** (`pattern_brain/no_regret.py`: log-space Hedge/multiplicative-weights). Oracle 6/6 (regret ≤ √((T/2)lnN), avg regret→0, concentrates on best expert, no harm when equal). **Verdict: KEEP (meta-layer)** — on real panel with persistence/rolling-mean/AR(1) experts: regret within bound 10/10 (~10 vs 23.3), beats equal-weight 10/10, sits on best-fixed-in-hindsight; discovered AR(1) as best OOS unaided. Cannot beat best-in-hindsight (that's the oracle it chases). Follow-up: register as a fusion layer over bank nodes.
- **IDEA-029 · Mandelbrot Stylized-Facts Validator** (`pattern_brain/stylized_facts.py`: heavy tails/Hill, vol-clustering ACF, linear-autocorr, aggregational gaussianity, leverage). Oracle 11 pass (real BTC → facts TRUE; Gaussian IID → FALSE). **Verdict: KEEP (validation gate)** — across 10 real assets all core facts hold (excess kurt 3.6–7.2, |r|-ACF +0.11..+0.22); correctly REJECTS Gaussian IID and ACCEPTS a faithful GARCH(1,1) generator (accepts a *good* model, not just non-Gaussian). Judged by correctness, not PnL.

Running bank scorecard: KEEP — RMT(risk/sizing), Signatures(feature), No-Regret(meta-combiner), Stylized-Facts(validator); conditional — OU; SHADOW/REJECT — Hawkes, Chaos, Causal(daily), RMT+momentum(regime artifact). Theme holds: the durable wins are tools/features/meta-layers; standalone return-prediction keeps failing honest OOS.

### 2026-06-23 — TIER-3 timeless-foundations build (12 nodes, 4 parallel agents) — TIER-3 ✅ COMPLETE
Built the 12 remaining TIER-3 ⬜ concepts as registered Nodes via 4 file-isolated agents, each oracle-test-first (52 oracle tests pass; full bank still conforms, 69/69 on tier3+bank+interlingua). Per-node detail in `MODEL_REPORT_tier3_{probability,stochastic,control,pricing}.md`. **Honest verdicts (the project lesson held — none beats persistence on point forecast):**

| node_type | belief | oracle | real-data verdict |
|---|---|---|---|
| `bayes_update` | forecast | Normal-Normal posterior <1e-10 | REJECT-as-forecaster (skill hollow, perm p≈0.64–1.0, 0/10); keep as uncertainty feature |
| `maxent_dist` | density | recovers μ,σ <1e-6, ∫pdf=1 | KEEP-as-utility (flags ill-conditioning on heavy tails) |
| `chebyshev_bound` | forecast | k=3 coverage ≥88.9% on t(4) | KEEP-as-utility (distribution-free interval, 0.92–0.95 cov 10/10) |
| `gbm_baseline` | forecast | drift/vol recovery; lognormal mean <1e-8 | KEEP-as-utility (it IS the random-walk baseline, skill≈0 by design) |
| `langevin_sampler` | forecast | γ,μ recovery; cond-mean ≡ AR(1)/OU <1e-8 | SHADOW (explicit-SDE re-skin of OU; ties persistence on its design spreads) |
| `fokker_planck` | density | transition density ∫=1, moments match OU | KEEP-as-utility (calibrated next-value density) |
| `log_utility` | decision | f*=μ/σ² exact | KEEP-as-utility (Kelly sizing; proxy-Sharpe +0.017 vs full-exposure −0.034) |
| `lagrange_opt` | decision | weights vs analytic 5.6e-17 | KEEP-as-utility (min-var, 80.5% OOS variance reduction vs equal-weight) |
| `hjb_control` | decision | value-iteration residual→0 | KEEP-as-utility (Bellman/HJB optimal hold-vs-exit; covers D10 too) |
| `momentum_kinematics` | forecast | exact on constant-accel series | REJECT (0/10 vs persistence; momentum noise on near-random-walk crypto) |
| `almgren_exec` | decision | sinh schedule <1e-9, →TWAP as κ→0 | KEEP-as-utility (Pontryagin optimal execution; covers D10) |
| `bs_pricing` | signal | put-call parity 7e-15, Greek signs | KEEP-as-utility (pricing identity, not a predictor) |

Net: 2 REJECT-as-forecaster + 1 SHADOW + 9 KEEP-as-utility — consistent with the standing truth (tools/features/control utilities earn their place; standalone point-prediction does not). TIER-3 (timeless foundations) now 26/26 in `CONCEPT_EQUATION_BANK.md`. Bank: 177 registered light nodes (+12).

## 2026-06-24 — TIER-1 batch 2 (johansen_coint, har_rv, copula_dependence, bocpd_break)
| node | type | oracle | real-data verdict |
|---|---|---|---|
| `har_rv` | forecast | beats persistence on GARCH-sim RV | **KEEP — beats persistence on 10/10 panel symbols, skill +0.20…+0.48 (median ≈+0.46); first TIER-1 node to beat a baseline (forecasts VOLATILITY, which is predictable, not returns)** |
| `johansen_coint` | signal | ADF: RW non-stat / AR(.2) stat; EG recovers hedge 2.0, rejects indep walks | KEEP-as-utility — oracle-correct; honest real result = crypto large-caps near-unit-root / weakly cointegrated (1/5 spreads, 0/5 pairs clear the strict bar in this corpus); useful hedge-ratio/half-life/z-spread feature |
| `copula_dependence` | signal | t-copula tail-dep > Gaussian-copula | KEEP-as-utility — ETH/BTC τ=0.71, tail-dep≈0.72 vs ~0.06 independent control; surfaces joint-crash risk a correlation misses |
| `bocpd_break` | anomaly | mean shift localized ±8 | KEEP-as-utility — real spliced regime break t=300 detected at t=305 (±5); detector = run-length-posterior collapse (see IMP-002) |

Net: **1 genuine forecaster KEEP (har_rv)** + 3 KEEP-as-utility (risk/regime/stat tools). The project
truth holds and is sharpened: standalone *return* forecasters lose to persistence, but a *volatility*
forecaster (HAR-RV) wins decisively — vol is the predictable target. Bank: 80 registered Node types.

## 2026-06-24 — TIER-1 batch 3 (18 nodes via 5 parallel file-isolated agents) — completes the directly-buildable cluster
Built oracle-test-first under the Code-Generation Contract; 110 passed in the integrated regression subset; all conform. Per-family reports: MODEL_REPORT_tier1_{microstructure,allocation,stochastic,dynamical,softcomputing}.md.
| node | family | verdict | headline evidence |
|---|---|---|---|
| `ofi` | microstructure | KEEP-as-utility | corr(OFI, bar return) +0.26…+0.56 across 4 symbols vs ~0 shuffled control |
| `vpin` | microstructure | KEEP-as-utility | oracle 0.97 toxic / <0.25 balanced; toxicity↔vol NOT confirmed on a calm sample (needs stress data) |
| `kyle_impact` | microstructure | KEEP-as-utility | recovers λ=0.7 (r²>0.9); real λ>0 all symbols; cross-asset ordering confounded by price level (documented) |
| `hrp_alloc` | allocation | KEEP-as-utility | OOS variance 2.12e-5 vs equal-weight 6.27e-5 (~⅓), beats inverse-variance; finite on rank-deficient cov |
| `cvar_opt` | allocation | KEEP-as-utility | OOS worst-5% loss −6.25e-3 vs equal-weight −1.74e-2 (~64% smaller); under-weights fat-tailed asset |
| `merton_alloc` | sizing | KEEP-as-utility | closed-form (μ−r)/(γσ²) exact to 1e-12; correct monotonicities |
| `mpc_position` | control | KEEP-as-utility | tracks signal while smoothing; turnover 190 vs 966 naive (~5× less churn); cost-ablation monotone |
| `jump_diffusion` | stochastic | KEEP-as-utility | jump separation precision 0.99/recall 0.63 (sub-σ jumps unrecoverable); λ recovered; 1.7–4.4% crypto jump rate |
| `heston_vol` | stochastic | **KEEP** | recovers θ/κ; variance forecast **beats persistence 10/10** (not claimed to beat HAR) |
| `sv_particle` | stochastic | KEEP-as-utility | filtered vol corr 0.76 with true latent (fixed log-y² attenuation bias); 0.69 with \|returns\| real |
| `percolation_risk` | physics | KEEP-as-utility | giant-component phase transition recovered; systemic-risk series corr 0.45 with market-wide moves |
| `koopman_dmd` | dynamical | **REJECT (returns)** / KEEP (oscillator) | recovers sine freq/decay; on returns beats persistence 10/10 but loses to ZERO baseline 0/10 |
| `bsts_forecast` | structural | KEEP (levels) / SHADOW (returns) | Kalman level tracks price 0.097% rel-err; loses to zero baseline 0/10 on returns |
| `infogeom_distance` | geometric | KEEP-as-utility | matches Fisher-Rao closed form 1e-9; vol-jump spike >4×; dynamic range 10/10 |
| `diffusion_map` | manifold | KEEP-as-utility | recovers 1-D arc \|Spearman\|=1.0; **Rule-36 cross-domain pendigits kNN 0.80 vs 0.10 chance** |
| `fuzzy_ts` | soft | SHADOW | tracks smooth level corr 0.996; beats persistence 0/10 on panel |
| `grey_gm11` | soft | KEEP-as-utility | recovers exp growth from 6 points (err 0.24 vs persistence 17.3); data-poor extrapolator; 0/10 on returns |
| `anfis` | soft | KEEP (nonlinear) / SHADOW (returns) | sin(x1)+x2² RMSE 0.099 vs linear 1.169 (>10×); 0/10 vs zero on returns |

Net batch 3: **2 forecaster KEEPs (heston_vol; har_rv was batch 2) — both VOLATILITY** + 11 KEEP-as-utility + 2 conditional KEEP (oscillator/nonlinear home turf) + 3 SHADOW/REJECT on returns. The project truth holds decisively: **standalone RETURN forecasters lose to the zero/persistence baseline; VOL/RISK/CONTROL tools earn their keep by correctness.** TIER-1 directly-buildable cluster COMPLETE (23/23). Bank: 96 registered Node types.
⏳ Rule-36 dual-domain (one trading + one real NON-trading dataset per general-purpose model) verification pass: next.
