# Ideas Backlog — Pattern Brain
_Append-only log of every 💡 Ideas block Claude generates (owner rule 29, 2026-06-23)._
_Newest at the bottom, dated. Each idea is a PROPOSAL — nothing here is implemented until the owner says "implement"._

**Status legend:** `proposed` (default, awaiting decision) · `approved` (greenlit, not yet built) · `implemented` (built + verified) · `dismissed` (declined). Update an idea's status in place when it changes.
**IDs:** every idea has a stable `IDEA-NNN` id (never reused, even if dismissed) so it can be referenced directly ("approve IDEA-003"). **next id: IDEA-088**

---

## ✅ Approved — ready to implement
_Ideas the owner has greenlit (status `approved`). When the owner says "implement", build these in order. Empty = nothing waiting._

- _(none yet)_

---

## 📊 Build-priority shortlist (ranked — all 3 tiers folded, 2026-06-23)
_Ranked by: proven edge > elegance · enablers/multipliers before single predictors · decorrelation > count ·
low blast radius · dependency order. Build top-down. Each is still a Rule-30 proposal (approval-gated)._

**WAVE 0 — Foundation & rigor — ✅ IMPLEMENTED 2026-06-23 (8/8 tests pass; verified on real BTC data)**
1. ✅ **IDEA-014 · Baseline + permutation-null + bootstrap-CI harness** — `pattern_brain/harness.py`. Verified: rewards AR(1) edge, rejects random walk; on real BTC correctly refused to certify a shrinkage artifact as edge.
2. ✅ **IDEA-019 · Common decision-vector belief-type (P0)** — `pattern_brain/decision_vector.py`; registered in interlingua, projects any belief.
3. ✅ **IDEA-027 · Conformal-prediction wrapper** — `pattern_brain/conformal.py`. Verified on real BTC: target 0.90 → empirical 0.897 coverage.

**WAVE 1 — Proven, direct, low-risk experts (prove the invention pipeline end-to-end)**
4. ✅ **IDEA-025 · Ornstein-Uhlenbeck mean-reversion** (Hooke→OU) — BUILT + TESTED 2026-06-23 (`OrnsteinUhlenbeckNode`, econometrics.py). **Verdict: KEEP (conditional, mean-reversion expert).** Design-data: mean-reverting spread LEVELS, horizon ≈ half-life (Rule 35 tag). At the CORRECT horizon OU beats persistence on **3/5** spreads, significant (p=0.000) — `tools/eval_ou_horizon.py`. Caveats: ties (≠ beats) AR(1); no edge on raw returns/1-step. ⚠️ Earlier "REJECT 0/5" RETRACTED — it was a wrong-horizon (1-step) test; Rule 34/35 caught the false-reject. First full Rule-30 pipeline run; triggered Rule 34/35 + the real-data download.
5. ✅ **IDEA-026 · RMT covariance cleaning (Marchenko-Pastur eigenvalue clipping)** — BUILT + TESTED 2026-06-23 (`pattern_brain/rmt.py`). **Verdict: KEEP** — first bank model to genuinely BEAT its baseline: OOS min-variance portfolio risk on a 60-asset universe, RMT < sample covariance on 3/5 train-window conditions (98%/84% of windows at q=0.67/0.50), also beating Ledoit-Wolf 89%/74% in-regime. Design-data (Rule 35): high-dim return universe (`data/universe/`), task = portfolio risk (not 1-step forecasting). Caveat: q≥1 falls back to sample (LW dominates there → see IDEA-052); low-q edge fades. Bun-Bouchaud RIE not yet built (clipping only).
6. ⚠️ **IDEA-024 · Hawkes intensity (order-flow self-excitation)** — BUILT + TESTED 2026-06-23 (`pattern_brain/hawkes.py`). **Verdict: SHADOW/REJECT** — code-correct (oracle: O(N) loglik==bruteforce, MLE recovers branching ratio, Poisson→n≈0), beats Poisson null 4/4 on REAL tick data (`data/ticks/`) BUT the edge survives inter-arrival shuffling 4/4 → it's a marginal-distribution artifact, not genuine self-excitation. §F adversarial control PREVENTED a false KEEP. Design-data (Rule 35): aggTrades tick streams. Follow-up IDEA-055 (renewal baseline + cross-window history) before any KEEP.
7. ⚠️ **IDEA-001/005 · Chaos / Lyapunov ("structure vs noise")** — BUILT + TESTED 2026-06-23 (`pattern_brain/chaos.py`: Takens embed + Rosenstein LLE + NN analog forecast). **Verdict: REJECT as forecaster, SHADOW as regime feature** — code-correct + VALIDATED on canonical chaos (logistic LLE 0.689≈ln2 genuine 0.61; Lorenz genuine 0.58) but **0/10 genuine structure on real returns** (markets aren't low-dim chaos). Metric = genuine structure = skill(real)−skill(shuffled) (§F control; fixes the mean-regression confound). Reinforced the horizon/sampling trap (Lorenz fine-sampling) + the zero-mean mean-regression confound.

**WAVE 2 — Diversifiers & multipliers (uncorrelated experts + rigor)**
8. ✅ **IDEA-023 · Path-signature features** — BUILT + TESTED 2026-06-23 (`pattern_brain/signatures.py`: truncated signature via Chen's identity). **Verdict: KEEP (feature multiplier, not forecaster)** — code-correct (oracle: level-1=increment, Lévy area=0.5, Chen identity, reparam invariance) + capability PASS (order-dependent label: signatures 0.995 vs order-invariant 0.553). No standalone return edge (both ~chance, expected). Value = universal order-aware features FEEDING experts (the IDEA-057 pivot). First KEEP as an enabler. **Cross-domain validated (Rule 34/36.5):** real UCI pendigits handwriting (NON-trading) — sig L3 0.847 vs order-blind 0.507 (chance 0.10); nuance: raw-flatten 0.916 wins on short fixed-length paths (signatures shine on long/variable-length). Triggered Rule 36 clause 5 (cross-domain test for general methods).
9. ⚠️ **IDEA-006 · Causal-inference expert (Granger / transfer-entropy)** — BUILT (agent) 2026-06-23 `pattern_brain/causal.py`. Oracle 6/6. **REJECT on daily crypto**: 0/15 directed pairs survive surrogate null (causality≠correlation shown); needs intraday/tick for full verdict.
10. ✅ **IDEA-028 · Online no-regret meta-combiner (Hedge/MW)** — BUILT (agent) 2026-06-23 `pattern_brain/no_regret.py`. Oracle 6/6 (regret≤√((T/2)lnN)). **KEEP (meta-layer)**: beats equal-weight 10/10 on real panel, tracks best expert unaided. Follow-up: register as fusion layer.
11. ✅ **IDEA-029 · Mandelbrot stylized-facts validator** — BUILT (agent) 2026-06-23 `pattern_brain/stylized_facts.py`. Oracle 11 pass. **KEEP (validation gate)**: detects facts in real crypto, absent in IID, accepts faithful GARCH. Auto-gate for generative models.
12. **IDEA-002 · Per-node Axis-D category tag** — cheap metadata that lets the router favor cross-discipline (decorrelated) combos.

**WAVE 3 — North-star architecture (Model Neural Network; sequence matters)**
13. **IDEA-015 · Universal Market Embedding (P1)** — the shared learned latent; the single most LLM-like piece.
14. **IDEA-021 / IDEA-016 · Gumbel-Softmax attention soft-router (P2)** — the trainable "router learns pathways" mechanism.
15. **IDEA-017 · Residual stream (P3)** + **IDEA-018 · animated dashboard (P-VIZ, parallel)**.
16. **IDEA-020 · Differentiable expert reimplementations** + **IDEA-022 · distillation** — raise how literal the end-to-end backprop can be.

**WAVE 4 — Frontier (gate HARDEST; assume baseline wins until proven)**
17+. diffusion synthetic data, Neural SDE/CDE, deep optimal stopping, Wasserstein DRO, distributional RL, TDA,
Schrödinger bridge, mean-field control, quantum. ⛔ NEVER: Gann / Fibonacci-as-prediction / Elliott / astro.

> Recommended first move: **WAVE 0 (IDEA-014 → 019 → 027), then IDEA-025 as the first real expert.** That proves
> the pipeline + rigor before spending effort on exotic models that usually lose to baselines.

---

## 📋 Idea log

### 2026-06-23 — ML model taxonomy (first pass)
- **IDEA-001 · Chaos/dynamical-systems expert node (Axis-D gap)** — cleanest missing discipline; answers "structure vs true noise." Highest confidence, lowest blast radius.
  **Status:** proposed
- **IDEA-002 · Per-node Axis-D category tag** — lets the gating network favor cross-discipline combos so correlated experts don't dominate. Cheap, metadata-only.
  **Status:** proposed
- **IDEA-003 · Information Bottleneck as the fusion objective** — formalizes "filter hard, keep only what predicts" via MI estimators already in `infotheory.py`. Highest payoff, most design work.
  **Status:** proposed

### 2026-06-23 — Taxonomy re-verification (axes + scientific categories)
- **IDEA-004 · Use Axis-D as a "diversity ledger," not a checklist** — the win isn't covering all 16; it's maximizing decorrelation between experts. Causal (#8) and game-theoretic (#9) experts are attractive precisely because their errors are unlikely to correlate with the existing statistical/physics nodes. Targets the real bottleneck (expert redundancy).
  **Status:** proposed
- **IDEA-005 · Chaos/dynamical expert (#7) as the cleanest first build** — lowest risk, pure-numpy, answers "structure vs noise." (Refines IDEA-001 with the corrected taxonomy numbering.)
  **Status:** proposed
- **IDEA-006 · Causal-inference expert (#8) — the sleeper pick** — distinguishes "X moved *because of* Y" from mere correlation; the one category that attacks the failure mode (spurious patterns) the whole "randomness→pattern" thesis is most exposed to. Higher design cost, highest conceptual payoff.
  **Status:** proposed

### 2026-06-23 — Idea-backlog management (meta)
- **IDEA-007 · Status field per idea** — tag each entry proposed / approved / dismissed / implemented so the backlog stays actionable; mirror the dashboard `IdeationAdvisor` status model.
  **Status:** implemented
- **IDEA-008 · Unify IDEAS.md with the agent's `ideas.json`** — the in-code `IdeationAdvisor` already maintains a ranked idea backlog (`agent_state/ideas.json`); merging conversation-ideas into the same store (with dedup) gives one backlog instead of two drifting lists. Medium; needs a small sync.
  **Status:** proposed

### 2026-06-23 — Backlog UX
- **IDEA-009 · One-line status summary at the top of IDEAS.md** — an auto-maintained tally (e.g. "N proposed · N approved · N implemented") so the backlog's shape is visible without scrolling. Cheap; refreshed on each append.
  **Status:** proposed
- **IDEA-010 · "Approved queue" section** — a short *Approved — ready to implement* list at the top so the moment the owner says "implement" there's an unambiguous build order; keeps Rule 27's gate clean.
  **Status:** implemented
- **IDEA-011 · Auto status-tally line** — the still-open counter from IDEA-009, refreshed whenever the file is touched. (Duplicate of IDEA-009; kept for traceability.)
  **Status:** proposed
- **IDEA-012 · Stable idea IDs** — give each idea a short `IDEA-NNN` id so the owner can approve/dismiss by reference instead of re-quoting text.
  **Status:** implemented

### 2026-06-23 — First model-invention candidates (Rule 30 prep)
- **IDEA-013 · Koopman-operator linear-embedding forecaster** — a dynamical-systems model (Axis-D #7) that lifts nonlinear dynamics into a space where they evolve ~linearly, then forecasts there. Directly on-theme with "randomness→pattern" (finds the hidden linear structure of a chaotic-looking series); pure-numpy (DMD/EDMD). Must beat AR/persistence OOS. Candidate names: "Koopman Lift Forecaster" / `koopman_dmd`.
  **Status:** proposed
- **IDEA-014 · Reusable baseline+null-test harness for new models** — a small helper that auto-benchmarks any new node vs persistence/linear/threshold baselines with a permutation null + bootstrap CI, so every invention (Rule 30 step 5) is tested identically. Process tool, not a model. Makes the rigor cheap and consistent.
  **Status:** implemented (`pattern_brain/harness.py`, 2026-06-23; tests/test_wave0.py)

### 2026-06-23 — Model Neural Network north-star (gaps to complete the LLM-style graph)
- **IDEA-015 · Universal Market Embedding encoder** — a learned encoder fusing all raw sources (candles, orderbook, funding, OI, news, sentiment, options, on-chain) into ONE shared latent vector every node reads. Gap #1 — the single most LLM-like missing piece (today nodes see raw (T,D)). High payoff, real design.
  **Status:** proposed
- **IDEA-016 · Attention soft-router + gradient-trained edge weights** — upgrade the `gated` combiner from regime-accuracy weights to a learned softmax-attention gate over experts (top-k activation), trained by backprop on outcomes. Gap #3. The differentiable part of the "backprop like an LLM" wish — feasible.
  **Status:** proposed
- **IDEA-017 · Residual stream in the StackedDAG** — carry a running representation through layers (skip connections) so info isn't destroyed across depth (the 0.9^20 problem). Gap #2.
  **Status:** proposed
- **IDEA-018 · Animated LLM-style NN dashboard viz** — on the EXISTING React Flow graph + `/ws/run` stream, add particle/packet flow along edges, node activation glow, edge-width=weight, attention overlay, residual-stream spine. Visual target: poloclub Transformer Explainer. Parallel track P-VIZ; can start early.
  **Status:** proposed
- **IDEA-019 · Common decision-vector belief-type** — standardize a shared output schema (e.g. {bullish, bearish, volatility, confidence}) as an interlingua belief-type so ANY node can connect to ANY node. Phase P0 — the enabler for the whole graph.
  **Status:** implemented (`pattern_brain/decision_vector.py`, 2026-06-23; tests/test_wave0.py)

### 2026-06-23 — Differentiable-graph enablers (from Rule 31 / IMP-001 research)
- **IDEA-020 · Differentiable expert reimplementations** — port experts that have differentiable forms (Kalman, HMM, soft decision trees, DSP) into PyTorch/JAX so they become TRUE gradient-trained neurons (refs: akloss/differentiable_filters, 30stomercury/hmm-backprop). Raises how literal the LLM-style backprop can be. Effort per expert; do the high-value ones first.
  **Status:** proposed
- **IDEA-021 · Gumbel-Softmax differentiable router** — make the attention router (IDEA-016) trainable end-to-end via Gumbel-Softmax + Straight-Through (differentiable top-k expert selection). The concrete mechanism that makes "the router learns by backprop" real.
  **Status:** proposed
- **IDEA-022 · Distillation wrapper for black-box experts** — distill non-differentiable experts (HDBSCAN, PySR) into small differentiable student nets so they can sit inside the gradient graph; fall back to REINFORCE if distillation underperforms.
  **Status:** proposed

### 2026-06-23 — Standout model candidates from CONCEPT_EQUATION_BANK.md
- **IDEA-023 · Path-signature feature node** ⭐ — signature transform of the price/volume path = a universal, ordered feature set (rough-path theory); cutting-edge in quant 2025, directly on-theme with "randomness→pattern." Feeds every downstream expert. → `path_signature`.
  **Status:** proposed
- **IDEA-024 · Hawkes-process intensity node** ⭐ — self-exciting point process λ(t)=μ+Σα e^{−β(t−tᵢ)} on trade/quote arrivals; direct microstructure edge (order-flow clustering, toxicity). → `hawkes_intensity`.
  **Status:** proposed
- **IDEA-025 · Ornstein-Uhlenbeck mean-reversion node** — dX=θ(μ−X)dt+σdW; clean, classic, direct (pairs/vol mean-reversion) and a strong honest baseline-beater candidate. → `ou_meanrevert`.
  **Status:** proposed

### 2026-06-23 — TIER-2 frontier standouts (from CONCEPT_EQUATION_BANK.md T-section)
- **IDEA-026 · RMT covariance-cleaning node (Marchenko-Pastur + Bun-Bouchaud RIE)** ⭐ — separate true correlation eigenvalues from noise; the proven, DIRECT, high-value frontier method for multi-asset risk/allocation/portfolio sizing. Honest pick: strong real-world track record (not just elegant). → `rmt_clean`.
  **Status:** proposed
- **IDEA-027 · Conformal-prediction wrapper** ⭐ — distribution-free, coverage-guaranteed prediction intervals around ANY node's output (Wasserstein-regularized for regime shift). A UNIVERSAL upgrade — makes the whole bank's outputs honestly calibrated for risk. → `conformal_interval`.
  **Status:** implemented (`pattern_brain/conformal.py`, 2026-06-23; real-BTC coverage 0.897 vs 0.90 target)
- **IDEA-028 · Online no-regret meta-combiner (multiplicative weights / FTRL)** ⭐ — a theory-backed meta-fuser that provably tracks the best expert in hindsight; a principled alternative/complement to the gated/attention combiner in the Model Neural Network north-star. → `online_no_regret`.
  **Status:** proposed

### 2026-06-23 — TIER-3 timeless-foundations standout
- **IDEA-029 · Mandelbrot stylized-facts validator** — a test harness asserting any synthetic-data/generative model (e.g. IDEA-`diffusion_synth`) reproduces the PROVEN market stylized facts: heavy/fat tails, volatility clustering, long memory (Hurst), aggregational Gaussianity. Turns Mandelbrot's proven empirical laws into an automatic gate. Complements IDEA-014 (baseline harness). → `stylized_facts_validator`.
  **Status:** proposed
- **GUARD (not a build) · Pseudoscience screen** — Gann / Fibonacci-as-prediction / Elliott Wave / astro-"Tesla 3-6-9" are SCREENED OUT (not statistically proven); if ever requested, require an extra-strict permutation-null + OOS bar. Recorded in CONCEPT_EQUATION_BANK.md ⛔ section.

### 2026-06-23 — Code-quality & invention-pipeline infra ideas (BACKFILLED per Rule 29)
_These 💡 blocks were generated across the code-skills/contract/OU work and should have been logged as produced; recorded here. Most are already built._
- **IDEA-030 · Blind-write → oracle-diff code-skills benchmark** — write a model from scratch, numerically diff vs a library oracle for an objective quality score. → `code_skills_test/`. **Status:** implemented
- **IDEA-031 · Test/oracle-first as the standing code rule** — write the equivalence/acceptance test before the code (TDD-LLM +38.6% MBPP). **Status:** implemented (Code-Generation Contract, CLAUDE.md)
- **IDEA-032 · Bake the Code-Generation Contract into the agent** — PERSONA + CLAUDE.md + ML_ENGINEERING_PRACTICES §3. **Status:** implemented
- **IDEA-033 · Reusable `oracle_compare(mine,ref,dataset,kind)` gate** — classifier/regressor/transformer/clusterer. **Status:** implemented
- **IDEA-034 · Blind-write benchmark across 5 model types** (LogReg/GaussianNB/Ridge/PCA/KMeans). **Status:** implemented (5/5 library-equivalent)
- **IDEA-035 · Hard panel — contract OFF↔ON A/B on hostile regimes** (scaling/separability/drift). **Status:** implemented (3 axes, robust)
- **IDEA-036 · Phase-4 verify gate wired into `promote_to_node`** — no promotion without an objective check. **Status:** implemented
- **IDEA-037 · Longitudinal `SCORECARD_HISTORY.md`** — dated row per run. **Status:** implemented
- **IDEA-038 · Separability axis (calibration-judged)** — labels miss overconfidence. **Status:** implemented
- **IDEA-039 · verify_block → Researcher learning signal** — blocks down-weight a feature's next pick. **Status:** implemented
- **IDEA-040 · Adversarial-regime requirement** — every model stressed on ≥1 hostile regime before KEEP. **Status:** implemented (Rule via ML_ENGINEERING_PRACTICES §F)
- **IDEA-041 · Distribution-drift axis (multi-seed)** — train-A/test-B covariate shift; reported as N/seeds. **Status:** implemented
- **IDEA-042 · Persisted per-family verify_blocks fingerprint** — cross-session memory of failing families. **Status:** implemented (`agent_state/verify_blocks.json`)
- **IDEA-043 · Dashboard "📈 Codegen Quality" + avoid-list cards** — trend + blocked families. **Status:** implemented
- **IDEA-044 · OU fair re-test (multi-step / z_dev horizon)** — its 1-step forecast can't beat AR(1) by construction; test at the reversion horizon. **Status:** approved (building this turn)
- **IDEA-045 · Next models: RMT covariance cleaning, then Hawkes** — RMT reuses panel; Hawkes needs tick/aggTrades data (sharp Rule-34 test). **Status:** proposed
- **IDEA-046 · Design-data tag per idea (Rule 34 planned, not scrambled)** — map each model to its required dataset at proposal time. **Status:** approved → being codified as Rule 35 this turn
- **IDEA-047 · Quasi-Newton solver option for the from-scratch logreg** — close the regime-B optimizer gap. **Status:** proposed
- **IDEA-048 · Distribution-drift as a standing model gate** — fold the drift A/B into the general invention gate, not just the codegen benchmark. **Status:** proposed

### 2026-06-23 — Post-OU-correction ideas
- **IDEA-049 · Horizon-aware benchmark in `harness.py`** — generalize the OU lesson: let `benchmark`/`evaluate_across_panel` sweep/auto-pick the forecast horizon (e.g. from estimated half-life) so EVERY future model is tested at its natural horizon automatically (Rule 36 made automatic, not manual). Highest leverage — prevents repeat wrong-horizon false-rejects. **Status:** proposed
- **IDEA-050 · `conftest.py` real-data guard** — a pytest fixture that points every test's data_dir/state_dir at a temp dir and asserts no test path resolves to the real `data/` — defense-in-depth so a test can NEVER again delete downloaded data. **Status:** proposed
- **IDEA-051 · Build IDEA-026 (RMT covariance cleaning) next** — design-data tag (Rule 35): the multi-asset return panel's covariance (reuses downloaded data); horizon = allocation/risk, not 1-step; test vs sample-covariance + Ledoit-Wolf baselines across the panel + an adversarial high-dim/low-sample regime. **Status:** proposed

### 2026-06-23 — RMT follow-ups
- **IDEA-052 · RMT q≥1 fallback to Ledoit-Wolf (not sample cov)** — the eval showed LW dominates at q≥1 where clipping is ill-posed; falling back to LW instead of sample makes RMT robust across ALL regimes. Cheap, removes the one weak spot. **Status:** proposed
- **IDEA-053 · Bun-Bouchaud Rotationally-Invariant Estimator (RIE)** — the optimal nonlinear eigenvalue shrinkage (vs the simpler clipping built now); the 2017 state-of-the-art. Test whether it beats clipping OOS before adopting. **Status:** proposed
- **IDEA-054 · Wire RMT into allocation/Kelly sizing** — feed the cleaned covariance into a min-variance / risk-parity / Kelly allocator so the proven risk edge actually drives position sizing (turn the estimator into a consumed capability, not a shelf tool). **Status:** proposed

### 2026-06-23 — Hawkes follow-up
- **IDEA-055 · Hawkes vs explicit RENEWAL baseline (+ cross-window history)** — the SHADOW verdict showed Hawkes beats Poisson only via the over-dispersed inter-arrival marginal, not clustering. A fair clustering test compares Hawkes to a renewal process (fit the inter-arrival distribution, no self-excitation) and carries cross-window history into the held-out intensity. Only KEEP Hawkes if it beats the renewal baseline. **Status:** proposed
- **IDEA-056 · Marked / multivariate Hawkes (buy vs sell flow)** — use the aggressor side (aggTrades `m` flag) for a 2-D mutually-exciting Hawkes (buy→sell cross-excitation = order-flow toxicity). Richer microstructure signal; only after IDEA-055 proves the univariate clustering edge is real. **Status:** proposed

### 2026-06-23 — Post-Chaos strategic ideas (4-build pattern read)
- **IDEA-057 · Pivot to MULTIPLIERS/ENABLERS over standalone forecasters** — evidence from 4 builds: standalone return-forecasters keep failing (OU ties AR1, Hawkes SHADOW, Chaos no-edge); only RMT (a risk/allocation tool) earned a clear edge. Next builds should favor features that FEED every expert — esp. IDEA-023 path-signatures (universal sequence features) — rather than more standalone predictors. **Status:** proposed
- **IDEA-058 · Consume the validated SHADOW features (don't shelve them)** — wire the chaos predictability/LLE estimate + OU half-life/z_dev into the router/regime-gate as features (they're validated diagnostics even if not forecasters), and RMT's cleaned covariance into sizing. Turns three partial wins into one consumed capability. **Status:** proposed

### 2026-06-23 — Post-signatures ideas
- **IDEA-059 · Validate signatures where they SHOULD beat raw: a long/variable-length real sequence task** — pendigits (8 fixed points) let raw-flatten win; signatures' true edge is long/irregular/variable-length multichannel paths. Test on a UCR long-series or a variable-length gesture/speech set to show the regime where signatures BEAT raw (completes the honest picture). **Status:** proposed
- ✅ **IDEA-060 · CONSUME the validated features (IMPLEMENTED 2026-06-23)** — `pattern_brain/integrated.py`: rmt_allocator (RMT→sizing, beats sample+equal-weight OOS 3/4 conditions, ~10x risk cut vs equal-weight), regime_diagnostics (chaos genuine-structure + OU half-life/z_dev), integrated_features (signatures+chaos+OU unified expert input). Oracle 4/4. ORIGINAL: — we now have 1 KEEP forecaster-tie (OU), 1 KEEP risk tool (RMT), 1 KEEP feature (signatures), 2 validated diagnostics (chaos-LLE, Hawkes branching). Wire signatures as inputs to experts + chaos-LLE/OU-halflife as router regime-gates + RMT covariance into sizing. Shift from building to integrating. **Status:** proposed

### 2026-06-23 — Post-integration ideas
- ✅ **IDEA-061 · Cost-aware backtest of the RMT allocator (IMPLEMENTED 2026-06-23)** — `tools/eval_rmt_backtest.py`. RMT CONFIRMED cost-robust: lowest net vol (0.217 vs sample 0.274 vs equal 0.663), 6.6x lower turnover (0.37 vs 2.46), gentle cost degradation (Sharpe -5.4→-6.2 vs sample -5.4→-9.2), smallest drawdown. All Sharpes negative = falling-market window + min-var minimizes RISK not return → usable as a SIZING/RISK overlay, needs a return signal for alpha. Long-only = realistic tradable form. ORIG: — the OOS variance win (~10x vs equal-weight) is real but min-variance portfolios can churn; validate net-of-turnover/fees + add weight constraints (long-only / box) before trusting RMT sizing for live capital. Makes the one real edge actually usable. **Status:** proposed
- **IDEA-062 · Register integrated_features as a bank Node** — wrap `integrated_features` in a `Node` (+@register) so the unified signatures+regime vector flows through the router/pipeline as a real expert input, not just a library function. Consume into the live pick→open path. **Status:** proposed

### 2026-06-23 — Post-cost-backtest ideas
- **IDEA-063 · Multi-regime backtest data (up + down + sideways)** — the RMT backtest's negative Sharpe is a single falling-window artifact; download longer history spanning bull/bear/chop (Binance klines paginated, or daily bars over years) so the risk-overlay verdict is regime-fair (Rule 36). Highest leverage — current verdict is window-limited. **Status:** proposed
- ✅ **IDEA-064 · Pair RMT overlay with a return signal (IMPLEMENTED 2026-06-23, PROMISING)** — `tools/eval_rmt_momentum.py`. RMT-stabilized cross-sectional momentum (w∝Σ_rmt⁻¹·μ) = FIRST positive net Sharpe of the sweep (+3.94 @10bps), robust across all lookbacks (1.6-4.6), RMT decisively beats sample-cov net of cost (+3.94 vs +0.32), passes signal-shuffle null (p=0.000). DOWNGRADED→REJECT (momentum-alpha): multi-regime OOS (IDEA-065) gave only +0.30 full-period, p=0.20 (NOT significant); the +3.94 was a regime artifact. Durable edge = RMT-risk-overlay (rmt_minvar +1.29 multi-regime). ORIG: — min-var minimizes risk, not return; combine RMT sizing with a directional signal (cross-sectional momentum, or the validated OU mean-reversion / regime gates) and test whether risk-overlay + return-signal yields positive net Sharpe. The direct follow-through on the verdict. **Status:** proposed

### 2026-06-23 — Post RMT+momentum ideas
- **IDEA-065 · Multi-regime OOS confirmation of RMT+momentum (the blocker to KEEP)** — the +3.94 net Sharpe is single-regime/in-window-significant only. Download long multi-year/multi-regime data (IDEA-063), lock a true out-of-sample holdout, re-run with per-regime permutation nulls. This is the ONE thing standing between PROMISING and a deployable KEEP. **Status:** proposed
- **IDEA-066 · Guard against my own multiple-testing** — the lookback sweep (L=90..250) was on the same data; for an honest absolute verdict, pick L on a train split and confirm on a disjoint test split (nested walk-forward), so the chosen params aren't implicitly fit to the window. **Status:** proposed

### 2026-06-23 — IDEA-065 multi-regime OOS DONE (decisive)
- ✅ **IDEA-065 · Multi-regime OOS confirmation (IMPLEMENTED 2026-06-23)** — `tools/eval_multiregime.py` + `tools/download_daily.py` (890d×42 assets, 2022 bear→2024 rally). RESULT: the RMT+momentum +3.94 did NOT survive — full-period net Sharpe +0.30, permutation p=0.20 (not significant), regime-dependent. **Downgrades IDEA-064 to REJECT (momentum alpha).** Durable multi-regime edge = pure RMT risk overlay (rmt_minvar +1.29 full-period). The rigor framework prevented a false alpha. **Status:** implemented
- ✅ **IDEA-067 · Significance + low-vol-tilt audit of rmt_minvar (IMPLEMENTED 2026-06-23)** — `tools/audit_rmt_minvar.py`. RESULT: cleaning is REAL (beats inverse-vol −0.64; time-shuffle null p=0.000 → uses genuine correlation structure, NOT just low-vol anomaly) BUT Sharpe NOT significant (block-bootstrap CI [−0.06, 2.52] includes 0; concentrated top-5=46%, low-vol tilt +0.84). RMT = validated risk MECHANISM, not (yet) proven money-maker. ORIG:'s +1.29** — the pure RMT risk overlay is now the best multi-regime performer; before believing it, run its own permutation null + check it's not just harvesting the low-volatility anomaly / a few large caps (decompose the +1.29). **Status:** proposed

### 2026-06-23 — Post-audit ideas
- **IDEA-068 · Constrain + extend the RMT min-var portfolio, then re-test significance** — the audit showed concentration (top-5=46%) + a strong low-vol tilt + a Sharpe CI that includes 0. Add box/position caps + a longer history (more years/assets) and re-run the bootstrap CI; only call it a deployable edge if the CI clears 0 with constraints. **Status:** proposed
- ✅ **IDEA-069 · RMT as a risk-reduction overlay (IMPLEMENTED 2026-06-23)** — `tools/eval_risk_overlay.py`. Vol-targeting overlay WORKS (correlations essential: diagonal→77% vol); but RMT≈sample for vol FORECASTING (slightly worse on momentum). PRECISE scoping: RMT helps in min-var WEIGHT CONSTRUCTION (inversion noise), NOT vol forecasting (wᵀΣw needs no inverse). Deploy = rmt_minvar weights + sample/RMT vol-scaling. ORIG:, measured by risk not Sharpe** — RMT's robust, significant wins are lower OOS variance + lower turnover + cost-robustness (all proven); its Sharpe is not. Deploy/track it as a variance-reduction overlay on top of an external return signal, and judge it by realized-vol reduction, not standalone Sharpe. **Status:** proposed

### 2026-06-23 — Post-overlay scoping ideas
- **IDEA-070 · Package the deployable risk overlay cleanly** — one function: rmt_minvar WEIGHTS (the inversion-sensitive step where RMT is proven) + optional vol-target scaling with sample-cov (RMT not needed there). Ship as `pattern_brain.integrated.deployable_risk_overlay(signal_weights, returns, target_vol)` with the scoping baked in. **Status:** proposed
- ✅ **IDEA-071 · RMT for Kelly/max-Sharpe sizing (IMPLEMENTED 2026-06-23)** — `tools/eval_rmt_kelly.py`. CONFIRMED (controlled synthetic): RMT recovers 85% of optimal max-Sharpe vs sample 58%; RMT advantage larger for max-Sharpe than min-var → cleaned inverse matters most where inversion-noise is worst. Real multi-regime: not significant (signal-limited, IDEA-065). Completes the RMT map: no-inverse(none)→min-var(modest)→max-Sharpe(largest). ORIG: — both invert Σ and are even more inversion-noise-sensitive than min-var; RMT should help more there than in vol-targeting. Test Σ_rmt⁻¹μ-style sizing vs sample under the full multi-regime + bootstrap-CI discipline. **Status:** proposed

### 2026-06-23 — Post-Kelly ideas
- **IDEA-072 · The binding constraint is the SIGNAL, not the sizing — make finding a regime-robust μ the explicit goal** — RMT sizing is now fully mapped/validated; every result (064/065/071) shows the blocker is a return signal that survives multi-regime OOS. Reorient model-invention to hunt a regime-robust μ (cross-sectional value/carry/quality, or a learned ensemble) under the multi-regime + permutation-null + bootstrap-CI gate, then plug into RMT max-Sharpe sizing. **Status:** proposed
- **IDEA-073 · Ledoit-Wolf vs RMT head-to-head for the inverse** — both clean the covariance; LW dominated RMT at q≥1 (IDEA-026). Run the synthetic max-Sharpe recovery test for LW too — pick the better inverse-estimator (or blend) as the deployable default. **Status:** proposed

### 2026-06-23 — TIER-3 timeless-foundations build (multi-agent, owner-directed "complete tier 3")
- ✅ **IDEA-074 · Complete TIER-3 of CONCEPT_EQUATION_BANK.md via 4 parallel file-isolated agents (IMPLEMENTED)** — built the 12 remaining TIER-3 ⬜ concepts as registered Nodes, oracle-test-first per the Code-Generation Contract, each agent owning one module (`tier3_probability/stochastic/control/pricing.py`) + test + report; orchestrator wired `nodes/__init__.py`, ran the suite (69/69 green), updated `tools/reconcile_concept_bank.py` (multi-candidate-aware) + the doc markers + `MODEL_PERFORMANCE_REPORT.md`. Nodes: bayes_update, maxent_dist, chebyshev_bound, gbm_baseline, langevin_sampler, fokker_planck, log_utility, lagrange_opt, hjb_control, momentum_kinematics, almgren_exec, bs_pricing. **Verdicts:** 9 KEEP-as-utility, 1 SHADOW (langevin = OU re-skin), 2 REJECT-as-forecaster (bayes_update, momentum_kinematics) — the project lesson held (no standalone point-forecaster beats persistence). TIER-3 now ✅ 26/26; concept-bank done 65→85/160. Parallel-agent pattern (WAVE-2 precedent) worked again for independent single-module builds. **Status:** implemented
- **IDEA-075 · Promote the 5 strongest TIER-3 utilities into consumed capabilities** — wire `lagrange_opt`/`hjb_control`/`log_utility` sizing + `chebyshev_bound`/`fokker_planck` intervals into the allocator/risk path (like RMT was consumed via integrated.py), so they're not shelf nodes. **Status:** proposed
- **IDEA-076 · Make the concept-bank tally auto-write into the doc header** — the LIVE TALLY line is hand-updated; have `reconcile_concept_bank.py` rewrite it + add a liveness check so a renamed/removed node can't leave a stale ✅. **Status:** proposed

### 2026-06-23 — TIER-1 directly-buildable cluster, ONE AT A TIME (owner-directed)
- ✅ **IDEA-077 · `evt_tail_risk` (Extreme Value Theory tail tool) — BUILT, KEEP-as-utility** — first node of the TIER-1 directly-buildable cluster, built solo (not parallel agents) oracle-test-first per the Code-Generation Contract. Hill tail index + GPD Peaks-Over-Threshold → tail_index/ξ/VaR₉₉/ES₉₉, emits `signal`. Oracle 7/7 (recovers Pareto α, GPD ξ, VaR monotone). Real crypto: recovers the ~2.5 cubic-law fat-tail index, separates heavy (crypto/t3) from thin (Gaussian), realistic 1–3% VaR. A self-inflicted eval bug (flattened OHLCV incl. volume) was caught + fixed before reporting (Rule 14/21). Risk tool, judged by correctness like RMT/conformal — NOT a forecaster. `pattern_brain/nodes/tier1_classics.py`; report `MODEL_REPORT_tier1_evt.md`. **Status:** implemented
- **Remaining directly-buildable order (next):** `jump_diffusion` (Merton) → `kyle_impact`/`ofi`/`vpin` (microstructure, design-data = `data/ticks/`) → `hrp_alloc`/`cvar_opt`/`merton_alloc` (allocation) → `koopman_dmd` (forecaster) → `fuzzy_ts`/`grey_gm11` → `infogeom_distance`/`diffusion_map`. Excluded (heavy/research): quantum, foundation-TS, LLM-sentiment, neuro-symbolic.

### 2026-06-24 — TIER-1 batch 2 (johansen_coint, har_rv, copula_dependence, bocpd_break)
- ✅ **IDEA-078 · TIER-1 batch 2 BUILT (4 nodes, owner-directed ~3–4/turn)** — oracle-test-first per the Code-Generation Contract, light numpy/scipy only (own ADF/cointegration + BOCPD, no statsmodels). 18/18 tier-1 oracles pass; full bank still conforms. **Verdicts:** `har_rv` = **KEEP** (Corsi HAR realized-vol forecaster — beats persistence on **10/10** panel symbols, skill +0.20…+0.48; the FIRST TIER-1 node to beat a baseline, because it forecasts *volatility*, which is predictable, not *returns*). `johansen_coint` / `copula_dependence` / `bocpd_break` = **KEEP-as-utility** (correctness-validated risk/regime/stat tools): cointegration is oracle-correct but crypto pairs are honestly near-unit-root here (1/5 spreads, 0/5 pairs clear the strict bar); copula recovers τ=0.71 / tail-dep≈0.72 on ETH/BTC vs ~0.06 independent; BOCPD localizes a real spliced break to ±5. Build wall (BOCPD flat P(r=0)≡hazard) solved via run-length-posterior collapse → IMP-002. Files: `pattern_brain/nodes/tier1_classics.py`, report `MODEL_REPORT_tier1_batch2.md`. **Status:** implemented

### 2026-06-24 — 💡 Ideas block (post-batch-2)
- **IDEA-079 · Build a focused VOLATILITY-forecasting cluster next (HAR-RV is the breakthrough)** — har_rv is the first TIER-1 node to genuinely beat a baseline, and it does so by targeting *vol*, not *returns*. Capitalize: add HAR-RV's natural family — **HARQ** (realized-quarticity-adjusted HAR), **log-HAR** / **HAR-RV-J** (jump component), and feed `egarch_vol` + `evt_tail_risk` + `har_rv` into one **consumed volatility layer** (vol forecast → position sizing / risk gating). This is where the project actually has edge; lean into it instead of more return forecasters. **Status:** proposed
- **IDEA-080 · Pivot `johansen_coint` to a corpus with genuinely cointegrated assets (Rule 36 breadth)** — the node is oracle-correct but crypto large-caps aren't reliably cointegrated in the current 1000-bar window, so it can't be fairly KEEP/REJECT'd as a *signal*. Download a known-cointegrated set (e.g. an ETF/constituent pair, dual-listed shares, or a stablecoin basket) per Rule 33 and re-test the stat-arb spread signal there before judging it as more than a utility. **Status:** proposed

### 2026-06-24 — TIER-1 batch 3: 18 nodes via 5 parallel agents → cluster COMPLETE
- ✅ **IDEA-081 · TIER-1 directly-buildable cluster COMPLETED (18 nodes, 5 parallel file-isolated agents, owner-directed)** — each agent owned one module+test+report (microstructure / allocation / stochastic / dynamical / soft-computing), oracle-test-first per the Code-Generation Contract, light numpy/scipy/sklearn only. Orchestrator pre-fetched microstructure trade data (price+qty+side → `data/ticks_micro/`, Rule 35), wired `nodes/__init__.py`, ran the suite (110 passed, all conform), updated reconcile/doc/perf-report. **Forecaster KEEPs: `heston_vol`** (variance beats persistence 10/10) joins har_rv — both VOLATILITY. **KEEP-as-utility (11):** ofi, vpin, kyle_impact, hrp_alloc (OOS var ⅓ of EW), cvar_opt (−64% tail loss), merton_alloc, mpc_position (5× less turnover), jump_diffusion, sv_particle, percolation_risk, infogeom_distance, diffusion_map, grey_gm11. **SHADOW/REJECT on returns (project lesson, measured against the ZERO baseline not just persistence):** koopman_dmd, bsts_forecast, fuzzy_ts, anfis. Bank 80→96 Node types; concept-bank 106/160 buildable done (~66%). Reports: `MODEL_REPORT_tier1_{microstructure,allocation,stochastic,dynamical,softcomputing}.md`. **Status:** implemented
- **Also updated `ML_ENGINEERING_PRACTICES.md` §G (owner-mandated):** "Test on the DESIGN-INPUT data the model was built to consume" + a design-input data map (returns/RV/spread/order-flow/covariance/event-times/cross-domain) — now a binding part of the engineering standard.

### 2026-06-24 — 💡 Ideas block (post-batch-3)
- ✅ **IDEA-082 · Rule-36 DUAL-DOMAIN verification pass (IMPLEMENTED 2026-06-24)** — `tools/eval_crossdomain_tier1.py` + `data/crossdomain/` (sunspots, airline, temps, UCI diabetes, UCI digits) + `MODEL_REPORT_crossdomain_tier1.md`. Each general-purpose node re-tested on a real NON-trading dataset of the type its theory consumes. **CONFIRMED on a 2nd domain:** koopman_dmd (sunspots BEATS), heston_vol+har_rv (temps RV BEATS), sv_particle (+0.57), infogeom_distance, jump_diffusion, percolation_risk, diffusion_map. **CAUGHT over-generous verdicts (the rule's payoff): anfis→SHADOW** (UCI diabetes RMSE 174 vs linear 53 — fails real multivariate regression), **grey_gm11** scope-narrowed to smooth/monotone, **bsts_forecast→SHADOW** (trend filter, not forecaster). Microstructure exempt (trading-intrinsic); merton/mpc/cvar = oracle-validated domain-agnostic math. Also fixed a harness bug (Rule 14): a har_rv "blow-up" was feeding pre-computed RV to a node that wants a level — har_rv itself unchanged, confirmed on both domains. **Status:** implemented
- **IDEA-083 · Consume the two VOLATILITY KEEPs (har_rv + heston_vol) into one vol layer** — these are the only genuine forecaster wins; fuse them with egarch_vol + evt_tail_risk into a single consumed volatility-forecast → sizing/risk-gating capability (mirrors how RMT was consumed), rather than leaving them as shelf nodes. **Status:** proposed

### 2026-06-24 — 💡 Ideas block (Five Stages of Learning curriculum)
- **IDEA-084 · Stage-1 Self-Supervised Universal Market Embedding (the keystone)** — pretrain the shared encoder (gap #1 / P1) on ALL unlabeled history via masked-reconstruction + contrastive (TS2Vec / PatchTST / TimeDART / diffusion-SSL). Highest-leverage stage: defeats the limited-trades constraint by learning from every bar, and produces the latent every other neuron reads. Acceptance: the SSL embedding must beat raw-feature inputs OOS (Rule 30/36) before any downstream stage trusts it. Design-data tag (Rule 35): the full unlabeled bar history already on the VPS; horizon = representation quality (probe with a frozen linear head), not 1-step PnL. **Status:** proposed
- **IDEA-085 · Stage-5 RL decision-maker — attention soft-router + lifecycle policy** — replace/extend the `gated` router (gap #3 / P2) with a learned softmax-attention top-k policy trained by policy-gradient/Gumbel-Softmax, plus an RL trade-lifecycle policy (size/SL/TP/exit) on cumulative reward. Stabilize with 2025 MoE-RL methods (RSPO router-shift, routing-replay, IS-correction; T2MIR token+task MoE-for-RL). Hardest/highest-variance → staged LAST; acceptance: must beat today's `gated` router OOS. **Status:** proposed
- **IDEA-086 · Stage-4 Semi-Supervised label-expander** — confidence-thresholded pseudo-labeling + consistency regularization (Mean-Teacher/FixMatch) to stretch the few realized-trade labels onto the vast unlabeled bars; feeds the supervised expert heads more effective data. Explicit risk gate: pseudo-label snowballing → hard confidence threshold + OOS validation, must beat supervised-only OOS. **Status:** proposed
- **IDEA-087 · Continual-learning LOOP wrapper around the 5 stages** — make the curriculum a cycle, not a one-shot line: new unlabeled data re-enters Stage 1 (re-pretrain) under drift detection (ADWIN/Page-Hinkley) + forgetting guards (EWC/replay), so the network keeps learning without catastrophic forgetting. Ties the 5 stages into a standing pipeline. **Status:** proposed
