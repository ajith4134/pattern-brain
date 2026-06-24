# Concept & Equation Bank — raw material for inventing ML models (Rule 30)
_Created 2026-06-23. Screened for trading relevance (direct/indirect). LIVING doc — v1 is broad coverage of
PhD/cutting-edge concepts across the 5 axes (mainly Axis D); each category can be drilled deeper on request._
_Legend: ⭐ = cutting-edge 2024–26 · [D]=direct trading use · [I]=indirect (feature/regime/risk) · → candidate node._
_Honesty: this is NOT literally "every equation ever" (that's unbounded) — it's a dense, expert-screened map of
what's usable. Each entry must still beat baselines OOS (Rule 30) before it earns a place in the bank._

**📊 BUILD STATUS COLUMN (added 2026-06-23, auto-maintained by `tools/reconcile_concept_bank.py`):**
each concept now carries a build marker so this menu is a LIVE tracker, not a static list. Re-run the
tool after adding nodes to refresh it.
- **✅ built (Node)** — registered in the bank, callable now.
- **🔬 built (module)** — the capability exists as a utility module (e.g. `rmt.py`, `signatures.py`), not yet a registered `Node`.
- **⬜ NOT built** — an open model candidate (this is the "what's left to implement" list).
- **▫️ foundational** — an underpinning concept / metric, not meant to be its own node.

**LIVE TALLY (193 concepts): 55 ✅ built-Node · 10 🔬 built-module · 94 ⬜ NOT built · 34 ▫️ foundational.**
→ **65 of 159 buildable concepts are done (~41%); 94 remain to implement** (most are TIER-2 ⭐ frontier — gate hardest, Rule 30).

> Axes A/B/C/E (paradigm/task/architecture/property) are HOW we wrap these; Axis D (below) is WHAT the model
> knows. See ML_MODEL_TAXONOMY.md. Each concept becomes a `Node` (see ML_ENGINEERING_PRACTICES.md).

---

## D1 — Statistical / Econometric
- ✅ **ARIMA / SARIMA / ARFIMA** — autoregressive + (fractional) integration + MA; ARFIMA captures long memory. [D] → `arfima_forecast`
- ✅ **VAR / VECM** — vector autoregression + error correction for cointegrated series. [D] cross-asset → `vecm_pairs`
- ⬜ **Cointegration (Engle-Granger, Johansen)** — long-run equilibrium between non-stationary series. [D] pairs/stat-arb → `johansen_coint`
- ✅ **GARCH family (GARCH, EGARCH, GJR, FIGARCH, GARCH-MIDAS)** — conditional variance; asymmetry/leverage; long-memory vol. [D] vol/risk → `egarch_vol` (have GARCH/EGARCH)
- ⬜ **Stochastic Volatility (SV)** — latent vol process (vs GARCH's deterministic). [D] → `sv_particle`
- ✅ **Markov Regime-Switching (Hamilton)** — params switch by hidden regime. [D] regime → `markov_switching`
- ✅ **State-space / Kalman / particle filters** — latent state + noisy obs; ⭐ differentiable Kalman exists. [D] → `kalman_state` (have)
- ⬜ **Copulas (Gaussian, t, Clayton, Gumbel, vine)** — dependency structure separate from marginals; tail co-movement. [I] portfolio/risk → `copula_dependence`
- ⬜ **Extreme Value Theory (GPD, GEV, Hill estimator)** — tail distribution, max drawdown, VaR/ES. [D] risk → `evt_tail_risk`
- ✅ **Quantile regression / CAViaR** — predict conditional quantiles (not just mean); dynamic VaR. [D] → `quantile_band`
- ✅ **Hawkes processes (self-exciting point process)** ⭐ — event intensity λ(t)=μ+Σα e^{-β(t-tᵢ)}; trade/order clustering. [D] microstructure → `hawkes_intensity`
- ✅ **Long memory / Hurst (R/S, DFA)** — persistence vs mean-reversion. [I] regime → `hurst_dfa` (see D7)
- ⬜ **Change-point: CUSUM, Bayesian Online ChangePoint (BOCPD)** — structural breaks in real time. [D] regime → `bocpd_break`
- ⬜ **Realized volatility / HAR-RV** — multi-scale realized-vol regression. [D] vol → `har_rv`

## D2 — Probabilistic / Bayesian
- ⬜ **Bayes' rule / Bayesian updating** — posterior ∝ likelihood × prior. [D] belief fusion → `bayes_update`
- ✅ **Gaussian Processes (GP)** — nonparametric distribution over functions + uncertainty. [D] forecast w/ error bars → `gp_forecast`
- ✅ **Hidden Markov / Hidden Semi-Markov (HMM/HSMM)** ⭐ differentiable HMM exists — latent discrete regimes. [D] regime → `hmm_regime` (have)
- ✅ **Dirichlet Process / Bayesian nonparametrics** — infinite mixtures, auto-cluster count. [I] regime discovery → `dp_mixture`
- ✅ **Variational Inference / ELBO** — approximate posteriors at scale; basis of VAEs. [I] embedding → `vae_latent`
- ⬜ **Bayesian Structural Time Series (BSTS)** — trend+seasonal+regression, causal impact. [D] → `bsts_forecast`
- ✅ **Particle filters / Sequential Monte Carlo** — nonlinear/non-Gaussian state tracking. [D] → `smc_state`

## D3 — Information-Theoretic
- ✅ **Shannon entropy H=−Σp log p** — randomness/uncertainty of returns. [I] regime → `entropy_regime` (have infotheory)
- ✅ **Mutual Information / conditional MI** — nonlinear dependence between series. [I] feature select → `mi_screen` (have)
- ✅ **Transfer Entropy** — directed information flow X→Y (model-free causality). [D] lead-lag → `transfer_entropy` (have)
- ✅ **Kolmogorov complexity / MDL (compression)** — compressibility = pattern. [I] → `mdl_compress` (have)
- ✅ **Permutation / sample / approximate entropy** — complexity of ordinal patterns; cheap regime signal. [I] → `perm_entropy`
- ▫️ **Information Bottleneck max I(T;Y)−βI(T;X)** ⭐ — compress inputs keeping only predictive info. [I] fusion objective → (IDEA-003)
- ▫️ **Fisher information / Jeffreys** — sensitivity/identifiability; bridges to info geometry (D6). [I]

## D4 — Physics-based / Stochastic Processes (the "physics models")
- ⬜ **Geometric Brownian Motion dS=μS dt+σS dW** — baseline price SDE. [D] sim/baseline → `gbm_baseline`
- ✅ **Ornstein-Uhlenbeck dX=θ(μ−X)dt+σdW** — mean reversion (pairs, vol). [D] mean-revert → `ou_meanrevert`
- ⬜ **Jump-diffusion (Merton), Kou** — Brownian + Poisson jumps; fat tails/gaps. [D] → `jump_diffusion`
- ⬜ **Heston stochastic-vol model** — vol is its own mean-reverting SDE. [D] vol surface → `heston_vol`
- ✅ **Rough volatility (rough Bergomi, fractional)** ⭐ — vol driven by fractional BM, H<0.5; matches data. [D] → `rough_vol`
- ⬜ **Fokker-Planck / Kolmogorov forward** — evolution of the probability density of an SDE. [I] distribution forecast → `fokker_planck`
- ⬜ **Langevin dynamics** — force + noise; sampling & dynamics. [I] → `langevin_sampler`
- ⬜ **Ising / mean-field / spin-glass** ⭐ econophysics — trader herding → magnetization; bubbles/crashes. [I] sentiment/herding → `ising_herding`
- ✅ **Self-Organized Criticality / power laws / avalanches** — heavy-tailed crash sizes. [I] tail/crash → `soc_avalanche`
- ⬜ **Percolation / network contagion** — connectivity phase transition; systemic risk. [I] → `percolation_risk`
- ⬜ **Superstatistics** ⭐ — superposition of Gaussian processes w/ fluctuating β → fat tails/long memory. [I] → `superstat_vol`
- ▫️ **Fluctuation-Dissipation / response functions** — how markets respond to shocks. [I]

## D5 — Thermodynamic / Energy-based
- ⬜ **Maximum Entropy (MaxEnt) p∝exp(−Σλᵢfᵢ)** — least-biased distribution given constraints. [D] distribution est → `maxent_dist`
- ✅ **Boltzmann / Restricted Boltzmann Machines** — energy-based generative; learn joint structure. [I] embedding → `rbm_energy`
- ⬜ **Free-Energy Principle / Active Inference** ⭐ — minimize surprise/prediction error. [I] meta-control → `active_inference`
- ▫️ **Partition function / free energy** — normalizer; phase behavior of the market "ensemble." [I]

## D6 — Geometric / Topological / Algebraic
- ✅ **Topological Data Analysis / Persistent Homology** ⭐ — shape of point clouds; crash precursors (loops/voids). [I] crash signal → `tda_persistence`
- ⬜ **Information Geometry (Fisher-Rao metric)** — distance between probability models on a manifold. [I] regime distance → `infogeom_distance`
- ⬜ **Optimal Transport / Wasserstein distance** ⭐ — minimal cost to morph one distribution into another. [I] regime shift/dist → `wasserstein_shift`
- ⬜ **Manifold learning (diffusion maps, UMAP, Isomap, Laplacian eigenmaps)** — nonlinear low-dim structure. [I] embedding → `diffusion_map`
- 🔬 **Path Signatures / rough paths** ⭐ — signature transform = ordered moments of a path; universal feature set; signature kernels. [D] sequence feature → `path_signature`
- ⬜ **Tensor networks (MPS/TT)** ⭐ — compress high-dim correlations (quantum-inspired). [I] → `tensor_network` (see D15)

## D7 — Dynamical Systems / Chaos / Fractal
- ✅ **Lyapunov exponent λ** — sensitivity to initial conditions; chaos vs noise. [I] regime → `lyapunov_chaos` (IDEA-001/005)
- ✅ **Takens embedding / attractor reconstruction** — rebuild state space from one series. [I] → `takens_embed`
- ⬜ **Koopman operator / DMD / EDMD** ⭐ — lift nonlinear dynamics to linear evolution; forecast there. [D] → `koopman_dmd` (IDEA-013)
- ✅ **Reservoir computing / Echo State Networks** — fixed random recurrent net + linear readout; great on chaos. [D] → `reservoir_esn`
- ✅ **Detrended Fluctuation Analysis (DFA) / Hurst** — long-range correlations, persistence. [I] → `hurst_dfa`
- ✅ **Multifractal (MF-DFA, MMAR)** ⭐ — spectrum of scaling exponents; vol clustering. [I] → `multifractal`
- ✅ **Recurrence Quantification Analysis (RQA)** — determinism/laminarity from recurrence plots. [I] → `rqa_determinism`
- ▫️ **0-1 test for chaos** — binary chaos detector. [I]

## D8 — Causal Inference
- 🔬 **Granger causality** — does X's past improve Y's forecast. [D] lead-lag → `granger_cause`
- ⬜ **Convergent Cross Mapping (CCM)** ⭐ — nonlinear causality in dynamical/coupled systems. [D] → `ccm_cause`
- ⬜ **Structural Causal Models / do-calculus** — interventions & counterfactuals. [I] robustness → `scm_intervene`
- ⬜ **PCMCI / causal discovery for time series** ⭐ — graph of time-lagged causes. [I] feature graph → `pcmci_graph`
- ▫️ **Double ML / instrumental variables** — debiased causal effect estimates. [I]

## D9 — Game-Theoretic
- ⬜ **Kyle model / Glosten-Milgrom** — informed vs noise traders; price impact λ. [D] microstructure → `kyle_impact`
- ⬜ **Nash / minimax / adversarial** ⭐ — robust strategy vs worst-case; GAN-style. [I] robustness → `adversarial_robust`
- ⬜ **Mean-field games** ⭐ — equilibrium among many small agents. [I] crowding → `mfg_crowding`

## D10 — Control-Theoretic
- ⬜ **Stochastic optimal control / HJB** — optimal action under dynamics+noise. [D] sizing/exit → `hjb_control`
- ⬜ **Merton portfolio problem** — optimal consumption/allocation. [D] sizing → `merton_alloc`
- ⬜ **Almgren-Chriss optimal execution** — minimize impact+risk of liquidation. [D] execution → `almgren_exec`
- ⬜ **Model Predictive Control (MPC) / LQG** — rolling optimization under a model. [D] position control → `mpc_position`
- ✅ **RL for execution/sizing (PPO, SAC, DQN)** — learned policy from reward. [D] → `rl_policy` (have rl.py)

## D11 — Optimization / Operations Research
- 🔬 **Markowitz mean-variance / efficient frontier** — risk-return tradeoff. [D] allocation → `markowitz`
- 🔬 **Kelly criterion / fractional Kelly f*=μ/σ²** — growth-optimal bet sizing. [D] sizing → `kelly_size`
- ⬜ **Risk parity / HRP (hierarchical risk parity)** ⭐ — allocate by risk contribution; ML-clustered. [D] → `hrp_alloc`
- ⬜ **CVaR / robust optimization** — optimize tail risk. [D] → `cvar_opt`
- 🔬 **Bayesian optimization** — sample-efficient hyperparam/strategy search. [I] tuning → `bayes_opt`
- ✅ **Metaheuristics (GA, PSO, ACO, simulated annealing)** — global search of strategy space. [I] → (have evolution.py)

## D12 — Bio-inspired (neural + evolutionary + swarm)
- ✅ **Deep seq models: LSTM, GRU, TCN, Transformer** — nonlinear sequence forecasting. [D] → `deep_seq` (have deep.py)
- ✅ **PatchTST, N-BEATS, N-HiTS, TSMixer, iTransformer** ⭐ — SOTA TS-specific deep nets. [D] → `patchtst`
- ✅ **Mamba / state-space (S4, S-Mamba)** ⭐ — linear-time long-context sequence model. [D] → `mamba_ssm`
- ⬜ **Foundation TS models: TimesFM, Chronos, Moirai, Time-MoE; FinCast, Kronos, LiT (order-book)** ⭐ — pretrained zero-shot forecasters. [D] (heavy) → `tsfm_zeroshot`
- ▫️ **Genetic programming / NEAT** — evolve formulas/architectures. [I] discovery → (eqgen/evolution)
- ▫️ **Swarm (PSO/ACO)** — population search. [I]

## D13 — Symbolic / Logic
- ✅ **Symbolic Regression (PySR, AI Feynman, SINDy)** ⭐ — discover closed-form equations from data. [D] interpretable signal → `symbolic_regression` (have eqgen)
- ▫️ **Genetic programming rule induction / decision lists** — human-readable rules. [I]

## D14 — Soft Computing (uncertainty/imprecision)
- ⬜ **Fuzzy time series / fuzzy logic** — linguistic rules over fuzzy sets. [I] regime rules → `fuzzy_ts`
- ⬜ **Grey systems / GM(1,1)** — forecasting from very few points. [I] data-poor → `grey_gm11`
- ▫️ **Rough sets** — decision rules under indiscernibility. [I]
- ⬜ **Neuro-fuzzy (ANFIS)** — learnable fuzzy rules. [I] → `anfis`

## D15 — Quantum / Quantum-inspired
- ⬜ **Quantum kernels / variational quantum circuits** — feature maps in Hilbert space (simulator). [I] research → `quantum_kernel` (bench in trading bot)
- ⬜ **Tensor networks (MPS) for sequences** ⭐ — quantum-inspired, interpretable, efficient. [I] → `tensor_network`
- ⬜ **Quantum annealing for portfolio optimization** — QUBO formulation of allocation. [I] → `qubo_portfolio`

## D16 — Linguistic / Neuro-symbolic / NLP
- ⬜ **LLM sentiment / news embeddings** — text → signal (you already use cloud LLMs). [D] sentiment → `news_sentiment`
- ⬜ **Neuro-symbolic (LLM + verifier)** ⭐ — reasoning + constraints. [I] meta → `neuro_symbolic`

## D-Signal — Signal Processing / Spectral (cross-cuts D4/D1)
- ✅ **Fourier / FFT, spectral density** — hidden cycles. [I] → `fft_cycles` (have signal.py)
- ✅ **Wavelets (CWT/DWT) / wavelet coherence** — time-frequency localization. [I] → `wavelet`
- ✅ **Hilbert-Huang / EMD / EEMD** — adaptive decomposition of nonstationary signals. [I] → `emd` (have decomposition)
- ✅ **Singular Spectrum Analysis (SSA)** — trend/oscillation extraction via embedding+SVD. [I] → `ssa` (have)
- ✅ **Kalman / Wiener / matched filters** — optimal denoising/detection. [I] → (have)
- ✅ **Hilbert transform / instantaneous frequency & phase** — cycle phase for timing. [I] → `hilbert_phase`

## MICROSTRUCTURE / FINANCE-SPECIFIC (high-value, applies the above to order flow)
- ⬜ **Order Flow Imbalance (OFI)** — net buy/sell pressure from L1/L2. [D] → `ofi`
- ⬜ **VPIN (volume-synchronized prob. of informed trading)** — toxicity/flow-informedness. [D] → `vpin`
- ⬜ **Avellaneda-Stoikov market making** — optimal quotes around reservation price. [D] → `as_market_make`
- ▫️ **Hasbrouck information share / price discovery** — who leads price. [I]
- ⬜ **Limit-order-book models (queue-reactive, Hawkes-LOB)** ⭐ — LOB dynamics. [D] → `lob_hawkes`
- ⬜ **Rough Hawkes-Heston** ⭐ — jumps + rough vol + self-excitation unified. [D] → `rough_hawkes_heston`

---

---

# ===== TIER-2 — ULTRA-ADVANCED / RESEARCH FRONTIER (above-PhD) =====
_Added 2026-06-23 (2nd sweep). Genuinely frontier — most are active 2024–26 research. Higher payoff, higher
build cost, and (Rule 30) still must beat simple baselines OOS. Clustered by theme; ⭐ all frontier._

## T1 — Rough paths & signatures (deep)
- 🔬 **Signature transform / log-signature** — universal ordered-moment feature of a path; basis of rough-path ML. [D] feature → `path_signature` (IDEA-023)
- ⬜ **Signature kernels (PDE-solved) & signature-MMD** — kernel between paths; non-adversarial Neural-SDE training. [D] → `sig_kernel`
- ▫️ **Expected signature / signature cumulants** — moments of a stochastic process in signature space. [I]
- ⬜ **Neural CDE / Neural RDE (rough)** — continuous-time deep nets driven by the data path; SOTA for irregular/tick series. [D] → `neural_cde`

## T2 — Continuous-time deep learning & generative SDEs
- ⬜ **Neural ODE / Neural SDE** — NN-parameterized drift+diffusion; learn dynamics. [D] → `neural_sde`
- ⬜ **Deep BSDE / 2BSDE solvers** — solve high-dim HJB / pricing / optimal control via backward SDEs. [D] sizing/exit → `deep_bsde`
- ⬜ **Score-based diffusion generative models for financial series** — generate synthetic paths reproducing fat tails/vol-clustering/leverage. [D] scenario gen / augmentation / stress test → `diffusion_synth`
- ⬜ **Schrödinger Bridge (entropic OT on path space)** — minimal-entropy interpolation between distributions; robust TS synthesis. [I] → `schrodinger_bridge`

## T3 — Random Matrix Theory & Free Probability (HIGH-VALUE, direct)
- ✅ **Marchenko-Pastur law** — eigenvalue spectrum of pure-noise covariance; separates signal eigenvalues from noise. [D] covariance cleaning → `rmt_clean`
- 🔬 **Ledoit-Péché / Bun-Bouchaud rotationally-invariant estimator (RIE)** — optimal nonlinear shrinkage of the covariance. [D] portfolio/risk → `rie_covariance`
- ⬜ **Tracy-Widom distribution** — largest-eigenvalue fluctuations; detect a real factor vs noise. [I] → `tracy_widom_test`
- ▫️ **Free probability (Voiculescu)** — algebra of large random matrices; sums/products of covariances. [I]

## T4 — Stochastic Portfolio Theory & relative arbitrage
- ⬜ **Fernholz functional portfolio generation** — outperform the market from diversity/entropy functions (relative arbitrage). [D] allocation → `spt_diversity`
- ⬜ **Market diversity / entropy-weighted portfolios** — rank-based structural alpha. [D] → `entropy_portfolio`

## T5 — Large deviations & extreme events
- ⬜ **Large Deviation Principle / rate functions** — probability & most-likely path of rare moves. [I] crash/tail → `ldp_rate`
- ⬜ **Freidlin-Wentzell / instanton (optimal path to a crash)** — the most probable trajectory of a rare event. [I] → `instanton_crash`

## T6 — Robust & calibrated decision-making
- ⬜ **Wasserstein Distributionally Robust Optimization (DRO)** ⭐ — optimize worst-case over a Wasserstein ball of distributions; robust to regime shift. [D] robust sizing/alloc → `wasserstein_dro`
- ✅ **Conformal prediction / conformalized quantile regression** ⭐ — distribution-free prediction intervals with coverage guarantees; Wasserstein-regularized variants for shift. [D] calibrated risk bands → `conformal_interval`
- ⬜ **Distributional RL (C51, QR-DQN, IQN)** — learn the full return distribution, not just the mean. [D] sizing/exit → `distributional_rl`
- ⬜ **Entropic OT / Sinkhorn** — fast regularized optimal transport; distances between market states. [I] → `sinkhorn_dist`

## T7 — Statistical mechanics of learning (theory → better models)
- ⬜ **Neural Tangent Kernel (NTK)** — wide nets ≈ kernel regression; predicts generalization. [I] model design → `ntk_analysis`
- ▫️ **Replica method / spin-glass loss landscape** — typical-case generalization & capacity. [I]
- ⬜ **Renormalization Group ↔ deep learning / multiscale** — coarse-graining = depth; multiscale features. [I] → `rg_multiscale`
- ▫️ **Neural scaling laws** — performance vs data/params; budget allocation. [I]

## T8 — Advanced information theory
- ⬜ **Partial Information Decomposition (PID): unique / redundant / synergistic** ⭐ — decompose how multiple signals jointly inform the target (synergy = combos that only work together). [I] feature/expert synergy → `pid_synergy`
- ⬜ **Predictive information / predictive rate-distortion** — the part of the past that predicts the future. [I] → `predictive_info`
- ▫️ **O-information / integrated information** — higher-order (beyond-pairwise) dependencies. [I]

## T9 — Point processes (deep) — order flow
- ⬜ **Neural / Transformer Hawkes, neural temporal point processes** ⭐ — learn event-intensity from data (nonlinear, marked). [D] microstructure → `neural_hawkes`
- ⬜ **Nonlinear / multivariate marked Hawkes** — cross-excitation between event types. [D] → `marked_hawkes`

## T10 — Optimal stopping / free-boundary (exit timing)
- ⬜ **Snell envelope / optimal stopping** — when to exit to maximize expected payoff. [D] exit → `optimal_stopping`
- ⬜ **Deep optimal stopping (Becker-Cheridito-Jentzen)** ⭐ — NN solves high-dim stopping (American-option style). [D] exit → `deep_stopping`
- ▫️ **Reflected BSDE / free-boundary PDE** — math behind early-exercise/exit. [I]

## T11 — Geometric / topological frontier
- ⬜ **Zigzag persistence / persistence landscapes / Euler characteristic transform** ⭐ — evolving topology of the market; vectorized for ML. [I] crash precursor → `tda_zigzag`
- ⬜ **Sheaf theory / topological signal processing** ⭐ — consistency of local data over a network. [I] cross-asset → `sheaf_consistency`
- ▫️ **Geometric deep learning / equivariant nets (Bronstein)** — symmetry-aware models. [I]
- ⬜ **Hyperbolic embeddings** — represent hierarchy (sectors/assets) with low distortion. [I] → `hyperbolic_embed`
- ▫️ **Wasserstein gradient flows / JKO scheme** — distributions evolving by steepest descent. [I]

## T12 — Stochastic filtering & control frontier
- ⬜ **Zakai / Kushner-Stratonovich nonlinear filtering SPDEs** — optimal state estimate for nonlinear/non-Gaussian. [D] latent state → `nonlinear_filter`
- ⬜ **McKean-Vlasov / mean-field control & games** ⭐ — optimal control among many interacting agents (crowding). [I] → `meanfield_control`
- ⬜ **HJB-Isaacs / viscosity solutions / robust control** — worst-case optimal control. [D] robust exit → `hjb_isaacs`

## T13 — Quantum frontier (mostly research/simulator)
- ⬜ **Quantum amplitude estimation** — quadratic speedup for Monte-Carlo pricing/VaR. [I] research → `q_amplitude_est`
- ▫️ **Quantum signal processing / quantum walks** — frontier transforms. [I]
- ⬜ **Quantum path signatures / quantum reservoir computing** ⭐ — quantum versions of T1/D7. [I] → `q_reservoir`

## T14 — Online learning / regret minimization (model-free, robust)
- ⬜ **Cover's Universal Portfolio** — provably competitive with the best constant-rebalanced portfolio in hindsight. [D] allocation → `universal_portfolio`
- 🔬 **Online Convex Optimization / Follow-the-Regularized-Leader / Multiplicative Weights** — no-regret weighting of experts (a principled meta-combiner!). [D] meta-fusion → `online_no_regret`
- ✅ **Bandits / Thompson sampling** — explore-exploit allocation across strategies/experts. [D] → `bandit_alloc`

## T15 — Causal discovery frontier
- ⬜ **PCMCI+ / LPCMCI** ⭐ — time-lagged causal graph discovery with hidden confounders. [I] feature graph → `pcmci_plus`
- ⬜ **Causal representation learning / Invariant Risk Minimization** ⭐ — learn features that are stable across regimes (anti-overfit). [D] robustness → `irm_invariant`

---

# ===== TIER-3 — TIMELESS FOUNDATIONS (1650s → 1980s), PROVEN & HIGH-IMPACT =====
_Added 2026-06-23 (3rd sweep — "Newton/Tesla era to now, proven"). These are the bedrock; many are MORE robust
than the frontier and several already underpin the bank. Organized by era/originator. [D]/[I] + candidate node._

## E1 — Probability & calculus foundations (1650–1800)
- ▫️ **Pascal/Fermat (1654) — expected value** — the EV of a trade. [D] → core sizing logic
- ⬜ **D. Bernoulli (1738) — expected utility / log utility** — risk-averse sizing; ancestor of Kelly. [D] → `log_utility`
- ⬜ **Bayes (1763) — posterior ∝ likelihood × prior** — belief updating. [D] → `bayes_update`
- ⬜ **Newton/Leibniz — calculus; rate-of-change & inertia** — momentum & its derivatives (velocity/acceleration of price). [D] → `momentum_kinematics`
- ▫️ **Euler-Lagrange / calculus of variations** — optimize a functional → optimal paths/controls. [I] → control nodes
- ⬜ **Lagrange multipliers / Hamiltonian mechanics** — constrained optimization (the math under Markowitz); Hamiltonian dynamics → Hamiltonian NNs. [D] allocation → `lagrange_opt`
- ▫️ **Laplace — transform, de Moivre-Laplace CLT, Bayesian inference** — why aggregates go Gaussian (and when they don't). [I]
- ▫️ **Gauss/Legendre — normal distribution, least squares, Gauss-Markov** — the workhorse estimator. [D] → regression nodes

## E2 — Fields, heat, waves, early stats (1800–1870)
- ✅ **Fourier (1807) — Fourier series/transform, heat equation** — hidden cycles; the same diffusion PDE Bachelier later used for options. [D] → `fft_cycles` (have)
- ✅ **Hooke's law / damped harmonic oscillator** — restoring force = mean reversion; **Ornstein-Uhlenbeck is a noisy damped oscillator**. [D] → `ou_meanrevert` (IDEA-025)
- ⬜ **Chebyshev inequality** — distribution-free tail bounds (works without Gaussian assumption). [D] risk → `chebyshev_bound`
- ▫️ **Cauchy distribution** — heavy tails with no finite mean/variance (a caution + a model for extremes). [I]
- ▫️ **Clausius/Carnot — entropy & thermodynamics** — irreversibility, disorder. [I] (→ MaxEnt, D5)
- ▫️ **Riemann — Riemannian geometry** — curved spaces → information geometry (D6). [I]

## E3 — Statistical mechanics, dynamics, electricity (1870–1910)  ["Boltzmann/Poincaré/Tesla/Bachelier"]
- ⬜ **Boltzmann/Gibbs — S=k·lnW, Boltzmann distribution, ensembles, partition function** — the physics of many interacting agents → MaxEnt, Ising markets. [I] → `ising_herding`, `maxent_dist`
- ▫️ **Tesla / harmonic resonance & AC oscillation** — resonance and harmonics. HONEST: the *science* (resonance, harmonic/spectral analysis) is real and lives in Fourier/Hilbert nodes; "Tesla 3-6-9" market mysticism is NOT proven → SCREENED OUT below. [I] cycle → (use `fft_cycles`/`hilbert_phase`)
- ✅ **Poincaré (1890s) — dynamical systems, recurrence theorem, Poincaré maps, sensitive dependence** — origin of chaos theory (and Bachelier's advisor). [I] → `takens_embed`, `rqa_determinism`
- ✅ **Lyapunov (1892) — stability theory & Lyapunov exponents** — chaos vs noise. [I] → `lyapunov_chaos` (IDEA-001/005)
- ⬜ **Bachelier (1900) — random walk / Brownian model of prices + diffusion PDE for options** — THE origin of quant finance. [D] baseline → `gbm_baseline`
- ▫️ **Einstein (1905) / Smoluchowski — Brownian motion physics, diffusion equation, Einstein relation** — diffusion of price. [D] → `diffusion` (D4)
- ✅ **Markov (1906) — Markov chains** — memoryless state transitions → HMM/regime. [D] → `markov_switching`, `hmm_regime`
- ⬜ **Langevin (1908) — dX = −γX dt + noise** — stochastic dynamics; OU/mean-reversion engine. [D] → `langevin_sampler`
- ▫️ **Pearson (1901) — correlation & PCA** — linear dependence & factor extraction. [D] → factor/embedding nodes

## E4 — Modern probability, stochastic calculus, control, information (1910–1985)
- ⬜ **Fokker-Planck (1914) — evolution of the probability density** — full distribution forecast. [I] → `fokker_planck`
- ✅ **Wiener (1923/40s) — Wiener process; Wiener filter; Wiener-Khinchin (autocorr↔spectrum); cybernetics** — rigorous BM + optimal filtering. [D] → `wiener_filter`, `fft_cycles`
- ✅ **Yule/Slutsky (1927) — autoregression (Yule-Walker); random shocks make cycles** — AR(p) foundation. [D] → `ar_forecast`
- ▫️ **Wold (1938) — decomposition: stationary = deterministic + MA(∞)** — the theorem ARMA rests on. [D] → ARMA nodes
- ✅ **Kolmogorov (1933) — probability axioms; forward/backward eqs; turbulence scaling (1941); complexity (1965)** — the rigorous base + the turbulence→multifractal-markets link. [D] → `multifractal`, `mdl_compress`
- ▫️ **Itô (1944) — Itô calculus / Itô's lemma** — the engine of every SDE/option model. [D] → underpins all D4 SDE nodes
- ▫️ **von Neumann-Morgenstern (1944) — game theory + expected-utility theorem** — rational decision under risk. [D] → decision layer
- ⬜ **Bellman (1953) — dynamic programming / Bellman equation / principle of optimality** — the root of RL & HJB. [D] sizing/exit → `hjb_control`, `rl_policy`
- ⬜ **Pontryagin (1956) — maximum principle** — optimal control of trajectories. [D] execution → `almgren_exec`
- 🔬 **Markowitz (1952) — mean-variance portfolio** — risk-return optimization. [D] → `markowitz`
- 🔬 **Kelly (1956) — growth-optimal bet sizing f*=μ/σ²** — the proven sizing law. [D] → `kelly_size`
- ✅ **Kalman (1960) — recursive optimal linear filter** — latent state from noisy obs. [D] → `kalman_state` (have)
- ▫️ **Sharpe (1964) — CAPM & Sharpe ratio** — risk-adjusted performance. [D] → evaluation metric
- ✅ **Mandelbrot (1963→) — fat tails, stable laws, fractals, Hurst long-memory, multifractal markets** — markets are wilder than Gaussian (PROVEN stylized facts: heavy tails, vol clustering, long memory). [D] risk/regime → `multifractal`, `hurst_dfa`
- ⬜ **Black-Scholes-Merton (1973) — option pricing PDE** — the diffusion PDE applied to derivatives. [D] → `bs_pricing`
- ✅ **Engle (1982) ARCH / Bollerslev (1986) GARCH** — volatility clustering. [D] → `egarch_vol` (have)

## ⛔ SCREENED OUT — popular but NOT statistically proven (senior-engineer guard, Rule 30)
These attach to "old-master + trading" but lack rigorous, leakage-free, out-of-sample evidence. Do NOT build as
standalone signals; if ever tested, demand an especially strict permutation-null + OOS bar (likely self-fulfilling):
- **Gann angles / squares** — placement subjective; "not meaningfully backtestable" per quant sources.
- **Fibonacci retracement as PREDICTION** — the sequence is real math, but predictive edge is unproven / self-fulfilling.
- **Elliott Wave** — subjective, not falsifiable; weak/anecdotal evidence.
- **Astro/planetary cycles, "Tesla 3-6-9", sacred geometry** — pseudoscience; exclude.
- KEEP the legitimate cousin instead: **Fourier/wavelet/Hilbert spectral cycle analysis** (proven) — that's the real "harmonics."

## Sources (representative; full links in ML_DISCUSSIONS.md entry)
TimesFM (google-research/timesfm), Chronos/Moirai, FinCast (CIKM 2025), Kronos, LiT order-book (2025);
rough volatility + path signatures (arXiv 2507.23392, 2402.01820, 2508.02759), rough Hawkes-Heston (2210.12393);
econophysics Ising/multifractal/agent-based (arXiv 1404.0243, 0909.1974, 2506.23837); TDA + optimal transport
(arXiv 2507.19504, 2403.19097); Mamba TS survey (2411.02941); plus classical quant/stat texts (Tsay, Cont,
Bouchaud-Potters "Theory of Financial Risk", Lopez de Prado "Advances in Financial ML").
TIER-2 sources: signature kernels / Neural CDE (arXiv 2303.17671, ICLR 2025), Neural SDE training (2410.03973),
quantum path signatures (2508.05103); RMT covariance cleaning (Bun-Bouchaud-Potters arXiv 1610.08104, 2107.01352);
diffusion synthetic finance (2507.19003, 2410.18897); Wasserstein DRO (NeurIPS 2024) + conformal under shift
(ICLR 2025, arXiv 2501.13430); Schrödinger-bridge TS (2503.02943); NTK / stat-mech of learning (2012.04030,
Nat.Commun. 2021); PID (arXiv 1801.09010); deep optimal stopping (Becker-Cheridito-Jentzen); Fernholz SPT;
Cover universal portfolio. (Frontier — many are simulator/research-grade; gate hard per Rule 30.)
TIER-3 sources: Bachelier "Théorie de la Spéculation" (1900, MacTutor bio); Physics & Financial Economics
1776–2014 (arXiv 1404.0243); Mandelbrot "(Mis)Behavior of Markets" + multifractal cascades (arXiv cond-mat/0102369,
2602.02078); plus the canonical originals (Boltzmann, Poincaré, Lyapunov, Einstein 1905, Wiener, Kolmogorov,
Itô, Bellman, Kalman, Markowitz, Kelly, Black-Scholes-Merton). Screened-out evidence: QuantifiedStrategies
(Gann "not meaningfully backtestable"), ResearchGate survey on Fibonacci/Elliott (evidence anecdotal).
