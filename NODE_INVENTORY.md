# NODE & MODEL INVENTORY — full pattern-brain scan
_Auto-generated 2026-06-24 by scanning the live registry + the source tree (owner request 2026-06-24, for the MODEL_NEURAL_NETWORK_VISION build). Regenerate: the python block in the ML_DISCUSSIONS entry of this date._

## A. Registered nodes — **219 node types** across 8 functional layers
These are the bank: every class registered via `@register` in `pattern_brain/nodes/*.py`, keyed by `node_type`, callable through `registry.create(node_type)`.

### Layer `decision` — 33 nodes
`adaboost`, `almgren_exec`, `bagging`, `co_training`, `crf_tagger`, `cvar_opt`, `decision_tree`, `extra_trees`, `gaussian_nb`, `gradient_boosting`, `hist_gradient_boosting`, `hjb_control`, `hrp_alloc`, `knn_classifier`, `label_propagation`, `label_spreading`, `lagrange_opt`, `lda`, `log_utility`, `logistic_regression`, `merton_alloc`, `mlp_classifier`, `mpc_position`, `nearest_centroid`, `passive_aggressive_classifier`, `qda`, `random_forest`, `ridge_classifier`, `self_training`, `sgd_classifier`, `sign_vote`, `svm_classifier`, `threshold_policy`

### Layer `equation` — 22 nodes
`anfis`, `ard_regression`, `bsts_forecast`, `elasticnet_regression`, `genetic_symbolic_regression`, `har_rv`, `heston_vol`, `huber_regression`, `koopman_dmd`, `kyle_impact`, `lars_regression`, `lasso_lars_regression`, `lasso_regression`, `linear_regression`, `linear_svr`, `omp_regression`, `polynomial_regression`, `quantile_regression`, `ridge_regression`, `sindy_regression`, `symbolic_regression`, `theil_sen_regression`

### Layer `noise` — 17 nodes
`elliptic_envelope`, `histogram_anomaly`, `ica_denoise`, `iqr_anomaly`, `isolation_forest`, `kernel_pca_denoise`, `knn_distance_anomaly`, `local_outlier_factor`, `mad_anomaly`, `mahalanobis_anomaly`, `one_class_svm`, `pca_denoise`, `pca_residual_anomaly`, `rmt_denoise`, `savgol_denoise`, `svd_denoise`, `zscore_anomaly`

### Layer `pattern` — 21 nodes
`affinity_propagation`, `agglomerative`, `apriori_patterns`, `bayesian_gaussian_mixture`, `birch`, `bisecting_kmeans`, `dbscan`, `eclat_patterns`, `fpgrowth_patterns`, `gmm`, `gsp_patterns`, `hdbscan`, `kmeans`, `kmedoids`, `mean_shift`, `minibatch_kmeans`, `optics`, `prefixspan_patterns`, `som`, `spade_patterns`, `spectral_clustering`

### Layer `probability` — 13 nodes
`bayes_update`, `bayesian_bootstrap`, `bayesian_network`, `bayesian_ridge`, `chebyshev_bound`, `conformal_forecaster`, `fokker_planck`, `gaussian_process`, `kalman_filter`, `kde_anomaly`, `maxent_dist`, `particle_filter`, `sv_particle`

### Layer `rl` — 19 nodes
`deepwalk`, `discounted_ucb`, `double_dqn`, `dueling_dqn`, `epsilon_greedy_bandit`, `exp3_bandit`, `gradient_bandit`, `maddpg_lite`, `mixture_density_gate`, `node2vec`, `q_learning`, `qmix_lite`, `rainbow_lite`, `random_policy`, `sarsa`, `softmax_policy`, `thompson_bandit`, `trpo_lite`, `ucb_bandit`

### Layer `sequence` — 41 nodes
`adaboost_forecast`, `arima_forecast`, `arma_forecast`, `autoregressive`, `bayesian_ar_ensemble`, `deterministic_esn_forecaster`, `drift_forecast`, `dynamic_bayesian_network`, `egarch_forecast`, `esn_forecaster`, `ewma_volatility`, `exp_smoothing`, `extra_trees_forecast`, `fuzzy_ts`, `garch_forecast`, `gaussian_hmm`, `gbm_baseline`, `gbr_forecast`, `grey_gm11`, `hierarchical_hmm`, `higher_order_markov`, `holt_linear_forecast`, `holt_winters_forecast`, `kernel_ridge_forecast`, `knn_forecast`, `langevin_sampler`, `markov_chain`, `mlp_regressor_forecast`, `momentum_kinematics`, `naive_mean_forecast`, `ou_mean_reversion`, `prophet_like_forecast`, `quantile_forecaster`, `rf_forecast`, `ridge_forecast`, `sarima_forecast`, `seasonal_naive_forecast`, `semi_markov`, `svr_forecast`, `theta_forecast`, `var_forecast`

### Layer `signal` — 53 nodes
`autocorr_period`, `bocpd_break`, `bs_pricing`, `butter_lowpass`, `conditional_entropy`, `copula_dependence`, `cross_entropy`, `cumsum`, `detrend`, `difference`, `diffusion_map`, `emd_decompose`, `evt_tail_risk`, `ewma`, `factor_analysis`, `fft`, `gaussian_smooth`, `hawkes_intensity`, `hilbert_envelope`, `hp_filter`, `hurst_exponent`, `infogeom_distance`, `johansen_coint`, `jump_diffusion`, `kolmogorov_complexity`, `lyapunov_exponent`, `mdl_complexity`, `median_filter`, `mfdfa`, `moving_average`, `mutual_information`, `ofi`, `percolation_risk`, `periodogram`, `permutation_entropy`, `persistent_homology`, `phase_space_takens`, `quadratic_hawkes`, `recurrence_rate`, `robust_pca`, `rough_volatility`, `sample_entropy`, `soc_criticality`, `spectral_entropy`, `ssa_decompose`, `transfer_entropy`, `tsallis_entropy`, `tsne_embed`, `vpin`, `wavelet_decompose`, `welch_psd`, `wiener_denoise`, `zscore_normalize`

---

## B. Scattered standalone model/math modules (NOT in the node registry)
A lot of equation/ML code lives as top-level `pattern_brain/*.py` modules — implemented algorithms that are either (i) wired *into* a registered node, (ii) used as a capability/meta-layer by the connector/evaluator/eval-scripts, or (iii) effectively shelf code. Owner flagged this scattering 2026-06-24.

| Module | What it implements | Wired as / used by |
|---|---|---|
| `rmt.py` | Random-matrix (Marchenko-Pastur) covariance cleaning | → node `rmt_denoise` + risk overlays |
| `rie.py` | Bun-Bouchaud rotationally-invariant estimator | covariance cleaning (eval scripts) |
| `hawkes.py` | Hawkes self-exciting intensity (MLE) | → node `hawkes_intensity` |
| `signatures.py` | Path-signature features (Chen) | → node (signature features) |
| `chaos.py` | Takens embed + Rosenstein LLE + analog forecast | → node `lyapunov_exponent`/`phase_space_takens` |
| `causal.py` | Granger / transfer-entropy causality | eval-only (REJECT on daily crypto) |
| `no_regret.py` | Hedge/MW online no-regret meta-combiner | **meta-layer** (fusion combiner) |
| `stylized_facts.py` | Mandelbrot stylized-facts validator | **validation gate** for generative models |
| `conformal.py` | Conformal-prediction coverage wrapper | uncertainty wrapper |
| `factors.py` | Factor-model construction | eval-only |
| `harness.py` | Baseline + permutation-null + bootstrap-CI | **rigor harness** (Rule 30 gate) |
| `evaluator.py` | Purged walk-forward CV, PSR/DSR | **evaluator** (core) |
| `reputation.py` | Per-edge discounted-UCB reputation | **edge memory** (core) |
| `routing.py` | Heuristic/LLM/Reputation routers (9 classes) | **router** (core) |
| `evolution.py` | Pathway mutate/crossover/prune (6 classes) | **evolution engine** (core) |
| `network.py` | `DAGSpec` + `StackedDAG` (mixture/gated/stacked) | **the graph** (core) |
| `interlingua.py` | Belief-type schema + migrations | **belief contract** (core) |
| `decision_vector.py` | `{direction,bullish,bearish,volatility,confidence}` | **P0 interlingua** ✅ |
| `adapters/encoders.py` | MultiModalEncoder → deterministic fused `(T,D)` | **input adapter** (deterministic, NOT learned) |

Plus `tools/eval_*.py` (~25 evaluation scripts) and `code_skills_test/` (the blind-write code-quality benchmark) — test/eval harnesses, not bank models.

---

## C. Vision → existing-code → GENUINE GAP map (what to actually build)
Mapping `MODEL_NEURAL_NETWORK_VISION.md` onto the scan. The vision's rule is "~70% exists, DO NOT rebuild" — confirmed:

| Vision piece | Status in code | Build action |
|---|---|---|
| model=neuron / layer / weighted combiner | ✅ `Node` / layers / `StackedDAG._mixture/_gated/_stacked` | reuse |
| universal belief / interlingua (P0) | ✅ `belief.py` + `decision_vector.py` | reuse |
| deterministic multimodal fusion → `(T,D)` | ✅ `adapters/encoders.py` MultiModalEncoder | reuse as the raw-fusion stage |
| **P1 / Stage-1 LEARNED self-supervised embedding** | ❌ **MISSING** — existing encoder is hand-crafted, not learned | **BUILD (keystone, IDEA-084)** |
| Stage-2 unsupervised regime discovery | ✅ pattern layer (hdbscan, gmm, kmeans, som…) | wire as gate context |
| Stage-3 supervised experts | ✅ decision/sequence/equation layers (many) | reuse |
| **Stage-4 semi-supervised label-expander** | ⚠️ partial — `self_training`, `co_training`, `label_propagation/spreading` nodes exist but not used to expand expert training | **WIRE (IDEA-086)** |
| **P2 / Stage-5 ATTENTION soft-router (learned, gradient)** | ⚠️ routers exist (heuristic/LLM/reputation) + `softmax_policy`/`mixture_density_gate` nodes, but NO learned softmax-attention top-k gate trained on outcomes | **BUILD (IDEA-085)** |
| Stage-5 RL lifecycle policy | ⚠️ rl layer rich (dqn/bandits/sarsa/trpo_lite…) but not on the live decision path | wire (later) |
| **P3 residual stream** | ❌ **MISSING** in `StackedDAG` | **BUILD (IDEA-017)** |
| P4 evolution drives structure | ✅ `evolution.py` Evolver | wire to P1-P3 |
| P-VIZ animated dashboard | ✅ React Flow + `/ws/run` exists | enhance (parallel) |

**Genuine net-new builds, in vision order:** P1 learned embedding → P2 attention router → P3 residual stream → P4 wire-evolution → (P-VIZ parallel). Everything else is reuse/wiring.
