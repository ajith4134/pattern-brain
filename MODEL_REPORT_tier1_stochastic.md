# MODEL REPORT — TIER-1 STOCHASTIC-PROCESS / PHYSICS nodes

Module: `pattern_brain/nodes/tier1_stochastic.py` · Tests: `tests/test_tier1_stochastic.py`
Built oracle-test-first (CLAUDE.md 5-phase contract). Light stack only (numpy/scipy/sklearn).
Run: `PYTHONPATH=. python3 -m pytest tests/test_tier1_stochastic.py -q` → **12 passed**.

## Verdict table

| node_type | verdict | headline evidence |
|---|---|---|
| `jump_diffusion` | **KEEP-as-utility** | Oracle: recovers λ=0.036 vs true 0.05 (±0.014); jump-separation precision **0.99** / recall **0.63** (the ~40% of simulated jumps drawn < 2σ of diffusion are physically unrecoverable by *any* threshold detector). Control: pure-Gaussian → λ≈0.002, ≤6 false jumps/2000. Real crypto: 1.7–4.4% jump rate per symbol, diffusion vol cleanly separated. A tail/gap detector judged by correctness, not PnL. |
| `heston_vol` | **KEEP** | Oracle: recovers θ=0.04 (±0.01) and κ within order of magnitude; control monotone (fast-reversion κ > slow-reversion κ). Real: walk-forward 1-step variance forecast **beats persistence (RV_{t+1}=RV_t) on 10/10 symbols** (e.g. ETH MSE 1.52e-8 vs 2.64e-8). Honest caveat: the bar was "≥50% vs persistence" — HAR-RV (already in the bank) likely beats it; this earns KEEP on its own design turf but is *not* claimed to dominate HAR. |
| `sv_particle` | **KEEP-as-utility** | Oracle: filtered vol corr **0.76** with true latent vol (φ-recovery via log-autocovariance slope fixes the log-y² attenuation bias: recovered φ≈0.92–0.98 vs true 0.95). Control φ=0 handled (finite, positive). Real: filtered vol vs \|returns\| mean corr **0.69** across 10 symbols, φ in realistic 0.84–0.97. A de-noised latent-vol estimator. |
| `percolation_risk` | **KEEP-as-utility** | Oracle: block-correlated system retains a giant component to a **higher** threshold than independent assets (0.75 vs 0.10–0.15); order parameter monotone in connectivity. Control extremes: perfectly-correlated → giant_frac 1.0/thr 0.95; independent → giant_frac 0.1/thr 0.1. Real universe (60 assets): giant_frac 0.90, percolation threshold 0.75, mean degree 39.5; rolling systemic-risk series corr **0.45** with market-wide move magnitude (rises in crisis windows). |

## What each node is and the design input (Rule 34/35, §G)

1. **`jump_diffusion`** — Merton jump-diffusion separator. Design input: a **return series** (panel close → diff-log). Robust local-vol (rolling 1.4826·MAD) standardizes returns; |z| > 3.5 flags compound-Poisson jumps; λ = #jumps/n, jump mean/std from flagged obs, diffusion vol = robust scale of the un-flagged remainder. Emits `signal` (per-point jump indicator) + payload {lambda, jump_mean, jump_std, diffusion_vol, n_jumps}. Threshold 3.5 chosen empirically: precision 0.99 / recall 0.63 on detectable jumps, ≤0.003 false-λ on clean Gaussian.

2. **`heston_vol`** — Heston/CIR stochastic-variance estimator. Design input: a **realized-variance series** (harness supplies RV = returns²). Method-of-moments on the exact CIR discretization: θ̂ = mean(RV); φ = exp(−κdt) = lag-1 autocorr of de-meaned RV ⇒ κ̂ = −ln φ; ξ̂ from the AR(1) residual variance scaled by the conditional-variance factor. One-step forecast = θ + φ(v_t − θ). Emits `forecast` {next_variance, kappa, theta, xi}.

3. **`sv_particle`** — Stochastic-Volatility bootstrap (SIR) particle filter. Design input: a **return series**. Latent log-vol AR(1) h_t = μ + φ(h_{t-1}−μ) + ση; obs y_t = exp(h_t/2)ε_t. Params from log(y²)+1.2704: μ = mean, **φ from the slope of log γ_k vs k** (the obs noise cancels at lags ≥1 — fixes the severe attenuation bias of the naive lag-1 autocorrelation), ση² = Var(h)(1−φ²). 400-particle filter with systematic resampling and an exp-overflow guard. Emits `forecast` {series=filtered vol, phi, mu, sigma_eta}.

4. **`percolation_risk`** — percolation / network-contagion systemic-risk. Design input: a **cross-section of returns** (T,D), D=assets. Threshold |correlation| graph; union-find giant-component fraction; sweep threshold 0.05→0.95 to locate the percolation transition (largest threshold where a giant component, frac>0.5, still spans the net). Rolling 60-window giant-component fraction at a fixed reference threshold (0.5) is the systemic-risk index. Emits `signal` {series, giant_component_frac, percolation_threshold, mean_degree, order_parameter_curve}.

## Rigor notes (ML_ENGINEERING_PRACTICES §A–G)
- **Oracle-first** (Phase 4 before Phase 3): every node has a parameter-recovery oracle + a control/null oracle written before the implementation.
- **Baselines:** heston vs persistence (the brutal vol baseline); sv vs realized \|returns\|; percolation vs independent-asset control.
- **Adversarial (§F):** constant/tiny/extreme-magnitude (1e6) series and perfectly-correlated vs independent cross-sections all produce finite, sane output with **no RuntimeWarnings** (verified under `-W error::RuntimeWarning`).
- **Numerical correctness:** float64; log-space likelihoods in the particle weights; log-autocovariance slope (not the attenuated lag-1 ratio) for φ; exp-overflow clamp on the log-vol state; robust MAD scales.

## Honest failures / limits
- `jump_diffusion` recall is capped at ~0.63 *by physics*, not by the code: jumps drawn smaller than the diffusion scale are indistinguishable from diffusion. This is correct behavior, not a defect.
- `heston_vol` beats persistence but the daily RV proxy (returns²) is extremely noisy; mean-reversion-to-θ wins largely because persistence on a noisy proxy is weak. It is **not** claimed to beat HAR-RV.
- `sv_particle` and `percolation_risk` are de-noising / risk-indicator tools (KEEP-as-utility), judged by correctness and design-input fidelity, not by standalone forecast PnL — consistent with the project lesson that standalone point-forecasters lose to persistence.
