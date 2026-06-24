# MODEL REPORT — TIER-1 DYNAMICAL / FORECAST / GEOMETRIC nodes

File: `pattern_brain/nodes/tier1_dynamical.py` · Tests: `tests/test_tier1_dynamical.py`
(23 oracle/acceptance tests, all passing). Built oracle-test-first per CLAUDE.md; rigor
per ML_ENGINEERING_PRACTICES.md §B/§F/§G and RULES.md 32/34/35/36.
Light-stack only (numpy/scipy/sklearn); `np.linalg.lstsq`/`eigh`/`solve`, never `inv`.

## Verdict summary

| node_type | display name | belief | verdict | headline evidence |
|---|---|---|---|---|
| `koopman_dmd` | Koopman DMD Forecaster | forecast | **SHADOW / REJECT (returns)** · KEEP (oscillator/spectral) | Recovers sine freq exactly (0.05 vs 0.05 true) & damped-osc decay; 5-step clean-oscillator forecast RMSE < 0.05. On panel returns it beats *persistence* 10/10 but **loses to the zero baseline 0/10** — it merely shrinks toward 0; no genuine return-forecast edge. |
| `bsts_forecast` | BSTS Local-Linear-Trend | forecast | **KEEP (levels/trend)** · SHADOW (returns) | Filtered level tracks BTC price with 0.097% rel-error; beats persistence on a clean trend (MSE 1.08 vs 1.77). On returns it does NOT beat the zero baseline (0/10) — honest. |
| `infogeom_distance` | Fisher-Rao Regime Distance | signal | **KEEP-as-utility** | Matches the Costa-Santos-Hancock closed form exactly; =0 for identical Gaussians, monotone in μ & σ separation; spike is >4× every non-jump window at an injected vol-jump; meaningful dynamic range (max > 1.5× mean) on 10/10 panel symbols. |
| `diffusion_map` | Diffusion-Map Embedding | signal | **KEEP-as-utility** | Recovers a noisy 1-D arc's intrinsic ordering (|Spearman| = 1.0); CROSS-DOMAIN (Rule 36) pendigits 3-D kNN accuracy **0.80 vs 0.10 chance** (and ≥ random-projection 0.77). |

## Honest forecaster note (the project lesson, measured)
**Both return-forecasters lose on returns.** The subtlety: *persistence is a strawman
baseline on returns* — predicting `r[t]` carries `r[t]`'s full variance as error. DMD
beats persistence 10/10 only because its forecast shrinks toward zero (pred std 0.0013 vs
return std 0.0046); against the correct **zero/mean baseline** it wins **0/10**. BSTS
likewise beats the zero baseline 0/10 on returns. Neither earns a return-forecast slot —
they earn their keep only on the data their theory consumes (oscillatory/spectral systems
for DMD; trending LEVEL series for BSTS).

## Per-node detail

### `koopman_dmd` — Koopman / Dynamic Mode Decomposition
- **Math:** delay-embed (Hankel) → fit `A` on time-shifted snapshots `X' ≈ A X` by
  `lstsq` on the transposed system → `eigvals(A)` → discrete-time mode summary
  (|λ|, angle/2π = cycles/step, ln|λ| = growth) → forecast by iterating `A`.
- **ORACLE (recovery):** sine (period 20) → recovered freq **0.05000** (true 0.05);
  damped osc (decay 0.99) → eigenvalue magnitude within 0.02 of 0.99; 5-step forecast of
  a two-tone clean oscillator **RMSE < 0.05**. **CONTROL:** ≤5-sample input degrades to
  persistence, no crash.
- **DESIGN-DATA (panel returns, 1-step):** beats persistence 10/10 but **loses to zero
  0/10** → REJECT for return forecasting; KEEP only as a spectral/oscillation analyzer.

### `bsts_forecast` — Bayesian Structural Time Series (local-linear-trend Kalman)
- **Math:** state `[level, slope]`; `F=[[1,1],[0,1]]`, `H=[1,0]`; light MoM variances
  (obs from 2nd-diff noise). Standard Kalman predict/update (scalar innovation, gain
  divides by scalar `S` — no `inv`). Forecast = `level + slope`, std from `P[0,0]+R`.
- **ORACLE:** filtered level tracks a linear trend with RMSE < the noise std; beats
  persistence on a clean trend (MSE **1.08 vs 1.77**). **CONTROL:** on white noise the
  forecast stays bounded near 0 (degrades to ~persistence), no blow-up.
- **DESIGN-DATA:** on price **LEVELS** the filtered level tracks BTC close with **0.097%**
  rel-error (its home turf → KEEP for trend extraction); on **returns** it beats the zero
  baseline **0/10** (SHADOW, honest).

### `infogeom_distance` — Fisher-Rao regime distance
- **Math:** each rolling window → Gaussian (μ,σ); closed-form Fisher-Rao geodesic
  `d = √2·arccosh(1 + [(μ1−μ2)² + 2(σ1−σ2)²]/(4σ1σ2))` between consecutive windows.
- **ORACLE:** =0 for identical; monotone increasing in both μ- and σ-separation; matches
  the closed form to 1e-9. **DESIGN-DATA/REGIME:** injected vol-jump → spike **>4×** every
  other window, at the correct window index; **CONTROL:** stationary series stays flat
  (max < 1.0). Real panel: max > 1.5× mean on 10/10 symbols.
- **Verdict: KEEP-as-utility** — a correct, fast regime-change distance tool (not a PnL
  forecaster; judged by correctness).

### `diffusion_map` — Diffusion-map manifold embedding (GENERAL-PURPOSE)
- **Math:** Gaussian affinity `W=exp(−‖xi−xj‖²/ε)` (ε = median sq-dist) → symmetric
  normalized `S = D^{-1/2} W D^{-1/2}` (eigenvalues of the Markov `P=D^{-1}W`, solved via
  stable `eigh`) → right eigenvectors scaled by eigenvalues = diffusion coordinates,
  skipping the trivial constant mode.
- **ORACLE:** noisy 1-D arc → leading diffusion coordinate is monotone in the true arc
  parameter (**|Spearman| = 1.0**). **CONTROL:** structureless Gaussian cloud → no
  spurious ordering (|Spearman| < 0.5 vs a random index).
- **CROSS-DOMAIN (Rule 36, MANDATORY for a general method) — pendigits handwriting:**
  3-D diffusion embedding + 7-NN gives **0.80 accuracy vs 0.10 chance** (and ≥ a same-dim
  random projection at 0.77). The embedding genuinely separates digit classes on real
  non-trading data.
- **Verdict: KEEP-as-utility** — a validated general manifold-embedding tool.

## Reproduce
```
PYTHONPATH=. python3 -m pytest tests/test_tier1_dynamical.py -q
# 23 passed
```
