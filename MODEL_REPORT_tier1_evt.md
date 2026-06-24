# MODEL_REPORT — TIER-1 · `evt_tail_risk` (Extreme Value Theory)

**Date:** 2026-06-23 · **Type:** risk/feature tool (not a forecaster) · **Verdict: KEEP-as-utility**
**Files:** `pattern_brain/nodes/tier1_classics.py` · `tests/test_tier1_classics.py`
First node of the TIER-1 directly-buildable cluster, built one-at-a-time under the Code-Generation Contract (oracle-test-first).

## What it computes
Characterizes the heavy tail of a generic series (D1 Extreme Value Theory):
- **Hill tail index** α (smaller = heavier) from the upper-tail log-spacings; for a Pareto tail P(X>x)~x^−α the Hill estimator → 1/α so α̂ = 1/H.
- **GPD Peaks-Over-Threshold** shape ξ and scale β by method-of-moments on the 95%-threshold excesses (ξ̂ = (1 − m²/v)/2, optimizer-free, stable for ξ<½).
- **VaR₉₉ / ES₉₉** via the McNeil-Frey POT formulas.
Emits a `signal` belief: `series` = a per-point extremeness signal (robust |deviation| in MAD units, length T), payload = {tail_index, xi, beta, var_99, es_99, threshold, n_exceedances}. Domain-agnostic (Rule 23).

## Oracle (Phase-4, written FIRST) — 7/7 PASS
`PYTHONPATH=. python3 -m pytest tests/test_tier1_classics.py -q` → **7 passed**.
- Hill recovers the Pareto tail index α∈{2,3,4} within ±0.4 on 60k samples.
- Hill ranks Student-t(3) heavier (smaller index) than Gaussian.
- GPD method-of-moments recovers ξ∈{0.1,0.25,0.4} within ±0.12.
- VaR monotone in level (VaR₉₉>VaR₉₅>0) and ES≥VaR.
- Node registers, emits a conformant `signal`, finite length-T series, reports a heavier (lower) index for t(3) than Gaussian, and degrades to low confidence on short input.

## Real-data check (Rule 34 — its design-appropriate home turf: fat-tailed returns)
Real crypto daily log-returns (`data/panel/*.npz`, 10 symbols, close column):

| metric | result | reading |
|---|---|---|
| crypto tail index | median **2.5** (2.26–2.80) | matches the textbook "cubic law" of financial returns (α≈3) — genuinely heavy |
| Gaussian control | **3.8** (thinner) | correctly lighter than crypto |
| Student-t(3) control | **2.4** | correctly ≈ crypto (heavy) |
| VaR₉₉ / ES₉₉ | 1–3% / 1.2–3.9% daily | realistic tail risk |

Consistent with the project's `stylized_facts` validator (heavy tails on 10/10). A self-inflicted eval bug was caught and fixed before reporting (Rule 14/21): the first run flattened the OHLCV matrix incl. volume → nonsense (Gaussian appeared heavier, VaR≈18); using the close column fixed it.

## Verdict: **KEEP-as-utility (risk tool).**
Oracle-correct and recovers the real fat-tail index on its design data, separating heavy from thin tails. Judged by correctness, not point-forecast PnL — same category as RMT/conformal (the tools that earn their keep in this project). Honest caveats: the Hill index is mildly k-sensitive and the GPD ξ at the 95% threshold is threshold-sensitive at n≈1000 (some ξ go slightly negative even where the Hill index reads heavy) — more data / a threshold sweep would tighten it. Not a forecaster.
