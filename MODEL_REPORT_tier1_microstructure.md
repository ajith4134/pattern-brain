# MODEL_REPORT — TIER-1 MICROSTRUCTURE nodes

Three microstructure nodes built oracle-test-first per the CLAUDE.md Code-Generation
Contract (Phase-4 oracle written before Phase-3 code). All judged by **correctness**
(recover a known synthetic answer + behave sanely on real trades), not PnL — these
are flow utilities, in the spirit of EVT/RMT.

- **Code:** `pattern_brain/nodes/tier1_microstructure.py`
- **Tests:** `tests/test_tier1_microstructure.py` — `PYTHONPATH=. python3 -m pytest tests/test_tier1_microstructure.py -q` → **10 passed**
- **Design data (§G):** `data/ticks_micro/{BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT}.npz` (~30k trades/symbol, ~80 min each). Keys: `times, price, qty, side(+1 buy / -1 sell)`; `signed_volume = side*qty`. The **node is domain-agnostic** ((T,D) in, Belief out); the **test harness** owns the trade→(T,D) mapping (10s time-bars for `ofi`/`kyle_impact`; per-trade signed-volume series + equal-volume buckets for `vpin`).

## Verdict table

| node_type | belief | verdict | headline evidence |
|---|---|---|---|
| `ofi` | signal | **KEEP-as-utility** | Oracle 3/3 (pos corr 0.8+; control \|corr\|<0.2; window-sum exact). REAL: corr(OFI, contemporaneous bar return) = **+0.56 / +0.36 / +0.42 / +0.26** (BTC/ETH/SOL/DOGE) vs shuffled-OFI control ≈ **0.00** all four. Genuine, correct, informative flow signal. |
| `vpin` | signal | **KEEP-as-utility** | Oracle 2/2 (one-sided toxic flow → mean_vpin **0.97**; balanced → **<0.25**). Estimator correct. REAL toxicity↔vol link **NOT confirmed** on this short calm sample — see caveat. |
| `kyle_impact` | equation | **KEEP-as-utility** | Oracle 2/2 (recover known λ=0.7 to **±0.05**, r²>0.9; independent flow → λ≈0, r²≈0). REAL: λ>0 on all 4 (net buying lifts price — correct sign), and flow explains more variance on majors (BTC r²=0.32 > DOGE r²=0.07). |

## Honest caveats / what I could not make work cleanly

1. **`vpin` toxicity↔volatility is the OPPOSITE sign on this data.** corr(per-bucket VPIN, per-bucket RV) is consistently **negative** (−0.45 / −0.51 / −0.22 / −0.51), and corr(VPIN, |net price move|) is ~0. The textbook positive link does not appear here, almost certainly because the ~80-min sample is too short/calm to contain a genuine toxic flash episode, and RV-within-a-volume-bucket is dominated by two-sided high-count churn (high RV ↔ low VPIN). The **estimator is correct** (oracle proves it); the **empirical toxicity claim is unproven** and needs a stress-period dataset (Rule 36: more varied evidence). The real-data test therefore asserts only the safe properties (VPIN∈[0,1], finite), not a positive RV correlation, so it does not encode a false claim.

2. **`kyle_impact` cross-asset λ ordering is confounded — the naive expectation was wrong.** Raw price-unit λ is dominated by the quote price LEVEL (BTC≈$62k vs DOGE≈$0.1) and by per-bar token volume, so "liquid BTC has the smallest λ" does **not** hold in absolute units (BTC λ_px=1.5 is the LARGEST). The comparable quantity is relative impact (λ on a log-return basis × typical \|signed vol\|): BTC +0.05 / ETH +0.07 / SOL +0.19 / DOGE +0.02 bps per typical bar-flow — still not a clean liquid→illiquid monotone ordering across these particular contracts. I corrected the test to assert the **defensible** facts (λ sign positive; flow-explained variance higher on majors) and documented the confound rather than asserting a false ordering.

3. **All three are utilities, not alpha.** They are correct, informative microstructure descriptors. None is claimed to beat a return baseline OOS, so none earns a plain KEEP under Rule 32 — KEEP-as-utility is the honest tier (same basis as EVT/RMT).
