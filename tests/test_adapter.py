"""Contract + behavior tests for the stock-data adapter (build-tracker step 3).

Run: python3 tests/test_adapter.py
Exits non-zero if any check fails. Encodes the step-3 spec (PLAN.md tracker
item 3) as checkable conditions per Rule 25: convert stock candles into the
generic (T, D) inputs the ALREADY-BUILT model bank + Connector (steps 1-2)
require — never reshaping the core. Covers the §6 gap-7f time/asynchrony fix
(align_to_grid / align_sources) and the §5/§6 frac-diff feature technique, and
proves the adapter output runs unchanged through default_connector().
"""
from __future__ import annotations

import os
import sys
import json
import tokenize

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain.adapters import (
    CandleSeries, StockDataAdapter,
    frac_diff, frac_diff_weights, align_to_grid, align_sources,
)

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    if not cond:
        FAILS.append(msg)
    return cond


def make_candles(T=400, seed=0, with_ts=True, step=60.0):
    """A synthetic but realistic OHLCV series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, T)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, T)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, T)))
    vol = rng.lognormal(mean=8.0, sigma=0.5, size=T)
    ts = np.arange(T, dtype=float) * step if with_ts else None
    return CandleSeries(open_, high, low, close, vol, timestamp=ts)


# ----------------------------------------------------------- frac-diff anchors
def test_fracdiff_anchor_cases():
    """d=0 is the identity; d=1 is a plain first difference (the two cases that
    make frac-diff checkable, AFML ch. 5)."""
    w0 = frac_diff_weights(0.0)
    check(np.allclose(w0, [1.0]), f"d=0 weights should be [1], got {w0}")
    w1 = frac_diff_weights(1.0)
    check(np.allclose(w1, [1.0, -1.0]), f"d=1 weights should be [1,-1], got {w1}")

    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    fd0 = frac_diff(x, 0.0)
    check(np.allclose(fd0, x), f"d=0 should be identity, got {fd0}")
    fd1 = frac_diff(x, 1.0)
    check(np.isnan(fd1[0]), "d=1 first output should be NaN (warm-up)")
    check(np.allclose(fd1[1:], np.diff(x)), f"d=1 should be first difference, got {fd1}")
    print(f"  frac-diff anchors: d=0 identity, d=1 first-difference  (w1={w1})")


def test_fracdiff_preserves_more_memory_than_returns():
    """The whole point of frac-diff (§5): at a fractional d it is stationary yet
    retains more correlation with the level than plain returns (d=1) do."""
    rng = np.random.default_rng(3)
    price = np.cumsum(rng.normal(0, 1, 600)) + 100.0  # a memoryful level series
    fd_frac = frac_diff(price, 0.4)
    fd_ret = frac_diff(price, 1.0)
    m = ~np.isnan(fd_frac) & ~np.isnan(fd_ret)
    corr_frac = abs(np.corrcoef(fd_frac[m], price[m])[0, 1])
    corr_ret = abs(np.corrcoef(fd_ret[m], price[m])[0, 1])
    check(corr_frac > corr_ret,
          f"frac-diff(0.4) should keep more memory than returns: {corr_frac:.3f} !> {corr_ret:.3f}")
    print(f"  memory: |corr(level)| frac-diff(0.4)={corr_frac:.3f} > returns={corr_ret:.3f}")


# --------------------------------------------------- time alignment (gap 7f)
def test_align_to_grid_is_causal():
    """ffill alignment is as-of (most recent obs at or before each grid point) —
    no lookahead. The correct closure of §6 gap 7f for a single source."""
    ts = np.array([0.0, 10.0, 25.0, 30.0])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    grid, out = align_to_grid(ts, vals, step=5.0)  # grid: 0,5,10,...,30
    out = out.ravel()
    check(np.allclose(grid, np.arange(0, 31, 5)), f"unexpected grid {grid}")
    # at t=5 the latest obs <= 5 is the one at t=0 (=1.0), NOT the t=10 one
    check(out[1] == 1.0, f"ffill peeked ahead at t=5: got {out[1]} expected 1.0")
    check(out[2] == 2.0, f"t=10 should equal obs at 10 (=2.0), got {out[2]}")
    check(out[grid.tolist().index(20.0)] == 2.0,
          "t=20 should still hold the t=10 value (no obs between 10 and 25)")
    print(f"  align: causal ffill, no lookahead  (grid {grid.size} pts)")


def test_align_sources_shared_grid():
    """Two sources sampled at DIFFERENT rates collapse onto one shared (T, D)
    matrix over their common overlap — the cross-source half of gap 7f."""
    ts_fast = np.arange(0, 100, 2.0)
    ts_slow = np.arange(0, 100, 7.0)
    a = (ts_fast, np.sin(ts_fast))
    b = (ts_slow, np.cos(ts_slow))
    grid, M = align_sources([a, b], step=5.0)
    check(M.ndim == 2 and M.shape[1] == 2, f"expected 2 aligned columns, got {M.shape}")
    check(M.shape[0] == grid.shape[0], "aligned rows != grid length")
    check(np.all(np.isfinite(M)), "aligned matrix has non-finite values")
    print(f"  align_sources: 2 rates ({ts_fast.size},{ts_slow.size}) -> shared {M.shape}")


# ------------------------------------------------ adapter -> generic (T, D)
def test_adapter_emits_generic_finite_TD():
    """The adapter output is a finite (T, D) float array — exactly the core's
    Node._validate contract. The data bends to the system (PLAN.md §0)."""
    cs = make_candles()
    ad = StockDataAdapter()  # default 4 features
    X = ad.transform(cs)
    check(X.ndim == 2, f"adapter output not 2-D: {X.shape}")
    check(X.shape[1] == len(ad.feature_names), f"D ({X.shape[1]}) != #features {ad.feature_names}")
    check(np.all(np.isfinite(X)), "adapter emitted non-finite values (violates core contract)")
    check(X.shape[0] > 0, "adapter emitted zero rows")
    # it must satisfy the SAME validator the core nodes use, unmodified
    from pattern_brain.node import Node
    Node._validate(X)  # raises if it doesn't meet the contract
    print(f"  adapter: candles({len(cs)}) -> finite (T,D)={X.shape}, cols={ad.feature_names}")


def test_adapter_feeds_default_connector_unchanged():
    """The whole point of step 3: adapter output runs through the ALREADY-BUILT
    Connector with no change to the core (Rule 21 — does its job, breaks nothing)."""
    X = StockDataAdapter().transform(make_candles())
    res = pb.default_connector().run(X)
    check([h.node_type for h in res.hops] == pb.DEFAULT_PATHWAY,
          "default pathway did not run end-to-end on adapter output")
    check(res.output is not None and res.output.type == "decision",
          f"no terminal decision on adapter output: {res.output}")
    # the full trace stays JSON-serializable (audit/dashboard requirement)
    json.dumps(res.to_dict())
    print(f"  integration: adapter -> {' -> '.join(res.pathway)} -> "
          f"action={res.output.payload.get('action')}")


def test_resample_absorbs_gaps():
    """With resample_step set, an irregular/gappy candle series is aligned onto a
    regular grid before features are built (§6 gap 7f), still yielding finite (T,D)."""
    cs_full = make_candles(T=300, step=60.0)
    # drop a chunk of candles to simulate a data gap / asynchronous feed
    keep = np.r_[0:100, 160:300]
    gappy = CandleSeries(
        cs_full.open[keep], cs_full.high[keep], cs_full.low[keep],
        cs_full.close[keep], cs_full.volume[keep], timestamp=cs_full.timestamp[keep],
    )
    ad = StockDataAdapter(resample_step=60.0)
    X = ad.transform(gappy)
    check(np.all(np.isfinite(X)), "resampled output has non-finite values")
    check(X.shape[0] > 0, "resampling produced no rows")
    print(f"  resample: gappy candles({len(gappy)}) -> regular-grid finite (T,D)={X.shape}")


# --------------------------------------------------------- validation / errors
def test_validation_fails_loudly():
    """Bad inputs raise rather than silently degrading."""
    try:
        CandleSeries(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                     np.array([1.0, 2.0]), np.array([1.0]))  # mismatched length
        check(False, "mismatched OHLCV lengths did not raise")
    except ValueError:
        pass
    try:
        CandleSeries(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                     np.array([-1.0]), np.array([1.0]))  # non-positive close
        check(False, "non-positive close did not raise")
    except ValueError:
        pass
    try:
        StockDataAdapter(features=["not_a_feature"])
        check(False, "unknown feature did not raise")
    except ValueError:
        pass
    try:
        StockDataAdapter(resample_step=60.0).transform(make_candles(with_ts=False))
        check(False, "resample without timestamps did not raise")
    except ValueError:
        pass
    print("  validation: length/price/feature/timestamp errors all raise")


# --------------------------------------------------- Rule 23 separation checks
def test_core_does_not_import_adapter():
    """Rule 23: the CORE package must stay domain-agnostic — its top-level
    __init__ must NOT import the adapter subpackage."""
    import pattern_brain
    src_init = os.path.join(os.path.dirname(pattern_brain.__file__), "__init__.py")
    with open(src_init) as fh:
        text = fh.read()
    check("adapters" not in text,
          "pattern_brain/__init__.py imports adapters — core no longer domain-clean")
    print("  separation: core __init__ does not import adapters (core stays clean)")


def test_generic_tools_are_domain_free():
    """Rule 23 / portability: the generic numeric tools (frac_diff, align_*) must
    carry NO stock identifiers, so a future domain reuses them unchanged. Domain
    words are allowed only inside CandleSeries / StockDataAdapter. Inspect code
    identifiers (tokenize), not docstrings (prose may name what code avoids)."""
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "adapters", "stock.py")
    # collect identifier tokens inside the generic functions only, by name span
    generic_names = {"frac_diff_weights", "frac_diff", "align_to_grid",
                     "align_sources", "_prepend_nan"}
    src_lines = open(path).read().splitlines()
    # crude but effective: gather def blocks for generic funcs, scan their tokens
    hits = []
    i = 0
    while i < len(src_lines):
        line = src_lines[i]
        stripped = line.strip()
        if stripped.startswith("def ") and any(
            stripped.startswith(f"def {g}(") for g in generic_names
        ):
            j = i + 1
            block = [line]
            while j < len(src_lines) and (src_lines[j].startswith((" ", "\t")) or not src_lines[j].strip()):
                block.append(src_lines[j])
                j += 1
            code = "\n".join(block)
            import io
            try:
                for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                    if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                        hits.append((stripped.split("(")[0], tok.string))
            except tokenize.TokenError:
                pass
            i = j
            continue
        i += 1
    check(not hits, f"domain identifiers leaked into generic tools: {hits}")
    print("  separation: generic numeric tools carry no stock identifiers (portable)")


def main():
    print("=" * 70)
    print("Pattern Brain — stock-data adapter (step 3): contract + behavior tests")
    print("=" * 70)
    test_fracdiff_anchor_cases()
    test_fracdiff_preserves_more_memory_than_returns()
    test_align_to_grid_is_causal()
    test_align_sources_shared_grid()
    test_adapter_emits_generic_finite_TD()
    test_adapter_feeds_default_connector_unchanged()
    test_resample_absorbs_gaps()
    test_validation_fails_loudly()
    test_core_does_not_import_adapter()
    test_generic_tools_are_domain_free()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
