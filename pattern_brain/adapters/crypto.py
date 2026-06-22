"""Phase 8 — slice 9 (capstone): crypto data acquisition (PLAN.md §12.1).

The ONE domain-specific surface for the capstone (Rule 23 — lives in ``adapters/``,
not imported by core). Fetches a single crypto symbol's market data via **ccxt**
(klines; funding/OI/order-book are best-effort extensions), turns it into the
generic ``(T, D)`` the whole bank consumes (column 0 = log-return = the "next move"
target), caches successful live pulls under ``data/crypto/`` (Rule 1), and falls
back to a realistic synthetic series when ccxt/network are unavailable — so the
capstone is runnable anywhere.

``run_crypto_capstone(symbol)`` is the thin wrapper that loads the symbol and calls
the domain-agnostic ``capstone.run_capstone`` (adapter → core dependency only).

LLM-driven acquisition (owner's ask): the agent tool ``fetch_crypto_symbol``
(``agent/tools.py``) calls ``load_symbol`` so the project's OWN LLM orchestrates the
fetch; this module does the deterministic download.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .stock import CandleSeries, StockDataAdapter

try:
    import ccxt
    _HAS_CCXT = True
except Exception:                                       # pragma: no cover - optional dep
    _HAS_CCXT = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "crypto")

# the capstone target is the next MOVE → put log_return at column 0
_CAPSTONE_FEATURES = ("log_return", "fracdiff_log_close", "hl_range", "log_volume")


def ccxt_available() -> bool:
    return _HAS_CCXT


def fetch_ohlcv(symbol: str = "BTC/USD", timeframe: str = "1h", limit: int = 600,
                exchange: str = "kraken") -> np.ndarray:
    """Live OHLCV via ccxt → (T, 6) [ts_ms, open, high, low, close, volume]."""
    if not _HAS_CCXT:
        raise RuntimeError("ccxt not installed")
    ex = getattr(ccxt, exchange)()
    arr = np.array(ex.fetch_ohlcv(symbol, timeframe, limit=limit), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6 or len(arr) < 20:
        raise RuntimeError(f"bad OHLCV from {exchange} {symbol}: shape {arr.shape}")
    return arr[:, :6]


def synthetic_ohlcv(T: int = 600, seed: int = 0, start_price: float = 50_000.0) -> np.ndarray:
    """A realistic-ish OHLCV (geometric returns + volatility clustering) for offline runs."""
    rng = np.random.default_rng(seed)
    vol = np.full(T, 0.01)
    ret = np.zeros(T)
    for t in range(1, T):
        vol[t] = 0.92 * vol[t - 1] + 0.08 * (0.008 + 0.6 * abs(ret[t - 1]))
        ret[t] = rng.normal(0.0, vol[t])
    close = start_price * np.exp(np.cumsum(ret))
    op = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(op, close) * (1.0 + np.abs(rng.normal(0, 0.002, T)))
    lo = np.minimum(op, close) * (1.0 - np.abs(rng.normal(0, 0.002, T)))
    vlm = np.abs(rng.normal(100, 30, T)) + 1.0
    ts = np.arange(T, dtype=float) * 3_600_000.0 + 1_700_000_000_000.0
    return np.column_stack([ts, op, hi, lo, close, vlm])


def _cache_path(symbol: str, timeframe: str, exchange: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = f"{exchange}_{symbol}_{timeframe}".replace("/", "-").replace(":", "-")
    p = os.path.abspath(os.path.join(_CACHE_DIR, safe + ".npz"))
    if not p.startswith(_CACHE_DIR + os.sep):
        raise ValueError("Rule 1: refusing crypto cache path outside the project folder")
    return p


def _causal_vol(r: np.ndarray, w: int = 12) -> np.ndarray:
    """Causal rolling realized volatility of a return series (no look-ahead)."""
    out = np.empty(len(r))
    for t in range(len(r)):
        seg = r[max(0, t - w + 1): t + 1]
        out[t] = float(np.std(seg)) if len(seg) > 1 else abs(float(r[t]))
    return out


def load_symbol(symbol: str = "BTC/USD", timeframe: str = "1h", limit: int = 600,
                exchange: str = "kraken", prefer_live: bool = True,
                use_cache: bool = True, allow_synthetic: bool = True,
                frac_thresh: float = 1e-3, target: str = "return") -> Dict[str, object]:
    """Load a symbol → generic ``(T, D)``. ``target`` picks column 0 (the prediction
    target): ``"return"`` (log-return — the next-move target), ``"volatility"``
    (causal realized vol — vol is strongly autocorrelated, so this is where the
    network is expected to show real predictability), or ``"abs_return"``. Tries
    live ccxt → on-disk cache → synthetic, reporting which ``source`` was used."""
    ohlcv, source = None, None
    cp = _cache_path(symbol, timeframe, exchange)
    if prefer_live and _HAS_CCXT:
        try:
            ohlcv, source = fetch_ohlcv(symbol, timeframe, limit, exchange), "live"
        except Exception:
            ohlcv = None
    if ohlcv is None and use_cache and os.path.exists(cp):
        ohlcv, source = np.load(cp)["ohlcv"], "cache"
    if ohlcv is None and allow_synthetic:
        ohlcv, source = synthetic_ohlcv(T=limit), "synthetic"
    if ohlcv is None:
        raise RuntimeError("no crypto data: ccxt/network unavailable, no cache, synthetic disabled")
    ohlcv = np.asarray(ohlcv, dtype=float)
    if source == "live" and use_cache:
        np.savez(cp, ohlcv=ohlcv)                       # cache the live pull (Rule 1, gitignored)
    cs = CandleSeries.from_ohlcv(ohlcv, has_timestamp=True)
    ad = StockDataAdapter(features=_CAPSTONE_FEATURES, frac_thresh=frac_thresh)
    matrix = ad.transform(cs)
    names = list(ad.feature_names)
    if target in ("volatility", "abs_return"):           # prepend the chosen target as col 0
        r = matrix[:, 0]
        col0 = _causal_vol(r) if target == "volatility" else np.abs(r)
        matrix = np.column_stack([col0, matrix])
        names = [("realized_vol" if target == "volatility" else "abs_return")] + names
    elif target != "return":
        raise ValueError(f"unknown target {target!r}; use return|volatility|abs_return")
    return {"symbol": symbol, "timeframe": timeframe, "exchange": exchange, "source": source,
            "n_candles": int(len(ohlcv)), "last_close": float(ohlcv[-1, 4]),
            "target": target, "matrix": matrix, "feature_names": names}


def run_crypto_capstone(symbol: str = "BTC/USD", timeframe: str = "1h", limit: int = 600,
                        exchange: str = "kraken", prefer_live: bool = True,
                        target: str = "return", **capstone_kw) -> Dict[str, object]:
    """Load the symbol, then run the domain-agnostic freeze→forward-test capstone."""
    from ..capstone import run_capstone
    d = load_symbol(symbol, timeframe, limit, exchange, prefer_live=prefer_live, target=target)
    meta = {k: d[k] for k in ("symbol", "timeframe", "exchange", "source", "n_candles",
                              "last_close", "target", "feature_names")}
    return run_capstone(d["matrix"], meta=meta, **capstone_kw)


def run_crypto_sweep(symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
                     timeframes=("1h", "4h"), exchange: str = "kraken",
                     prefer_live: bool = True, target: str = "return",
                     **capstone_kw) -> Dict[str, object]:
    """Robustness sweep: run the frozen forward-test across several symbols ×
    timeframes (proxy regimes). One symbol is one data point; the aggregate says
    whether the distributional skill is REAL STRUCTURE or luck."""
    keep = ("symbol", "timeframe", "source", "n_holdout", "history_skill", "forward_skill",
            "forward_dsr", "forward_calibrated", "net_directional_hit",
            "baseline_directional_hit", "verdict")
    runs: List[Dict[str, object]] = []
    for sym in symbols:
        for tf in timeframes:
            try:
                r = run_crypto_capstone(sym, tf, exchange=exchange, prefer_live=prefer_live,
                                        target=target, **capstone_kw)
                runs.append({k: r.get(k) for k in keep})
            except Exception as e:                       # one symbol failing must not kill the sweep
                runs.append({"symbol": sym, "timeframe": tf, "error": f"{type(e).__name__}: {e}"})
    ok = [r for r in runs if "error" not in r and r.get("forward_skill") is not None]

    def _med(key, pred=lambda r: True):
        vals = [float(r[key]) for r in ok if pred(r) and r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    agg = {
        "target": target, "n_runs": len(ok), "n_failed": len(runs) - len(ok),
        "n_pass": sum(1 for r in ok if r["verdict"] == "PASS"),
        "n_skill_positive": sum(1 for r in ok if r["forward_skill"] > 0),
        "n_dsr_significant": sum(1 for r in ok if (r["forward_dsr"] or 0) >= 0.95),
        "n_calibrated": sum(1 for r in ok if r["forward_calibrated"]),
        "median_forward_skill": _med("forward_skill"),
        "median_forward_dsr": _med("forward_dsr"),
        "median_net_dir_hit": _med("net_directional_hit"),
        "median_baseline_dir_hit": _med("baseline_directional_hit"),
    }
    return {"runs": runs, "aggregate": agg}


__all__ = ["ccxt_available", "fetch_ohlcv", "synthetic_ohlcv", "load_symbol",
           "run_crypto_capstone", "run_crypto_sweep"]
