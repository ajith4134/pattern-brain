"""Stock-data adapter (build-tracker step 3; PLAN.md §6 Block 4 / §5 frac-diff).

THIS FILE IS THE ONE PLACE STOCK-SPECIFIC CODE IS ALLOWED TO LIVE (Rule 23 /
PLAN.md §0). Its whole job is to bend real stock-market candle data into the
generic ``(T, D)`` finite-float array the already-built model bank + Connector
Intelligence (steps 1-2) already require — never the other way around. The
dependency only ever flows adapter -> core; the core package never imports this.

Three layers, deliberately separated so a *future* domain reuses the first two
unchanged (the portability test of PLAN.md §0):

1. **Generic numeric tools** (domain-agnostic — no OHLCV anywhere):
   * :func:`frac_diff` / :func:`frac_diff_weights` — López de Prado's fixed-width
     fractional differentiation (AFML ch. 5). The one Feature-Factory technique
     on record (§5/§6): makes a series stationary while preserving far more memory
     than plain returns / integer differencing.
   * :func:`align_to_grid` / :func:`align_sources` — causal "as-of" resampling of
     irregular, asynchronous, differently-sampled timestamped series onto one
     shared regular grid. This is the concrete closure of §6 gap 7f
     (time/asynchrony handling across sources).
2. **:class:`CandleSeries`** — a validated OHLCV container (the domain shape).
3. **:class:`StockDataAdapter`** — assembles candles into the generic ``(T, D)``
   array, then hands off to the unchanged core. The *only* component that knows
   the words open/high/low/close/volume.

Light stack only (numpy), per §0b "add only what's proven needed" — no pandas
pulled in just for grid alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# Layer 1 — GENERIC numeric tools (domain-agnostic; reusable by any adapter)
# ===========================================================================

def frac_diff_weights(d: float, thresh: float = 1e-4, max_terms: int = 100_000) -> np.ndarray:
    """Fixed-width fractional-differencing weights (López de Prado, AFML ch. 5).

    ``w[0] = 1`` and ``w[k] = -w[k-1] * (d - k + 1) / k``; the expansion stops
    once ``|w[k]| < thresh``. Returns weights newest-first-aligned as
    ``[w0, w1, ...]`` (``w0`` multiplies the most recent observation).

    For ``d == 0`` this is ``[1]`` (identity); for ``d == 1`` it is ``[1, -1]``
    (a plain first difference) — the two checkable anchor cases.
    """
    if d < 0:
        raise ValueError(f"fractional order d must be >= 0, got {d}")
    if not (thresh > 0):
        raise ValueError(f"thresh must be > 0, got {thresh}")
    w = [1.0]
    k = 1
    while k < max_terms:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < thresh:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w, dtype=float)


def frac_diff(series: Sequence[float], d: float, thresh: float = 1e-4) -> np.ndarray:
    """Fractionally difference a 1-D series by order ``d`` (fixed-width window).

    The first ``len(weights) - 1`` outputs are :data:`numpy.nan` (the window does
    not yet fit) — the caller is responsible for trimming them, which keeps this
    a pure, general transform with no hidden policy.
    """
    x = np.asarray(series, dtype=float).ravel()
    w = frac_diff_weights(d, thresh)
    K = w.shape[0]
    out = np.full(x.shape[0], np.nan, dtype=float)
    if K > x.shape[0]:
        return out
    wr = w[::-1]  # so wr aligns w0 with the newest sample in each window
    for t in range(K - 1, x.shape[0]):
        out[t] = float(np.dot(wr, x[t - K + 1 : t + 1]))
    return out


def align_to_grid(
    timestamps: Sequence[float],
    values: np.ndarray,
    step: float,
    *,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    fill: str = "ffill",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an irregular timestamped series onto a regular grid of ``step``.

    Domain-agnostic: ``values`` is any ``(T,)`` or ``(T, D)`` numeric array keyed
    by sorted ``timestamps``. This is the per-source half of the §6 gap-7f
    (asynchrony) fix.

    fill
        ``"ffill"`` (default) — causal "as-of": each grid point takes the most
        recent observation at or before it. **No lookahead** (the correct choice
        for a trading domain). Grid points before the first observation are NaN.
        ``"interp"`` — linear interpolation between observations; convenient but
        **non-causal** (peeks at the next observation), so only for offline use.

    Returns ``(grid, aligned)`` where ``aligned`` is ``(len(grid), D)``.
    """
    ts = np.asarray(timestamps, dtype=float).ravel()
    V = np.asarray(values, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    if ts.shape[0] != V.shape[0]:
        raise ValueError(f"timestamps ({ts.shape[0]}) and values ({V.shape[0]}) length mismatch")
    if ts.shape[0] == 0:
        raise ValueError("empty series")
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be sorted ascending")
    if not (step > 0):
        raise ValueError(f"step must be > 0, got {step}")

    g0 = ts[0] if start is None else float(start)
    g1 = ts[-1] if stop is None else float(stop)
    if g1 < g0:
        raise ValueError(f"empty grid: start {g0} > stop {g1}")
    n = int(np.floor((g1 - g0) / step + 1e-9)) + 1
    grid = g0 + step * np.arange(n)

    if fill == "ffill":
        idx = np.searchsorted(ts, grid, side="right") - 1
        out = np.full((grid.shape[0], V.shape[1]), np.nan, dtype=float)
        valid = idx >= 0
        out[valid] = V[idx[valid]]
    elif fill == "interp":
        out = np.column_stack([np.interp(grid, ts, V[:, j]) for j in range(V.shape[1])])
    else:
        raise ValueError(f"unknown fill {fill!r}; expected 'ffill' or 'interp'")
    return grid, out


def align_sources(
    sources: Sequence[Tuple[Sequence[float], np.ndarray]],
    step: float,
    *,
    fill: str = "ffill",
) -> Tuple[np.ndarray, np.ndarray]:
    """Align several differently-sampled timestamped sources onto ONE shared grid.

    Each source is ``(timestamps, values)``. The shared grid spans the common
    overlap (``max`` of starts to ``min`` of stops) so every source has data on
    it, then each is :func:`align_to_grid`-resampled and the results concatenated
    column-wise. This is the cross-source half of the §6 gap-7f (asynchrony) fix:
    sources sampled at *different rates* become a single aligned ``(T, D)`` matrix.
    """
    if not sources:
        raise ValueError("need at least one source")
    starts, stops = [], []
    for ts, _ in sources:
        ts = np.asarray(ts, dtype=float).ravel()
        if ts.shape[0] == 0:
            raise ValueError("a source has no timestamps")
        starts.append(ts[0])
        stops.append(ts[-1])
    start, stop = max(starts), min(stops)
    if stop < start:
        raise ValueError("sources do not overlap in time — no common grid")
    grid = None
    cols: List[np.ndarray] = []
    for ts, vals in sources:
        g, aligned = align_to_grid(ts, vals, step, start=start, stop=stop, fill=fill)
        grid = g if grid is None else grid
        cols.append(aligned)
    return grid, np.column_stack(cols)


# ===========================================================================
# Layer 2 — the DOMAIN shape: an OHLCV candle container
# ===========================================================================

@dataclass
class CandleSeries:
    """A validated OHLCV candle series — the stock-domain input shape.

    ``timestamp`` is optional; supply it (epoch seconds, ascending) to enable
    regular-grid resampling for irregular/gappy candles.
    """
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        arrs = {
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        }
        n = None
        for k, v in arrs.items():
            a = np.asarray(v, dtype=float).ravel()
            setattr(self, k, a)
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                raise ValueError(f"OHLCV column {k!r} has length {a.shape[0]} != {n}")
        if n == 0:
            raise ValueError("empty candle series")
        if self.timestamp is not None:
            ts = np.asarray(self.timestamp, dtype=float).ravel()
            if ts.shape[0] != n:
                raise ValueError(f"timestamp length {ts.shape[0]} != {n}")
            if np.any(np.diff(ts) < 0):
                raise ValueError("timestamps must be sorted ascending")
            self.timestamp = ts
        if np.any(self.close <= 0) or np.any(self.open <= 0):
            raise ValueError("non-positive prices — log features undefined")
        if np.any(self.volume < 0):
            raise ValueError("negative volume")

    def __len__(self) -> int:
        return self.close.shape[0]

    @classmethod
    def from_ohlcv(cls, matrix: np.ndarray, *, has_timestamp: bool = False) -> "CandleSeries":
        """Build from a 2-D matrix. Columns are ``[O, H, L, C, V]``, or
        ``[ts, O, H, L, C, V]`` when ``has_timestamp`` is True."""
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {m.shape}")
        if has_timestamp:
            if m.shape[1] < 6:
                raise ValueError("has_timestamp=True needs >=6 columns [ts,O,H,L,C,V]")
            ts, o, h, l, c, v = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5]
            return cls(o, h, l, c, v, timestamp=ts)
        if m.shape[1] < 5:
            raise ValueError("need >=5 columns [O,H,L,C,V]")
        o, h, l, c, v = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]
        return cls(o, h, l, c, v)


# ===========================================================================
# Layer 3 — the adapter: candles -> the generic (T, D) the core already wants
# ===========================================================================

# Feature builders map a CandleSeries to a 1-D column. Price-derived columns are
# made stationary (frac-diff / log-return) so nodes see a well-behaved series;
# the raw non-stationary close is available but off by default.
def _features(d: float, thresh: float) -> Dict[str, Callable[[CandleSeries], np.ndarray]]:
    return {
        # stationary-with-memory log price (the §5/§6 named technique)
        "fracdiff_log_close": lambda cs: frac_diff(np.log(cs.close), d, thresh),
        # close-to-close log return (stationary; leading NaN)
        "log_return": lambda cs: _prepend_nan(np.diff(np.log(cs.close))),
        # intrabar range as a volatility proxy (already stationary, scale-free)
        "hl_range": lambda cs: (cs.high - cs.low) / cs.close,
        # volume level and change
        "log_volume": lambda cs: np.log1p(cs.volume),
        "log_volume_change": lambda cs: _prepend_nan(np.diff(np.log1p(cs.volume))),
        # raw close — non-stationary, off by default; here for completeness
        "close": lambda cs: cs.close.astype(float),
    }


def _prepend_nan(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], a])


class StockDataAdapter:
    """Convert stock candles into the generic ``(T, D)`` finite-float array the
    model bank + Connector already consume (steps 1-2). The data bends to the
    system; the system is untouched (PLAN.md §0).

    Parameters
    ----------
    features : sequence of str
        Which feature columns to emit, left-to-right -> the D axis. Default is a
        small, stationary candle set (NOT the full ~10k-feature universal version,
        which is later; this step is "one data source — candles").
    frac_d, frac_thresh : float
        Fractional-differencing order and weight cutoff for ``fracdiff_*`` columns.
    resample_step : float, optional
        If set (and the candles carry timestamps), candles are first aligned onto
        a regular grid of this step (causal ffill) to absorb gaps/asynchrony
        before features are computed (§6 gap 7f). If None, candles are used as-is.
    fill : str
        Resampling fill mode passed to :func:`align_to_grid`.
    """

    DEFAULT_FEATURES: Tuple[str, ...] = (
        "fracdiff_log_close", "log_return", "hl_range", "log_volume",
    )

    def __init__(
        self,
        features: Sequence[str] = DEFAULT_FEATURES,
        *,
        frac_d: float = 0.4,
        frac_thresh: float = 1e-4,
        resample_step: Optional[float] = None,
        fill: str = "ffill",
    ) -> None:
        if not features:
            raise ValueError("need at least one feature")
        self._builders = _features(frac_d, frac_thresh)
        unknown = [f for f in features if f not in self._builders]
        if unknown:
            raise ValueError(f"unknown feature(s) {unknown}; known: {sorted(self._builders)}")
        self.features: Tuple[str, ...] = tuple(features)
        self.frac_d = frac_d
        self.frac_thresh = frac_thresh
        self.resample_step = resample_step
        self.fill = fill

    @property
    def feature_names(self) -> List[str]:
        return list(self.features)

    def _maybe_resample(self, cs: CandleSeries) -> CandleSeries:
        if self.resample_step is None:
            return cs
        if cs.timestamp is None:
            raise ValueError("resample_step set but candles carry no timestamp")
        cols = np.column_stack([cs.open, cs.high, cs.low, cs.close, cs.volume])
        grid, aligned = align_to_grid(cs.timestamp, cols, self.resample_step, fill=self.fill)
        # drop any leading grid rows with no observation yet (NaN under ffill)
        good = np.where(np.all(np.isfinite(aligned), axis=1))[0]
        if good.size == 0:
            raise ValueError("resampling produced no finite rows")
        aligned = aligned[good[0]:]
        grid = grid[good[0]:]
        return CandleSeries(
            open=aligned[:, 0], high=aligned[:, 1], low=aligned[:, 2],
            close=aligned[:, 3], volume=aligned[:, 4], timestamp=grid,
        )

    def transform(self, candles: CandleSeries) -> np.ndarray:
        """Candles -> finite ``(T, D)`` float array, ready for ``Node.process`` /
        ``Connector.run`` with no further handling. Leading rows that are NaN
        because of differencing/frac-diff windows are trimmed so the output meets
        the core's finite-input contract exactly."""
        if not isinstance(candles, CandleSeries):
            raise TypeError("transform() expects a CandleSeries")
        cs = self._maybe_resample(candles)
        cols = [self._builders[name](cs) for name in self.features]
        X = np.column_stack(cols)
        # trim the warm-up: keep from the first fully-finite row onward
        finite_rows = np.where(np.all(np.isfinite(X), axis=1))[0]
        if finite_rows.size == 0:
            raise ValueError(
                "no finite rows after feature build — series too short for the "
                f"frac-diff/diff warm-up (features={self.features}, d={self.frac_d})"
            )
        X = X[finite_rows[0]:]
        if not np.all(np.isfinite(X)):
            # interior NaN means a gap the warm-up trim can't fix — fail loudly
            raise ValueError("non-finite values remain inside the feature matrix")
        return np.ascontiguousarray(X, dtype=float)
