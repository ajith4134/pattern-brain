"""Signal layer (Block 18, layer 1): extract structure from the raw sequence.

All nodes operate on a generic (T, D) array — the "signal" is the cross-feature
mean unless a node says otherwise. Zero stock-specific assumptions (Rule 23).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class FFTNode(Node):
    """Dominant-frequency / spectral-concentration analysis via the real FFT."""
    layer = "signal"
    node_type = "fft"

    def _predict(self, X: np.ndarray) -> Belief:
        sig = X.mean(axis=1)
        sig = sig - sig.mean()
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig))
        if len(spec) <= 1 or spec[1:].sum() == 0:
            return Belief("spectral", {"dominant_freq": 0.0, "energy": 0.0}, 0.0, self.name)
        k = int(np.argmax(spec[1:]) + 1)
        power = spec[1:] ** 2
        concentration = float(spec[k] ** 2 / (power.sum() + 1e-12))
        return Belief(
            "spectral",
            {"dominant_freq": float(freqs[k]), "energy": float(spec[k] ** 2),
             "spectrum": spec.tolist()},
            concentration, self.name,
        )


@register
class MovingAverageNode(Node):
    """Smoothing transformer + trend-direction belief from the smoothed slope."""
    layer = "signal"
    node_type = "moving_average"
    is_transformer = True

    def __init__(self, window: int = 5, **kw):
        super().__init__(window=window, **kw)
        self.window = window

    def _transform(self, X: np.ndarray) -> np.ndarray:
        w = max(1, min(self.window, X.shape[0]))
        kernel = np.ones(w) / w
        return np.column_stack([np.convolve(X[:, j], kernel, mode="same")
                                for j in range(X.shape[1])])

    def _predict(self, X: np.ndarray) -> Belief:
        sm = self._transform(X)
        m = sm.mean(axis=1)
        slope = float(np.polyfit(np.arange(len(m)), m, 1)[0]) if len(m) > 1 else 0.0
        direction = "up" if slope > 0 else ("down" if slope < 0 else "flat")
        return Belief("signal", {"direction": direction, "slope": slope,
                                 "series": sm.tolist()}, float(np.tanh(abs(slope))), self.name)


@register
class DifferenceNode(Node):
    """First-difference transformer (generic 'returns'-like) + change-magnitude belief."""
    layer = "signal"
    node_type = "difference"
    is_transformer = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        d = np.diff(X, axis=0)
        return np.vstack([np.zeros((1, X.shape[1])), d])

    def _predict(self, X: np.ndarray) -> Belief:
        d = self._transform(X)
        vol = float(np.mean(np.abs(d)))
        return Belief("signal", {"mean_abs_change": vol, "series": d.tolist()},
                      float(np.tanh(vol)), self.name)


@register
class HilbertEnvelopeNode(Node):
    """Analytic-signal amplitude envelope (instantaneous amplitude) via Hilbert transform."""
    layer = "signal"
    node_type = "hilbert_envelope"
    is_transformer = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.abs(hilbert(X, axis=0))

    def _predict(self, X: np.ndarray) -> Belief:
        env = self._transform(X)
        amp = float(env.mean())
        return Belief("signal", {"mean_envelope": amp, "series": env.tolist()},
                      float(np.tanh(amp)), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — light-stack expansion: more signal-processing nodes.
# --------------------------------------------------------------------------
from scipy.signal import detrend as _scipy_detrend, butter, filtfilt, welch  # noqa: E402


@register
class DetrendNode(Node):
    """Remove the linear trend from each feature (stationarity via detrending)."""
    layer = "signal"
    node_type = "detrend"
    is_transformer = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] < 2:
            return X.copy()
        return _scipy_detrend(X, axis=0, type="linear")

    def _predict(self, X: np.ndarray) -> Belief:
        out = self._transform(X)
        removed = float(np.mean(np.abs(X - out)))
        return Belief("signal",
                      {"series": out.tolist(), "mean_abs_change": removed},
                      float(np.tanh(removed)), self.name)


@register
class ButterLowpassNode(Node):
    """Zero-phase Butterworth low-pass filter (smooths out high-frequency noise)."""
    layer = "signal"
    node_type = "butter_lowpass"
    is_transformer = True

    def __init__(self, order: int = 3, cutoff: float = 0.2, **kw):
        super().__init__(order=order, cutoff=cutoff, **kw)
        self.order = order
        self.cutoff = cutoff

    def _transform(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        b, a = butter(self.order, min(0.99, max(0.01, self.cutoff)), btype="low")
        padlen = 3 * max(len(a), len(b))
        if T <= padlen:
            return X.copy()
        return filtfilt(b, a, X, axis=0)

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        resid = X - clean
        snr = float(np.var(clean) / (np.var(resid) + 1e-9))
        return Belief("signal",
                      {"series": clean.tolist(), "mean_abs_change": float(np.mean(np.abs(resid)))},
                      float(snr / (1 + snr)), self.name)


@register
class WelchPSDNode(Node):
    """Welch power-spectral-density estimate of feature 0 (smoother than a raw FFT)."""
    layer = "signal"
    node_type = "welch_psd"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0]
        nper = int(min(len(x), max(8, len(x) // 4)))
        freqs, psd = welch(x, nperseg=nper)
        if len(psd) > 1 and psd[1:].sum() > 0:
            k = int(np.argmax(psd[1:]) + 1)
            dom = float(freqs[k])
            conc = float(psd[k] / (psd.sum() + 1e-12))
        else:
            dom, conc = 0.0, 0.0
        return Belief("spectral",
                      {"dominant_freq": dom, "energy": float(psd.sum()),
                       "spectrum": psd.tolist()},
                      conc, self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — more transforms + a periodogram.
# --------------------------------------------------------------------------
from scipy.signal import periodogram as _periodogram  # noqa: E402


@register
class ZScoreNormalizeNode(Node):
    """Standardize each feature to zero-mean/unit-variance (a generic rescaler)."""
    layer = "signal"
    node_type = "zscore_normalize"
    is_transformer = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        return (X - mu) / sd

    def _predict(self, X: np.ndarray) -> Belief:
        out = self._transform(X)
        return Belief("signal", {"series": out.tolist(),
                                 "mean_abs_change": float(np.mean(np.abs(out - X)))},
                      0.5, self.name)


@register
class EWMANode(Node):
    """Exponentially-weighted moving average (recency-weighted smoother)."""
    layer = "signal"
    node_type = "ewma"
    is_transformer = True

    def __init__(self, alpha: float = 0.3, **kw):
        super().__init__(alpha=alpha, **kw)
        self.alpha = alpha

    def _transform(self, X: np.ndarray) -> np.ndarray:
        a = self.alpha
        out = np.empty_like(X, dtype=float)
        out[0] = X[0]
        for t in range(1, X.shape[0]):
            out[t] = a * X[t] + (1 - a) * out[t - 1]
        return out

    def _predict(self, X: np.ndarray) -> Belief:
        out = self._transform(X)
        resid = float(np.mean(np.abs(X - out)))
        snr = float(np.var(out) / (np.var(X - out) + 1e-9))
        return Belief("signal", {"series": out.tolist(), "mean_abs_change": resid},
                      float(snr / (1 + snr)), self.name)


@register
class CumSumNode(Node):
    """Cumulative sum (integrate the sequence) — the inverse of differencing."""
    layer = "signal"
    node_type = "cumsum"
    is_transformer = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.cumsum(X, axis=0)

    def _predict(self, X: np.ndarray) -> Belief:
        out = self._transform(X)
        return Belief("signal", {"series": out.tolist(),
                                 "mean_abs_change": float(np.mean(np.abs(np.diff(out, axis=0)))) if X.shape[0] > 1 else 0.0},
                      0.5, self.name)


@register
class PeriodogramNode(Node):
    """Classical periodogram power-spectral estimate of feature 0."""
    layer = "signal"
    node_type = "periodogram"

    def _predict(self, X: np.ndarray) -> Belief:
        f, pxx = _periodogram(X[:, 0])
        if len(pxx) > 1 and pxx[1:].sum() > 0:
            k = int(np.argmax(pxx[1:]) + 1)
            dom = float(f[k]); conc = float(pxx[k] / (pxx.sum() + 1e-12))
        else:
            dom, conc = 0.0, 0.0
        return Belief("spectral", {"dominant_freq": dom, "energy": float(pxx.sum()),
                                   "spectrum": pxx.tolist()}, conc, self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — classical filters + spectral measures.
# --------------------------------------------------------------------------
from scipy.ndimage import median_filter as _median_filter, gaussian_filter1d  # noqa: E402
from scipy.signal import wiener as _wiener  # noqa: E402


@register
class MedianFilterNode(Node):
    """Median filter — removes impulsive (spike) noise while keeping edges."""
    layer = "signal"
    node_type = "median_filter"
    is_transformer = True

    def __init__(self, window: int = 5, **kw):
        super().__init__(window=window, **kw)
        self.window = window

    def _transform(self, X: np.ndarray) -> np.ndarray:
        w = max(1, min(self.window, X.shape[0]))
        return _median_filter(X, size=(w, 1))

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        resid = float(np.mean(np.abs(X - clean)))
        snr = float(np.var(clean) / (np.var(X - clean) + 1e-9))
        return Belief("denoised", {"series": clean.tolist(), "residual": resid},
                      float(snr / (1 + snr)), self.name)


@register
class GaussianSmoothNode(Node):
    """Gaussian-kernel smoothing along time."""
    layer = "signal"
    node_type = "gaussian_smooth"
    is_transformer = True

    def __init__(self, sigma: float = 2.0, **kw):
        super().__init__(sigma=sigma, **kw)
        self.sigma = sigma

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(X, self.sigma, axis=0, mode="nearest")

    def _predict(self, X: np.ndarray) -> Belief:
        out = self._transform(X)
        return Belief("signal", {"series": out.tolist(),
                                 "mean_abs_change": float(np.mean(np.abs(X - out)))},
                      0.5, self.name)


@register
class WienerDenoiseNode(Node):
    """Wiener adaptive denoising filter (per feature)."""
    layer = "signal"
    node_type = "wiener_denoise"
    is_transformer = True

    def __init__(self, window: int = 5, **kw):
        super().__init__(window=window, **kw)
        self.window = window

    def _transform(self, X: np.ndarray) -> np.ndarray:
        w = max(1, min(self.window, X.shape[0]))
        cols = []
        for j in range(X.shape[1]):
            c = _wiener(X[:, j], mysize=w)
            cols.append(np.nan_to_num(c, nan=X[:, j].mean()))
        return np.column_stack(cols)

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        resid = float(np.mean(np.abs(X - clean)))
        snr = float(np.var(clean) / (np.var(X - clean) + 1e-9))
        return Belief("denoised", {"series": clean.tolist(), "residual": resid},
                      float(snr / (1 + snr)), self.name)


@register
class HodrickPrescottNode(Node):
    """Hodrick-Prescott trend filter (separates slow trend from cycle)."""
    layer = "signal"
    node_type = "hp_filter"
    is_transformer = True

    def __init__(self, lam: float = 100.0, **kw):
        super().__init__(lam=lam, **kw)
        self.lam = lam

    def _trend(self, x: np.ndarray) -> np.ndarray:
        T = len(x)
        if T < 3:
            return x.copy()
        I = np.eye(T)
        D = np.diff(I, n=2, axis=0)          # (T-2, T) second-difference operator
        return np.linalg.solve(I + self.lam * (D.T @ D), x)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self._trend(X[:, j]) for j in range(X.shape[1])])

    def _predict(self, X: np.ndarray) -> Belief:
        trend = self._transform(X)
        return Belief("signal", {"series": trend.tolist(),
                                 "mean_abs_change": float(np.mean(np.abs(X - trend)))},
                      0.5, self.name)


@register
class SpectralEntropyNode(Node):
    """Spectral entropy of feature 0 — flatness of the power spectrum (a
    randomness/complexity measure)."""
    layer = "signal"
    node_type = "spectral_entropy"

    def _predict(self, X: np.ndarray) -> Belief:
        f, pxx = _periodogram(X[:, 0])
        p = pxx / (pxx.sum() + 1e-12)
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / (np.log(len(p)) + 1e-12)) if len(p) > 1 else 0.0
        k = int(np.argmax(pxx[1:]) + 1) if len(pxx) > 1 and pxx[1:].sum() > 0 else 0
        dom = float(f[k]) if k else 0.0
        return Belief("spectral", {"dominant_freq": dom, "energy": float(pxx.sum()),
                                   "spectrum": pxx.tolist(), "entropy": ent},
                      float(1.0 - ent), self.name)


@register
class AutocorrPeriodNode(Node):
    """Dominant period of feature 0 via the autocorrelation function's first peak."""
    layer = "signal"
    node_type = "autocorr_period"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0] - X[:, 0].mean()
        n = len(x)
        ac = np.correlate(x, x, mode="full")[n - 1:]
        ac = ac / (ac[0] + 1e-12)
        period, conf = 0.0, 0.0
        if len(ac) > 2:
            d = np.diff(ac)
            for i in range(1, len(d)):
                if d[i - 1] > 0 and d[i] <= 0:    # first local maximum
                    period = float(i); conf = float(max(0.0, ac[i])); break
        dom = 1.0 / period if period > 0 else 0.0
        return Belief("spectral", {"dominant_freq": dom, "energy": float((x ** 2).sum()),
                                   "spectrum": ac.tolist(), "period": period},
                      conf, self.name)
