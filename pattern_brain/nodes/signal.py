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
