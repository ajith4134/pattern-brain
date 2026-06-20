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
