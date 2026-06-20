"""Noise layer (Block 18, layer 2): separate signal from noise / flag outliers.

Block 18's source note: signal-processing denoisers are "often better than ML for
noise removal" — so this layer mixes a Savitzky-Golay filter and PCA low-rank
reconstruction (denoisers) with two anomaly detectors.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class SavitzkyGolayNode(Node):
    """Savitzky-Golay smoothing denoiser; confidence = signal-to-noise ratio."""
    layer = "noise"
    node_type = "savgol_denoise"
    is_transformer = True

    def __init__(self, window: int = 7, poly: int = 2, **kw):
        super().__init__(window=window, poly=poly, **kw)
        self.window = window
        self.poly = poly

    def _transform(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        w = min(self.window, T if T % 2 == 1 else T - 1)
        if w < 3:
            return X.copy()
        if w % 2 == 0:
            w -= 1
        p = min(self.poly, w - 1)
        return savgol_filter(X, w, p, axis=0)

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        resid = X - clean
        snr = float(np.var(clean) / (np.var(resid) + 1e-9))
        return Belief("denoised",
                      {"residual": float(np.mean(np.abs(resid))), "snr": snr,
                       "series": clean.tolist()},
                      float(snr / (1 + snr)), self.name)


@register
class PCADenoiseNode(Node):
    """Low-rank PCA reconstruction denoiser; confidence = explained variance."""
    layer = "noise"
    node_type = "pca_denoise"
    is_transformer = True

    def __init__(self, n_components: int = 1, **kw):
        super().__init__(n_components=n_components, **kw)
        self.n_components = n_components
        self._pca = None

    def _fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[1], X.shape[0]))
        self._pca = PCA(n_components=k).fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.inverse_transform(self._pca.transform(X))

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        evr = float(self._pca.explained_variance_ratio_.sum())
        return Belief("denoised",
                      {"residual": float(np.mean(np.abs(X - clean))), "explained_var": evr,
                       "series": clean.tolist()},
                      evr, self.name)


@register
class ZScoreAnomalyNode(Node):
    """Per-timestep z-score outlier flags across features."""
    layer = "noise"
    node_type = "zscore_anomaly"

    def __init__(self, threshold: float = 3.0, **kw):
        super().__init__(threshold=threshold, **kw)
        self.threshold = threshold

    def _predict(self, X: np.ndarray) -> Belief:
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-9
        score = np.abs((X - mu) / sd).max(axis=1)
        flags = (score > self.threshold).astype(int)
        return Belief("anomaly",
                      {"n_anomalies": int(flags.sum()), "fraction": float(flags.mean()),
                       "scores": score.tolist(), "flags": flags.tolist()},
                      float(min(1.0, score.max() / (self.threshold * 2))), self.name)


@register
class IsolationForestNode(Node):
    """Isolation Forest anomaly detection (sklearn)."""
    layer = "noise"
    node_type = "isolation_forest"

    def __init__(self, contamination="auto", random_state: int = 0, **kw):
        super().__init__(contamination=contamination, random_state=random_state, **kw)
        self.contamination = contamination
        self.random_state = random_state
        self._m = None

    def _fit(self, X, y=None):
        self._m = IsolationForest(contamination=self.contamination,
                                  random_state=self.random_state).fit(X)

    def _predict(self, X: np.ndarray) -> Belief:
        pred = self._m.predict(X)            # -1 outlier, +1 inlier
        scores = -self._m.score_samples(X)   # higher = more anomalous
        flags = (pred == -1).astype(int)
        frac = float(flags.mean())
        return Belief("anomaly",
                      {"n_anomalies": int(flags.sum()), "fraction": frac,
                       "scores": scores.tolist(), "flags": flags.tolist()},
                      float(min(1.0, frac * 3)), self.name)
