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


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — more anomaly detectors + dimensionality-reduction
# reconstruction denoisers (Block 42 cats: anomaly detection, dim-reduction).
# --------------------------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402
from sklearn.covariance import EllipticEnvelope  # noqa: E402
from sklearn.decomposition import FastICA, TruncatedSVD, KernelPCA  # noqa: E402


def _anomaly_belief(flags, scores, name):
    flags = np.asarray(flags).astype(int)
    frac = float(flags.mean())
    return Belief("anomaly",
                  {"n_anomalies": int(flags.sum()), "fraction": frac,
                   "scores": np.asarray(scores, float).tolist(),
                   "flags": flags.tolist()},
                  float(min(1.0, frac * 3)), name)


@register
class LOFNode(Node):
    """Local Outlier Factor anomaly detection (density-deviation based)."""
    layer = "noise"
    node_type = "local_outlier_factor"

    def __init__(self, n_neighbors: int = 20, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        self.n_neighbors = n_neighbors

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(1, min(self.n_neighbors, X.shape[0] - 1))
        m = LocalOutlierFactor(n_neighbors=k)
        pred = m.fit_predict(X)
        scores = -m.negative_outlier_factor_
        return _anomaly_belief(pred == -1, scores, self.name)


@register
class OneClassSVMNode(Node):
    """One-Class SVM novelty/anomaly detection."""
    layer = "noise"
    node_type = "one_class_svm"

    def __init__(self, nu: float = 0.1, **kw):
        super().__init__(nu=nu, **kw)
        self.nu = nu
        self._m = None

    def _fit(self, X, y=None):
        self._m = OneClassSVM(nu=self.nu, gamma="scale").fit(X)

    def _predict(self, X: np.ndarray) -> Belief:
        pred = self._m.predict(X)
        scores = -self._m.score_samples(X)
        return _anomaly_belief(pred == -1, scores, self.name)


@register
class EllipticEnvelopeNode(Node):
    """Gaussian elliptic-envelope (robust covariance) outlier detection."""
    layer = "noise"
    node_type = "elliptic_envelope"

    def __init__(self, contamination: float = 0.1, random_state: int = 0, **kw):
        super().__init__(contamination=contamination, random_state=random_state, **kw)
        self.contamination = contamination
        self.random_state = random_state
        self._m = None

    def _fit(self, X, y=None):
        self._m = EllipticEnvelope(contamination=self.contamination,
                                   random_state=self.random_state,
                                   support_fraction=1.0).fit(X)

    def _predict(self, X: np.ndarray) -> Belief:
        pred = self._m.predict(X)
        scores = -self._m.score_samples(X)
        return _anomaly_belief(pred == -1, scores, self.name)


class _ReconDenoiser(Node):
    """Base for dimensionality-reduction denoisers: project to a low-rank space
    and reconstruct; the residual is the removed noise (emits 'denoised')."""
    layer = "noise"
    is_transformer = True

    def __init__(self, n_components: int = 1, **kw):
        super().__init__(n_components=n_components, **kw)
        self.n_components = n_components
        self._m = None

    def _make(self, k):  # pragma: no cover - overridden
        raise NotImplementedError

    def _fit(self, X, y=None):
        # Low-rank denoising of a single channel is a no-op (and some backends,
        # e.g. TruncatedSVD, require >=2 features) -> passthrough when D < 2.
        if X.shape[1] < 2:
            self._m = None
            return
        k = max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1 if X.shape[0] > 1 else 1))
        self._m = self._make(max(1, k)).fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._m is None:
            return X.copy()
        return self._m.inverse_transform(self._m.transform(X))

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        resid = float(np.mean(np.abs(X - clean)))
        denom = float(np.mean(np.abs(X))) + 1e-9
        return Belief("denoised",
                      {"series": clean.tolist(), "residual": resid},
                      float(max(0.0, 1.0 - resid / denom)), self.name)


@register
class ICADenoiseNode(_ReconDenoiser):
    """Independent Component Analysis low-rank reconstruction denoiser."""
    node_type = "ica_denoise"

    def _make(self, k):
        return FastICA(n_components=k, random_state=0, max_iter=500, whiten="unit-variance")


@register
class SVDDenoiseNode(_ReconDenoiser):
    """Truncated-SVD low-rank reconstruction denoiser."""
    node_type = "svd_denoise"

    def _make(self, k):
        kk = max(1, min(k, 1_000))
        return TruncatedSVD(n_components=kk, random_state=0)


@register
class KernelPCADenoiseNode(_ReconDenoiser):
    """Kernel-PCA reconstruction denoiser (non-linear low-rank)."""
    node_type = "kernel_pca_denoise"

    def _make(self, k):
        return KernelPCA(n_components=k, kernel="rbf", fit_inverse_transform=True,
                         gamma=None, random_state=0)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — more statistical anomaly detectors.
# --------------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors  # noqa: E402


@register
class MADAnomalyNode(Node):
    """Robust median-absolute-deviation outlier flags (per-step, across features)."""
    layer = "noise"
    node_type = "mad_anomaly"

    def __init__(self, threshold: float = 3.5, **kw):
        super().__init__(threshold=threshold, **kw)
        self.threshold = threshold

    def _predict(self, X: np.ndarray) -> Belief:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) + 1e-9
        score = (0.6745 * np.abs(X - med) / mad).max(axis=1)
        flags = (score > self.threshold).astype(int)
        return _anomaly_belief(flags, score, self.name)


@register
class IQRAnomalyNode(Node):
    """Interquartile-range (Tukey fence) outlier flags."""
    layer = "noise"
    node_type = "iqr_anomaly"

    def __init__(self, k: float = 1.5, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = (q3 - q1) + 1e-9
        lo, hi = q1 - self.k * iqr, q3 + self.k * iqr
        out = (X < lo) | (X > hi)
        flags = out.any(axis=1).astype(int)
        score = np.maximum((lo - X) / iqr, (X - hi) / iqr).max(axis=1)
        score = np.clip(score, 0, None)
        return _anomaly_belief(flags, score, self.name)


@register
class MahalanobisAnomalyNode(Node):
    """Mahalanobis-distance outlier flags (accounts for feature covariance)."""
    layer = "noise"
    node_type = "mahalanobis_anomaly"

    def __init__(self, quantile: float = 0.975, **kw):
        super().__init__(quantile=quantile, **kw)
        self.quantile = quantile

    def _predict(self, X: np.ndarray) -> Belief:
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov)
        try:
            inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            inv = np.eye(X.shape[1])
        d = X - mu
        score = np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))
        thr = float(np.quantile(score, self.quantile))
        flags = (score > thr).astype(int)
        return _anomaly_belief(flags, score, self.name)


@register
class KNNDistanceAnomalyNode(Node):
    """k-NN mean-distance outlier score (distance-based anomaly detection)."""
    layer = "noise"
    node_type = "knn_distance_anomaly"

    def __init__(self, n_neighbors: int = 5, quantile: float = 0.95, **kw):
        super().__init__(n_neighbors=n_neighbors, quantile=quantile, **kw)
        self.n_neighbors = n_neighbors
        self.quantile = quantile

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(1, min(self.n_neighbors, X.shape[0] - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, _ = nn.kneighbors(X)
        score = dist[:, 1:].mean(axis=1)  # exclude self (distance 0)
        thr = float(np.quantile(score, self.quantile))
        flags = (score > thr).astype(int)
        return _anomaly_belief(flags, score, self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — reconstruction- and histogram-based anomaly.
# --------------------------------------------------------------------------
@register
class PCAResidualAnomalyNode(Node):
    """Flag rows poorly reconstructed by a low-rank PCA model (subspace anomaly)."""
    layer = "noise"
    node_type = "pca_residual_anomaly"

    def __init__(self, n_components: int = 1, quantile: float = 0.95, **kw):
        super().__init__(n_components=n_components, quantile=quantile, **kw)
        self.n_components = n_components
        self.quantile = quantile

    def _predict(self, X: np.ndarray) -> Belief:
        if X.shape[1] < 2:
            score = np.abs(X[:, 0] - X[:, 0].mean())
        else:
            k = max(1, min(self.n_components, X.shape[1] - 1))
            p = PCA(n_components=k).fit(X)
            recon = p.inverse_transform(p.transform(X))
            score = np.abs(X - recon).mean(axis=1)
        thr = float(np.quantile(score, self.quantile))
        flags = (score > thr).astype(int)
        return _anomaly_belief(flags, score, self.name)


@register
class HistogramAnomalyNode(Node):
    """Histogram-based outlier score (HBOS-style): low-density bins per feature."""
    layer = "noise"
    node_type = "histogram_anomaly"

    def __init__(self, bins: int = 10, quantile: float = 0.95, **kw):
        super().__init__(bins=bins, quantile=quantile, **kw)
        self.bins = bins
        self.quantile = quantile

    def _predict(self, X: np.ndarray) -> Belief:
        score = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            hist, edges = np.histogram(X[:, j], bins=min(self.bins, X.shape[0]), density=True)
            idx = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(hist) - 1)
            dens = hist[idx] + 1e-9
            score += -np.log(dens)
        thr = float(np.quantile(score, self.quantile))
        flags = (score > thr).astype(int)
        return _anomaly_belief(flags, score, self.name)
