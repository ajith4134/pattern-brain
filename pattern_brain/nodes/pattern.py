"""Pattern layer (Block 18, layer 3): group similar structures (clustering)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from ..belief import Belief
from ..node import Node
from ..registry import register


def _silhouette_conf(X: np.ndarray, labels) -> float:
    """Map silhouette score from [-1, 1] to a [0, 1] confidence; 0 if undefined."""
    try:
        uniq = set(int(l) for l in labels)
        uniq.discard(-1)  # DBSCAN noise label
        if 1 < len(uniq) < len(labels):
            return float((silhouette_score(X, labels) + 1) / 2)
    except Exception:
        pass
    return 0.0


@register
class KMeansNode(Node):
    layer = "pattern"
    node_type = "kmeans"

    def __init__(self, n_clusters: int = 3, random_state: int = 0, **kw):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kw)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._m = None

    def _fit(self, X, y=None):
        k = max(1, min(self.n_clusters, X.shape[0]))
        self._m = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)

    def _predict(self, X: np.ndarray) -> Belief:
        labels = self._m.predict(X)
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(len(set(labels))),
                       "centers": self._m.cluster_centers_.tolist()},
                      _silhouette_conf(X, labels), self.name)


@register
class DBSCANNode(Node):
    """Density-based clustering (transductive: clusters the given window)."""
    layer = "pattern"
    node_type = "dbscan"

    def __init__(self, eps: float = 0.8, min_samples: int = 4, **kw):
        super().__init__(eps=eps, min_samples=min_samples, **kw)
        self.eps = eps
        self.min_samples = min_samples

    def _predict(self, X: np.ndarray) -> Belief:
        labels = DBSCAN(eps=self.eps,
                        min_samples=min(self.min_samples, X.shape[0])).fit_predict(X)
        n = len(set(labels) - {-1})
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(n),
                       "n_noise": int((labels == -1).sum())},
                      _silhouette_conf(X, labels), self.name)


@register
class HDBSCANNode(Node):
    """Hierarchical density-based clustering (named in the Block 3 starter set);
    finds variable-density clusters without a preset cluster count."""
    layer = "pattern"
    node_type = "hdbscan"

    def __init__(self, min_cluster_size: int = 5, **kw):
        super().__init__(min_cluster_size=min_cluster_size, **kw)
        self.min_cluster_size = min_cluster_size

    def _predict(self, X: np.ndarray) -> Belief:
        mcs = max(2, min(self.min_cluster_size, X.shape[0]))
        labels = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        n = len(set(labels) - {-1})
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(n),
                       "n_noise": int((labels == -1).sum())},
                      _silhouette_conf(X, labels), self.name)


@register
class GaussianMixtureNode(Node):
    """Soft (probabilistic) clustering; confidence = mean responsibility peak."""
    layer = "pattern"
    node_type = "gmm"

    def __init__(self, n_components: int = 3, random_state: int = 0, **kw):
        super().__init__(n_components=n_components, random_state=random_state, **kw)
        self.n_components = n_components
        self.random_state = random_state
        self._m = None

    def _fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[0]))
        self._m = GaussianMixture(n_components=k, random_state=self.random_state).fit(X)

    def _predict(self, X: np.ndarray) -> Belief:
        labels = self._m.predict(X)
        proba = self._m.predict_proba(X)
        return Belief("density",
                      {"labels": labels.tolist(), "max_proba": proba.max(axis=1).tolist(),
                       "bic": float(self._m.bic(X))},
                      float(proba.max(axis=1).mean()), self.name)


@register
class AgglomerativeNode(Node):
    """Hierarchical agglomerative clustering (transductive)."""
    layer = "pattern"
    node_type = "agglomerative"

    def __init__(self, n_clusters: int = 3, **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        self.n_clusters = n_clusters

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(1, min(self.n_clusters, X.shape[0]))
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — more clustering algorithms (Block 42 clustering cat).
# --------------------------------------------------------------------------
from sklearn.cluster import MeanShift, SpectralClustering, Birch, OPTICS  # noqa: E402


@register
class MeanShiftNode(Node):
    """Mean-shift clustering (no preset cluster count; bandwidth-driven)."""
    layer = "pattern"
    node_type = "mean_shift"

    def _predict(self, X: np.ndarray) -> Belief:
        try:
            labels = MeanShift(bin_seeding=True).fit_predict(X)
        except Exception:
            labels = np.zeros(X.shape[0], dtype=int)
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)


@register
class SpectralClusteringNode(Node):
    """Spectral clustering on an RBF affinity graph (transductive)."""
    layer = "pattern"
    node_type = "spectral_clustering"

    def __init__(self, n_clusters: int = 3, **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        self.n_clusters = n_clusters

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(2, min(self.n_clusters, X.shape[0] - 1))
        try:
            labels = SpectralClustering(n_clusters=k, affinity="rbf",
                                        assign_labels="discretize",
                                        random_state=0).fit_predict(X)
        except Exception:
            labels = np.zeros(X.shape[0], dtype=int)
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)


@register
class BirchNode(Node):
    """BIRCH incremental clustering (memory-efficient on large streams)."""
    layer = "pattern"
    node_type = "birch"

    def __init__(self, n_clusters: int = 3, **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        self.n_clusters = n_clusters

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(1, min(self.n_clusters, X.shape[0]))
        labels = Birch(n_clusters=k).fit_predict(X)
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)


@register
class OPTICSNode(Node):
    """OPTICS density clustering (variable-density, DBSCAN generalization)."""
    layer = "pattern"
    node_type = "optics"

    def __init__(self, min_samples: int = 5, **kw):
        super().__init__(min_samples=min_samples, **kw)
        self.min_samples = min_samples

    def _predict(self, X: np.ndarray) -> Belief:
        ms = max(2, min(self.min_samples, X.shape[0] - 1))
        labels = OPTICS(min_samples=ms).fit_predict(X)
        n = len(set(labels) - {-1})
        return Belief("cluster",
                      {"labels": labels.tolist(), "n_clusters": int(n),
                       "n_noise": int((labels == -1).sum())},
                      _silhouette_conf(X, labels), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — scalable k-means variants.
# --------------------------------------------------------------------------
from sklearn.cluster import MiniBatchKMeans, BisectingKMeans  # noqa: E402


@register
class MiniBatchKMeansNode(Node):
    """Mini-batch K-Means (scales to large streams; approximate K-Means)."""
    layer = "pattern"
    node_type = "minibatch_kmeans"

    def __init__(self, n_clusters: int = 3, random_state: int = 0, **kw):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kw)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(1, min(self.n_clusters, X.shape[0]))
        labels = MiniBatchKMeans(n_clusters=k, n_init=3,
                                 random_state=self.random_state).fit_predict(X)
        return Belief("cluster", {"labels": labels.tolist(),
                                  "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)


@register
class BisectingKMeansNode(Node):
    """Bisecting (divisive hierarchical) K-Means."""
    layer = "pattern"
    node_type = "bisecting_kmeans"

    def __init__(self, n_clusters: int = 3, random_state: int = 0, **kw):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **kw)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _predict(self, X: np.ndarray) -> Belief:
        k = max(2, min(self.n_clusters, X.shape[0]))
        labels = BisectingKMeans(n_clusters=k, random_state=self.random_state).fit_predict(X)
        return Belief("cluster", {"labels": labels.tolist(),
                                  "n_clusters": int(len(set(labels)))},
                      _silhouette_conf(X, labels), self.name)
