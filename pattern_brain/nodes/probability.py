"""Probability layer (Block 18, layer 5): estimate values WITH uncertainty.

Each node returns a forecast plus an explicit variance/std, so confidence is a
genuine inverse-uncertainty measure rather than a heuristic.
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import BayesianRidge

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class KalmanFilterNode(Node):
    """Per-feature constant-velocity Kalman filter; 1-step-ahead forecast + variance.
    A canonical hot-loop candidate for the C/C++ rewrite (PLAN.md §0b)."""
    layer = "probability"
    node_type = "kalman_filter"

    def __init__(self, process_var: float = 1e-3, meas_var: float = 1e-1, **kw):
        super().__init__(process_var=process_var, meas_var=meas_var, **kw)
        self.process_var = process_var
        self.meas_var = meas_var

    def _predict(self, X: np.ndarray) -> Belief:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = self.process_var * np.eye(2)
        R = np.array([[self.meas_var]])
        preds, variances = [], []
        for j in range(X.shape[1]):
            z = X[:, j]
            x = np.array([z[0], 0.0])
            P = np.eye(2)
            for t in range(len(z)):
                x = F @ x
                P = F @ P @ F.T + Q
                y = z[t] - (H @ x)[0]
                S = (H @ P @ H.T + R)[0, 0]
                K = (P @ H.T / S).ravel()
                x = x + K * y
                P = (np.eye(2) - np.outer(K, H)) @ P
            xp = F @ x
            Pp = F @ P @ F.T + Q
            preds.append(float(xp[0]))
            variances.append(float(Pp[0, 0]))
        conf = float(1.0 / (1.0 + np.mean(variances)))
        return Belief("forecast", {"next_vector": preds, "variance": variances},
                      conf, self.name)


@register
class GaussianProcessNode(Node):
    """Gaussian-process regression over the time index -> value, per feature.
    Returns a forecast and predictive std (uncertainty)."""
    layer = "probability"
    node_type = "gaussian_process"

    def __init__(self, max_points: int = 80, random_state: int = 0, **kw):
        super().__init__(max_points=max_points, random_state=random_state, **kw)
        self.max_points = max_points
        self.random_state = random_state

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        idx = np.arange(T)
        if T > self.max_points:
            sub = np.linspace(0, T - 1, self.max_points).astype(int)
        else:
            sub = idx
        kernel = ConstantKernel() * RBF(length_scale=max(1.0, T / 10)) + WhiteKernel()
        preds, stds = [], []
        for j in range(X.shape[1]):
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          random_state=self.random_state)
            gp.fit(sub.reshape(-1, 1), X[sub, j])
            m, s = gp.predict(np.array([[T]]), return_std=True)
            preds.append(float(m[0]))
            stds.append(float(s[0]))
        conf = float(1.0 / (1.0 + np.mean(stds)))
        return Belief("forecast", {"next_vector": preds, "std": stds}, conf, self.name)


@register
class BayesianRidgeNode(Node):
    """Per-feature Bayesian ridge regression on p lags; forecast + predictive std."""
    layer = "probability"
    node_type = "bayesian_ridge"

    def __init__(self, p: int = 3, **kw):
        super().__init__(p=p, **kw)
        self.p = p

    def _fit(self, X, y=None):
        p = max(1, min(self.p, X.shape[0] - 1))
        self._p = p
        self._models = []
        for j in range(X.shape[1]):
            x = X[:, j]
            rows = np.array([x[i - p:i] for i in range(p, len(x))])
            target = x[p:]
            m = BayesianRidge()
            if len(rows) >= 2:
                m.fit(rows, target)
                self._models.append(m)
            else:
                self._models.append(None)

    def _predict(self, X: np.ndarray) -> Belief:
        p = self._p
        preds, stds = [], []
        for j in range(X.shape[1]):
            m = self._models[j]
            window = X[-p:, j].reshape(1, -1)
            if m is None:
                preds.append(float(X[-1, j]))
                stds.append(float(X[:, j].std() + 1e-9))
            else:
                mean, std = m.predict(window, return_std=True)
                preds.append(float(mean[0]))
                stds.append(float(std[0]))
        conf = float(1.0 / (1.0 + np.mean(stds)))
        return Belief("forecast", {"next_vector": preds, "std": stds}, conf, self.name)


@register
class ParticleFilterNode(Node):
    """Sequential Monte Carlo (bootstrap particle filter) on the cross-feature
    mean signal; random-walk state, Gaussian likelihood, systematic resampling."""
    layer = "probability"
    node_type = "particle_filter"

    def __init__(self, n_particles: int = 300, process_var: float = 1e-2,
                 meas_var: float = 1e-1, random_state: int = 0, **kw):
        super().__init__(n_particles=n_particles, process_var=process_var,
                         meas_var=meas_var, random_state=random_state, **kw)
        self.n_particles = n_particles
        self.process_var = process_var
        self.meas_var = meas_var
        self.random_state = random_state

    def _predict(self, X: np.ndarray) -> Belief:
        rng = np.random.default_rng(self.random_state)
        z = X.mean(axis=1)
        N = self.n_particles
        p = rng.normal(z[0], 1.0, N)
        w = np.full(N, 1.0 / N)
        for t in range(len(z)):
            p = p + rng.normal(0.0, np.sqrt(self.process_var), N)
            w = w * np.exp(-0.5 * (z[t] - p) ** 2 / self.meas_var) + 1e-300
            w /= w.sum()
            if 1.0 / np.sum(w ** 2) < N / 2:  # resample on low ESS
                idx = rng.choice(N, N, p=w)
                p = p[idx]
                w = np.full(N, 1.0 / N)
        est = float(np.sum(w * p))
        var = float(np.sum(w * (p - est) ** 2))
        return Belief("forecast", {"estimate": est, "variance": var},
                      float(1.0 / (1.0 + var)), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — kernel density estimate as a probabilistic
# anomaly detector (low estimated density => outlier).
# --------------------------------------------------------------------------
import numpy as _np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402


@register
class KDEAnomalyNode(Node):
    """Gaussian kernel-density estimate; flags the lowest-density points."""
    layer = "probability"
    node_type = "kde_anomaly"

    def __init__(self, quantile: float = 0.05, **kw):
        super().__init__(quantile=quantile, **kw)
        self.quantile = quantile

    def _predict(self, X: _np.ndarray) -> Belief:
        try:
            kde = gaussian_kde(X.T)
            dens = kde(X.T)
        except Exception:
            dens = _np.ones(X.shape[0])
        score = -_np.log(dens + 1e-12)
        thr = float(_np.quantile(dens, self.quantile))
        flags = (dens < thr).astype(int)
        frac = float(flags.mean())
        return Belief("anomaly",
                      {"n_anomalies": int(flags.sum()), "fraction": frac,
                       "scores": score.tolist(), "flags": flags.tolist()},
                      float(min(1.0, frac * 3)), self.name)
