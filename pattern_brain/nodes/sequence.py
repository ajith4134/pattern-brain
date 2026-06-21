"""Sequence layer (Block 18, layer 4): predict the next state / value.

Includes the HMM-family representative named in the starter set (PLAN.md build
step 1): a compact Gaussian HMM trained by Baum-Welch, plus a discrete Markov
chain and two classical forecasters.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class MarkovChainNode(Node):
    """Discretize rows into states (KMeans), learn the transition matrix, predict
    the next-state distribution. Regime/next-state detection (Block 2 family 2)."""
    layer = "sequence"
    node_type = "markov_chain"

    def __init__(self, n_states: int = 3, random_state: int = 0, **kw):
        super().__init__(n_states=n_states, random_state=random_state, **kw)
        self.n_states = n_states
        self.random_state = random_state

    def _fit(self, X, y=None):
        k = max(2, min(self.n_states, X.shape[0]))
        self._km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
        states = self._km.labels_
        T = np.ones((k, k))  # Laplace smoothing
        for a, b in zip(states[:-1], states[1:]):
            T[a, b] += 1
        self._T = T / T.sum(axis=1, keepdims=True)
        self._k = k

    def _predict(self, X: np.ndarray) -> Belief:
        cur = int(self._km.predict(X)[-1])
        dist = self._T[cur]
        return Belief("next_state",
                      {"current_state": cur, "next_state": int(np.argmax(dist)),
                       "distribution": dist.tolist()},
                      float(dist.max()), self.name)


@register
class ARNode(Node):
    """Per-feature autoregressive AR(p) forecast via least squares."""
    layer = "sequence"
    node_type = "autoregressive"

    def __init__(self, p: int = 3, **kw):
        super().__init__(p=p, **kw)
        self.p = p

    def _fit(self, X, y=None):
        p = max(1, min(self.p, X.shape[0] - 1))
        self._p = p
        self._coef = []
        r2s = []
        for j in range(X.shape[1]):
            x = X[:, j]
            rows = np.array([x[i - p:i] for i in range(p, len(x))])
            target = x[p:]
            if len(rows) == 0:
                self._coef.append(np.zeros(p))
                continue
            c, *_ = np.linalg.lstsq(rows, target, rcond=None)
            self._coef.append(c)
            pred = rows @ c
            ss_res = float(np.sum((target - pred) ** 2))
            ss_tot = float(np.sum((target - target.mean()) ** 2)) + 1e-12
            r2s.append(max(0.0, 1 - ss_res / ss_tot))
        self._conf = float(np.mean(r2s)) if r2s else 0.0

    def _predict(self, X: np.ndarray) -> Belief:
        p = self._p
        preds = [float(np.dot(self._coef[j], X[-p:, j])) for j in range(X.shape[1])]
        return Belief("forecast", {"next_vector": preds, "order": p},
                      self._conf, self.name)


@register
class ExponentialSmoothingNode(Node):
    """Holt's linear-trend smoothing; 1-step forecast per feature."""
    layer = "sequence"
    node_type = "exp_smoothing"

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, **kw):
        super().__init__(alpha=alpha, beta=beta, **kw)
        self.alpha = alpha
        self.beta = beta

    def _predict(self, X: np.ndarray) -> Belief:
        preds, errs = [], []
        for j in range(X.shape[1]):
            x = X[:, j]
            level = x[0]
            trend = x[1] - x[0] if len(x) > 1 else 0.0
            ae = []
            for t in range(1, len(x)):
                ae.append(abs(x[t] - (level + trend)))
                prev = level
                level = self.alpha * x[t] + (1 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev) + (1 - self.beta) * trend
            preds.append(float(level + trend))
            scale = np.std(x) + 1e-9
            errs.append(np.mean(ae) / scale if ae else 1.0)
        conf = float(1.0 / (1.0 + np.mean(errs)))
        return Belief("forecast", {"next_vector": preds}, conf, self.name)


@register
class GaussianHMMNode(Node):
    """Compact Gaussian Hidden Markov Model (diagonal covariances) trained by
    Baum-Welch (scaled forward-backward EM). KMeans initialization. Emits the
    current most-likely state and the implied next-state distribution.

    The HMM-family representative of build step 1; hmmlearn is a later optional
    upgrade. The forward pass is a prime candidate for the C/C++ hot-path rewrite
    described in PLAN.md §0b once profiled."""
    layer = "sequence"
    node_type = "gaussian_hmm"

    def __init__(self, n_states: int = 2, n_iter: int = 15, random_state: int = 0, **kw):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state, **kw)
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state

    @staticmethod
    def _emission(X, means, covs):
        # diagonal-Gaussian likelihood per state -> (T, K)
        T = X.shape[0]
        K = means.shape[0]
        B = np.empty((T, K))
        for k in range(K):
            diff = X - means[k]
            log_norm = -0.5 * np.sum(diff * diff / covs[k], axis=1)
            log_det = -0.5 * np.sum(np.log(2 * np.pi * covs[k]))
            B[:, k] = np.exp(log_norm + log_det)
        return B + 1e-300

    def _fit(self, X, y=None):
        T, D = X.shape
        K = max(2, min(self.n_states, T))
        km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state).fit(X)
        means = km.cluster_centers_.copy()
        covs = np.stack([
            (np.var(X[km.labels_ == k], axis=0) + 1e-2) if np.sum(km.labels_ == k) > 1
            else (np.var(X, axis=0) + 1e-2) for k in range(K)
        ])
        pi = np.full(K, 1.0 / K)
        A = np.full((K, K), 1.0 / K)

        for _ in range(self.n_iter):
            B = self._emission(X, means, covs)
            # scaled forward
            alpha = np.zeros((T, K))
            c = np.zeros(T)
            alpha[0] = pi * B[0]
            c[0] = alpha[0].sum() + 1e-300
            alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                c[t] = alpha[t].sum() + 1e-300
                alpha[t] /= c[t]
            # scaled backward
            beta = np.zeros((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            xi = np.zeros((K, K))
            for t in range(T - 1):
                xi += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
            # M-step
            pi = gamma[0] + 1e-3
            pi /= pi.sum()
            A = xi + 1e-3
            A /= A.sum(axis=1, keepdims=True)
            for k in range(K):
                w = gamma[:, k]
                sw = w.sum() + 1e-300
                means[k] = (w[:, None] * X).sum(axis=0) / sw
                covs[k] = (w[:, None] * (X - means[k]) ** 2).sum(axis=0) / sw + 1e-2

        self._pi, self._A, self._means, self._covs, self._K = pi, A, means, covs, K
        self._loglik = float(np.log(c).sum())

    def _predict(self, X: np.ndarray) -> Belief:
        B = self._emission(X, self._means, self._covs)
        T, K = B.shape
        alpha = np.zeros((T, K))
        c0 = (self._pi * B[0]).sum() + 1e-300
        alpha[0] = self._pi * B[0] / c0
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self._A) * B[t]
            alpha[t] /= alpha[t].sum() + 1e-300
        cur_post = alpha[-1]
        cur = int(np.argmax(cur_post))
        nxt_dist = self._A[cur]
        return Belief("regime",
                      {"current_state": cur, "state_posterior": cur_post.tolist(),
                       "next_state": int(np.argmax(nxt_dist)),
                       "next_distribution": nxt_dist.tolist(),
                       "log_likelihood": self._loglik},
                      float(cur_post.max()), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — more classical forecasters (Block 42 time-series).
# Each predicts the next vector from the window; emits the loose 'forecast' type.
# --------------------------------------------------------------------------
def _forecast_belief(next_vec, resid_scale, name, extra=None):
    next_vec = np.asarray(next_vec, float).ravel()
    payload = {"next_vector": next_vec.tolist(), "estimate": float(next_vec[0])}
    if extra:
        payload.update(extra)
    conf = float(1.0 / (1.0 + resid_scale))
    return Belief("forecast", payload, max(0.0, min(1.0, conf)), name)


@register
class NaiveMeanForecastNode(Node):
    """Forecast the next value as the window mean (a strong, honest baseline)."""
    layer = "sequence"
    node_type = "naive_mean_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        nxt = X.mean(axis=0)
        resid = float(np.mean(np.abs(X - nxt)))
        return _forecast_belief(nxt, resid, self.name)


@register
class DriftForecastNode(Node):
    """Random-walk-with-drift forecast: last value + average per-step change."""
    layer = "sequence"
    node_type = "drift_forecast"

    def _predict(self, X: np.ndarray) -> Belief:
        if X.shape[0] < 2:
            nxt = X[-1]
            resid = 0.0
        else:
            drift = (X[-1] - X[0]) / (X.shape[0] - 1)
            nxt = X[-1] + drift
            resid = float(np.mean(np.abs(np.diff(X, axis=0) - drift)))
        return _forecast_belief(nxt, resid, self.name)


@register
class HoltLinearForecastNode(Node):
    """Holt's double exponential smoothing (level + trend) one-step forecast."""
    layer = "sequence"
    node_type = "holt_linear_forecast"

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, **kw):
        super().__init__(alpha=alpha, beta=beta, **kw)
        self.alpha = alpha
        self.beta = beta

    def _predict(self, X: np.ndarray) -> Belief:
        a, b = self.alpha, self.beta
        level = X[0].astype(float).copy()
        trend = (X[1] - X[0]).astype(float) if X.shape[0] > 1 else np.zeros(X.shape[1])
        errs = []
        for t in range(1, X.shape[0]):
            pred = level + trend
            errs.append(np.abs(X[t] - pred))
            new_level = a * X[t] + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            level = new_level
        nxt = level + trend
        resid = float(np.mean(errs)) if errs else 0.0
        return _forecast_belief(nxt, resid, self.name)


@register
class ThetaForecastNode(Node):
    """Theta-method-style forecast: average of a linear trend extrapolation and an
    exponentially-smoothed level (a robust competition-grade baseline)."""
    layer = "sequence"
    node_type = "theta_forecast"

    def __init__(self, alpha: float = 0.3, **kw):
        super().__init__(alpha=alpha, **kw)
        self.alpha = alpha

    def _predict(self, X: np.ndarray) -> Belief:
        T = X.shape[0]
        t = np.arange(T)
        nxt_cols = []
        for j in range(X.shape[1]):
            y = X[:, j]
            if T >= 2:
                m, c = np.polyfit(t, y, 1)
                trend_next = m * T + c
            else:
                trend_next = y[-1]
            lvl = y[0]
            for v in y[1:]:
                lvl = self.alpha * v + (1 - self.alpha) * lvl
            nxt_cols.append(0.5 * trend_next + 0.5 * lvl)
        nxt = np.array(nxt_cols)
        resid = float(np.std(X)) / (abs(float(np.mean(X))) + 1.0)
        return _forecast_belief(nxt, resid, self.name)
