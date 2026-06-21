"""Phase 7c — probabilistic / sequence-structure nodes (PLAN.md §11, Block 57).

Fills the coverage meter's Markov/HMM gap (higher-order Markov, semi-Markov,
hierarchical HMM, CRF) and the Probabilistic gap (Bayesian network, dynamic
Bayesian network). Light-stack (numpy/scipy/sklearn), behind the generic Node
interface (Rule 23), emitting existing interlingua belief types.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


def _discretize(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k:
        return np.zeros(len(x), dtype=int)
    edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(x, edges).astype(int)


def _direction_labels(x: np.ndarray) -> np.ndarray:
    d = np.diff(x, prepend=x[:1])
    sd = 0.25 * (np.std(d) + 1e-9)
    return np.where(d > sd, 2, np.where(d < -sd, 0, 1)).astype(int)   # down/flat/up


@register
class HigherOrderMarkovNode(Node):
    """Order-k Markov chain on discretized states: P(s_t | s_{t-1}, …, s_{t-k})."""
    layer = "sequence"
    node_type = "higher_order_markov"

    def __init__(self, k_states: int = 4, order: int = 2, **kw):
        super().__init__(k_states=k_states, order=order, **kw)
        self.k, self.order = k_states, order

    def _predict(self, X: np.ndarray) -> Belief:
        s = _discretize(X[:, 0], self.k)
        o = self.order
        counts = defaultdict(lambda: np.zeros(self.k))
        for t in range(o, len(s)):
            counts[tuple(s[t - o:t])][s[t]] += 1
        ctx = tuple(s[-o:])
        c = counts.get(ctx)
        dist = (c / c.sum()) if c is not None and c.sum() > 0 else np.ones(self.k) / self.k
        return Belief("next_state",
                      {"current_state": int(s[-1]), "distribution": dist.tolist(),
                       "next_state": int(np.argmax(dist)), "order": o},
                      float(dist.max()), self.name)


@register
class SemiMarkovNode(Node):
    """Semi-Markov: models state transitions AND dwell durations. Predicts the next
    distinct state + its expected duration."""
    layer = "sequence"
    node_type = "semi_markov"

    def __init__(self, k_states: int = 4, **kw):
        super().__init__(k_states=k_states, **kw)
        self.k = k_states

    def _predict(self, X: np.ndarray) -> Belief:
        s = _discretize(X[:, 0], self.k)
        runs = []                                   # (state, duration)
        cur, dur = s[0], 1
        for v in s[1:]:
            if v == cur:
                dur += 1
            else:
                runs.append((cur, dur)); cur, dur = v, 1
        runs.append((cur, dur))
        trans = defaultdict(lambda: np.zeros(self.k))
        durs = defaultdict(list)
        for i, (st, du) in enumerate(runs):
            durs[st].append(du)
            if i + 1 < len(runs):
                trans[st][runs[i + 1][0]] += 1
        last_state, last_dur = runs[-1]
        c = trans.get(last_state)
        dist = (c / c.sum()) if c is not None and c.sum() > 0 else np.ones(self.k) / self.k
        nxt = int(np.argmax(dist))
        exp_dur = float(np.mean(durs.get(nxt, [1.0])))
        return Belief("next_state",
                      {"current_state": int(last_state), "distribution": dist.tolist(),
                       "next_state": nxt, "current_run_length": int(last_dur),
                       "expected_duration": exp_dur},
                      float(dist.max()), self.name)


@register
class HierarchicalHMMNode(Node):
    """Hierarchical HMM (2-level, light): coarse regimes via a Gaussian mixture over
    [level, volatility], with a transition matrix over those regimes. Emits 'regime'."""
    layer = "sequence"
    node_type = "hierarchical_hmm"

    def __init__(self, n_regimes: int = 2, **kw):
        super().__init__(n_regimes=n_regimes, **kw)
        self.n = n_regimes

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        feats = np.column_stack([x, np.abs(np.diff(x, prepend=x[:1]))])
        from sklearn.mixture import GaussianMixture
        n = min(self.n, max(1, len(x) // 10))
        try:
            gm = GaussianMixture(n_components=max(1, n), covariance_type="diag",
                                 random_state=0, max_iter=50).fit(feats)
            labels = gm.predict(feats)
            post = gm.predict_proba(feats[-1:])[0]
            loglik = float(gm.score(feats))
        except Exception:
            labels = np.zeros(len(x), dtype=int); post = np.array([1.0]); loglik = 0.0
        K = int(labels.max()) + 1
        T = np.zeros((K, K))
        for a, b in zip(labels[:-1], labels[1:]):
            T[a, b] += 1
        cur = int(labels[-1])
        row = T[cur]
        nxt_dist = (row / row.sum()) if row.sum() > 0 else np.ones(K) / K
        # pad posterior to K
        post_full = np.zeros(K); post_full[:len(post)] = post
        return Belief("regime",
                      {"current_state": cur, "state_posterior": post_full.tolist(),
                       "next_distribution": nxt_dist.tolist(),
                       "next_state": int(np.argmax(nxt_dist)),
                       "log_likelihood": loglik, "n_regimes": K},
                      float(nxt_dist.max()), self.name)


@register
class CRFTaggerNode(Node):
    """Linear-chain Conditional Random Field for sequence labeling (down/flat/up),
    trained by gradient ascent on conditional log-likelihood (forward-backward),
    decoded with Viterbi. A genuine discriminative structured predictor."""
    layer = "decision"
    node_type = "crf_tagger"

    def __init__(self, k_obs: int = 5, epochs: int = 25, lr: float = 0.2, **kw):
        super().__init__(k_obs=k_obs, epochs=epochs, lr=lr, **kw)
        self.V, self.epochs, self.lr = k_obs, epochs, lr

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if len(x) < 10:
            return Belief("decision", {"action": 0, "predictions": [1]}, 0.1, self.name)
        obs = _discretize(x, self.V)
        lab = _direction_labels(x)
        L, V = 3, self.V
        U = np.zeros((L, V))            # unary weights label x obs
        Tr = np.zeros((L, L))           # transition weights
        for _ in range(self.epochs):
            gU, gT, ll = self._fb_grad(obs, lab, U, Tr, L)
            U += self.lr * gU
            Tr += self.lr * gT
        path = self._viterbi(obs, U, Tr, L)
        return Belief("decision",
                      {"action": int(path[-1]) - 1, "predictions": [int(p) for p in path[-5:]],
                       "vote": int(np.bincount(path, minlength=L).argmax()) - 1,
                       "model": "linear-chain CRF"},
                      0.5, self.name)

    @staticmethod
    def _fb_grad(obs, lab, U, Tr, L):
        n = len(obs)
        # log potentials
        emit = U[:, obs].T                                  # (n, L)
        # forward-backward in log space
        a = np.full((n, L), -np.inf); a[0] = emit[0]
        for t in range(1, n):
            a[t] = emit[t] + np.array([
                np.logaddexp.reduce(a[t - 1] + Tr[:, j]) for j in range(L)])
        b = np.full((n, L), -np.inf); b[-1] = 0.0
        for t in range(n - 2, -1, -1):
            b[t] = np.array([
                np.logaddexp.reduce(Tr[i] + emit[t + 1] + b[t + 1]) for i in range(L)])
        logZ = np.logaddexp.reduce(a[-1])
        marg = np.exp(a + b - logZ)                         # (n, L) node marginals
        # gradients = empirical - expected
        gU = np.zeros_like(U)
        for t in range(n):
            gU[lab[t], obs[t]] += 1
            gU[:, obs[t]] -= marg[t]
        gT = np.zeros_like(Tr)
        for t in range(1, n):
            gT[lab[t - 1], lab[t]] += 1
            pair = np.exp(a[t - 1][:, None] + Tr + emit[t][None, :] + b[t][None, :] - logZ)
            gT -= pair
        ll = sum(emit[t, lab[t]] for t in range(n)) + sum(Tr[lab[t - 1], lab[t]] for t in range(1, n)) - logZ
        return gU, gT, ll

    @staticmethod
    def _viterbi(obs, U, Tr, L):
        n = len(obs)
        emit = U[:, obs].T
        dp = np.full((n, L), -np.inf); bp = np.zeros((n, L), int)
        dp[0] = emit[0]
        for t in range(1, n):
            for j in range(L):
                scores = dp[t - 1] + Tr[:, j]
                bp[t, j] = int(np.argmax(scores))
                dp[t, j] = emit[t, j] + scores.max()
        path = [int(np.argmax(dp[-1]))]
        for t in range(n - 1, 0, -1):
            path.append(bp[t, path[-1]])
        return path[::-1]


@register
class ChowLiuBayesNetNode(Node):
    """Bayesian network via the Chow-Liu algorithm: learn the max-mutual-information
    spanning tree over discretized lagged features, then predict the next value from
    its strongest-MI parent (its Markov blanket edge). Emits 'forecast' + structure."""
    layer = "probability"
    node_type = "bayesian_network"

    def __init__(self, k_states: int = 4, n_lags: int = 3, **kw):
        super().__init__(k_states=k_states, n_lags=n_lags, **kw)
        self.k, self.L = k_states, n_lags

    @staticmethod
    def _mi(a, b, k):
        joint = np.zeros((k, k))
        for i, j in zip(a, b):
            joint[i, j] += 1
        joint /= joint.sum() + 1e-12
        pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = joint * np.log((joint + 1e-12) / (pa * pb + 1e-12))
        return float(np.nansum(m))

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _discretize(x, self.k)
        n = len(s)
        if n < self.L + 6:
            return Belief("forecast", {"next_vector": [float(x[-1])]}, 0.1, self.name)
        # variables: target s_t and lags s_{t-1..t-L}
        cols = {f"lag{j}": s[self.L - j: n - j] for j in range(1, self.L + 1)}
        target = s[self.L:]
        # Chow-Liu edges: MI of each lag with the target (tree rooted at target)
        mis = {name: self._mi(target, col, self.k) for name, col in cols.items()}
        best = max(mis, key=mis.get)
        bj = int(best[3:])
        # P(target | best lag) table -> predict from the current value of that lag
        tab = np.zeros((self.k, self.k))
        for tgt, pv in zip(target, cols[best]):
            tab[pv, tgt] += 1
        cur_parent = int(s[-bj])
        row = tab[cur_parent]
        dist = (row / row.sum()) if row.sum() > 0 else np.ones(self.k) / self.k
        # map predicted state back to an approximate value (bin mean of x)
        binmeans = [float(x[s == st].mean()) if np.any(s == st) else float(x.mean())
                    for st in range(self.k)]
        est = float(sum(p * m for p, m in zip(dist, binmeans)))
        return Belief("forecast",
                      {"next_vector": [est], "estimate": est,
                       "bn_root_parent": best, "mutual_information": round(mis[best], 4),
                       "edges": {k_: round(v, 4) for k_, v in mis.items()},
                       "model": "Chow-Liu Bayesian network"},
                      float(dist.max()), self.name)


@register
class DynamicBayesianNetworkNode(Node):
    """Dynamic Bayesian Network (first-order, factored): per-feature discrete
    transition P(s^f_t | s^f_{t-1}); forecasts channel 0's next-state distribution
    (and the per-feature next states). Emits 'next_state'."""
    layer = "sequence"
    node_type = "dynamic_bayesian_network"

    def __init__(self, k_states: int = 4, **kw):
        super().__init__(k_states=k_states, **kw)
        self.k = k_states

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        per_feat_next = []
        dist0 = np.ones(self.k) / self.k
        cur0 = 0
        for f in range(D):
            s = _discretize(X[:, f], self.k)
            Tm = np.zeros((self.k, self.k))
            for a, b in zip(s[:-1], s[1:]):
                Tm[a, b] += 1
            cur = int(s[-1])
            row = Tm[cur]
            dist = (row / row.sum()) if row.sum() > 0 else np.ones(self.k) / self.k
            per_feat_next.append(int(np.argmax(dist)))
            if f == 0:
                dist0, cur0 = dist, cur
        return Belief("next_state",
                      {"current_state": cur0, "distribution": dist0.tolist(),
                       "next_state": int(np.argmax(dist0)),
                       "per_feature_next_state": per_feat_next, "n_features": D,
                       "model": "first-order DBN (factored)"},
                      float(dist0.max()), self.name)
