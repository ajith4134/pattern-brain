"""RL layer (Block 18, layer 8): learn actions, not patterns.

Generic mapping that keeps these online policies inside the same (T, D) -> Belief
interface: for the bandits, each of the D columns is treated as an arm whose
reward history is its column; Q-learning learns from the sequence's own
state->move dynamics. The bandit nodes here are the same UCB family the
evaluator design adopts (Block 22).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..belief import Belief
from ..node import Node
from ..registry import register


@register
class EpsilonGreedyBanditNode(Node):
    """Epsilon-greedy multi-armed bandit over the D columns (arms)."""
    layer = "rl"
    node_type = "epsilon_greedy_bandit"

    def __init__(self, epsilon: float = 0.1, random_state: int = 0, **kw):
        super().__init__(epsilon=epsilon, random_state=random_state, **kw)
        self.epsilon = epsilon
        self.random_state = random_state

    def _fit(self, X, y=None):
        self._means = X.mean(axis=0)
        self._counts = np.full(X.shape[1], X.shape[0])

    def _predict(self, X: np.ndarray) -> Belief:
        rng = np.random.default_rng(self.random_state)
        m = self._means
        if rng.random() < self.epsilon:
            arm = int(rng.integers(len(m)))
        else:
            arm = int(np.argmax(m))
        e = np.exp(m - m.max())
        p = e / e.sum()
        return Belief("action",
                      {"arm": arm, "values": m.tolist(), "policy": "epsilon_greedy"},
                      float(p.max()), self.name)


@register
class UCBBanditNode(Node):
    """UCB1 multi-armed bandit over the D columns (arms). Optimism-under-
    uncertainty exploration — the mechanism the evaluator adopts (Block 22)."""
    layer = "rl"
    node_type = "ucb_bandit"

    def _fit(self, X, y=None):
        self._means = X.mean(axis=0)
        self._counts = np.full(X.shape[1], X.shape[0], dtype=float)
        self._t = float(X.shape[0] * X.shape[1])

    def _predict(self, X: np.ndarray) -> Belief:
        ucb = self._means + np.sqrt(2 * np.log(max(2.0, self._t)) / self._counts)
        arm = int(np.argmax(ucb))
        scale = np.abs(self._means).max() + 1e-9
        conf = float(min(1.0, max(0.0, self._means.max() / scale)))
        return Belief("action",
                      {"arm": arm, "ucb": ucb.tolist(), "values": self._means.tolist(),
                       "policy": "ucb1"}, conf, self.name)


@register
class QLearningNode(Node):
    """Tabular Q-learning. State = KMeans cluster of each row; action = sign of
    the next change in the cross-feature mean (sell/hold/buy); reward = that
    realized change. Learns Q along the sequence, predicts the best action for
    the final state. Self-supervised from the sequence itself."""
    layer = "rl"
    node_type = "q_learning"

    def __init__(self, n_states: int = 3, alpha: float = 0.5, gamma: float = 0.9,
                 random_state: int = 0, **kw):
        super().__init__(n_states=n_states, alpha=alpha, gamma=gamma,
                         random_state=random_state, **kw)
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state

    def _fit(self, X, y=None):
        T = X.shape[0]
        K = max(2, min(self.n_states, T))
        self._km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state).fit(X)
        s = self._km.labels_
        dm = np.diff(X.mean(axis=1))
        acts = (np.sign(dm).astype(int) + 1)  # 0=down,1=flat,2=up
        Q = np.zeros((K, 3))
        for t in range(len(dm) - 1):
            st, act, r, st2 = s[t], acts[t], dm[t], s[t + 1]
            Q[st, act] += self.alpha * (r + self.gamma * Q[st2].max() - Q[st, act])
        self._Q = Q
        self._K = K

    def _predict(self, X: np.ndarray) -> Belief:
        st = int(self._km.predict(X)[-1])
        q = self._Q[st]
        a = int(np.argmax(q))
        action = {0: "sell", 1: "hold", 2: "buy"}[a]
        e = np.exp(q - q.max())
        p = e / e.sum()
        return Belief("action",
                      {"state": st, "action": action, "q_values": q.tolist()},
                      float(p.max()), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a) — more bandit policies (Block 42 RL; bandit family).
# --------------------------------------------------------------------------
@register
class ThompsonSamplingBanditNode(Node):
    """Thompson-sampling bandit over the D columns (arms): sample each arm's mean
    from a Gaussian posterior and pick the argmax (Bayesian exploration)."""
    layer = "rl"
    node_type = "thompson_bandit"

    def __init__(self, random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.random_state = random_state

    def _fit(self, X, y=None):
        self._means = X.mean(axis=0)
        self._sd = X.std(axis=0) / np.sqrt(max(1, X.shape[0]))

    def _predict(self, X: np.ndarray) -> Belief:
        rng = np.random.default_rng(self.random_state)
        samples = rng.normal(self._means, self._sd + 1e-9)
        arm = int(np.argmax(samples))
        e = np.exp(self._means - self._means.max())
        p = e / e.sum()
        return Belief("action",
                      {"arm": arm, "values": self._means.tolist(),
                       "samples": samples.tolist(), "policy": "thompson"},
                      float(p.max()), self.name)


@register
class GradientBanditNode(Node):
    """Gradient (softmax-preference) bandit over the D columns (arms)."""
    layer = "rl"
    node_type = "gradient_bandit"

    def _fit(self, X, y=None):
        self._means = X.mean(axis=0)

    def _predict(self, X: np.ndarray) -> Belief:
        h = self._means - self._means.mean()
        e = np.exp(h - h.max())
        p = e / e.sum()
        arm = int(np.argmax(p))
        return Belief("action",
                      {"arm": arm, "policy": "gradient", "values": self._means.tolist(),
                       "preferences": p.tolist()},
                      float(p.max()), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 2) — baseline policies (Block 42 RL).
# --------------------------------------------------------------------------
@register
class RandomPolicyNode(Node):
    """Uniform-random arm choice — the honest baseline every policy must beat."""
    layer = "rl"
    node_type = "random_policy"

    def __init__(self, random_state: int = 0, **kw):
        super().__init__(random_state=random_state, **kw)
        self.random_state = random_state

    def _predict(self, X: np.ndarray) -> Belief:
        rng = np.random.default_rng(self.random_state)
        arm = int(rng.integers(X.shape[1]))
        return Belief("action", {"arm": arm, "policy": "random",
                                 "values": X.mean(axis=0).tolist()},
                      float(1.0 / X.shape[1]), self.name)


@register
class SoftmaxPolicyNode(Node):
    """Softmax (Boltzmann) policy over the D columns' mean rewards, temperature τ."""
    layer = "rl"
    node_type = "softmax_policy"

    def __init__(self, tau: float = 1.0, **kw):
        super().__init__(tau=tau, **kw)
        self.tau = tau

    def _predict(self, X: np.ndarray) -> Belief:
        m = X.mean(axis=0)
        z = m / max(1e-9, self.tau)
        e = np.exp(z - z.max())
        p = e / e.sum()
        arm = int(np.argmax(p))
        return Belief("action", {"arm": arm, "policy": "softmax",
                                 "probabilities": p.tolist(), "values": m.tolist()},
                      float(p.max()), self.name)


# --------------------------------------------------------------------------
# Build step 6 (Phase 6a, batch 3) — non-stationary / adversarial bandits.
# --------------------------------------------------------------------------
@register
class DiscountedUCBNode(Node):
    """Discounted-UCB bandit — UCB1 with geometric forgetting, for non-stationary
    rewards (the Sliding-Window/Discounted-UCB family the evaluator adopts)."""
    layer = "rl"
    node_type = "discounted_ucb"

    def __init__(self, gamma: float = 0.95, **kw):
        super().__init__(gamma=gamma, **kw)
        self.gamma = gamma

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        w = self.gamma ** np.arange(T)[::-1]        # recent rows weigh more
        wsum = w.sum() + 1e-9
        means = (w[:, None] * X).sum(axis=0) / wsum
        ucb = means + np.sqrt(2 * np.log(max(2.0, wsum)) / wsum) * np.ones(D)
        arm = int(np.argmax(ucb))
        scale = np.abs(means).max() + 1e-9
        return Belief("action", {"arm": arm, "ucb": ucb.tolist(),
                                 "values": means.tolist(), "policy": "discounted_ucb"},
                      float(min(1.0, max(0.0, means.max() / scale))), self.name)


@register
class EXP3BanditNode(Node):
    """EXP3 adversarial bandit — exponential-weights over the D arms."""
    layer = "rl"
    node_type = "exp3_bandit"

    def __init__(self, eta: float = 0.1, **kw):
        super().__init__(eta=eta, **kw)
        self.eta = eta

    def _predict(self, X: np.ndarray) -> Belief:
        rewards = X.mean(axis=0)
        r = (rewards - rewards.min()) / (np.ptp(rewards) + 1e-9)   # normalize to [0,1]
        w = np.exp(self.eta * np.cumsum(np.ones_like(r)) * 0 + self.eta * r)
        p = w / w.sum()
        arm = int(np.argmax(p))
        return Belief("action", {"arm": arm, "policy": "exp3",
                                 "probabilities": p.tolist(), "values": rewards.tolist()},
                      float(p.max()), self.name)
