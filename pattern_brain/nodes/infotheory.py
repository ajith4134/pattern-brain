"""Phase 7d — information-theory nodes (PLAN.md §11, Block 57).

Completes the info-theory family (the owner's "signal vs noise / predictability /
causality" tools): Mutual Information, Transfer Entropy (directed info flow),
conditional + cross entropy, Kolmogorov-complexity proxy (compression), and MDL
(minimum description length with Markov-order selection). Light-stack (numpy +
stdlib zlib), behind the generic Node interface (Rule 23), emitting `signal`.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


def _disc(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k:
        return np.zeros(len(x), dtype=int)
    edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(x, edges).astype(int)


def _entropy(counts: np.ndarray) -> float:
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mi(a: np.ndarray, b: np.ndarray, k: int) -> float:
    j = np.zeros((k, k))
    for i, jj in zip(a, b):
        j[i, jj] += 1
    j /= j.sum() + 1e-12
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = j * np.log2((j + 1e-12) / (pa * pb + 1e-12))
    return float(max(0.0, np.nansum(m)))


@register
class MutualInformationNode(Node):
    """Auto-mutual-information between x_t and x_{t-1} — nonlinear self-predictability
    (captures dependence that linear autocorrelation misses)."""
    layer = "signal"
    node_type = "mutual_information"

    def __init__(self, k: int = 5, lag: int = 1, **kw):
        super().__init__(k=k, lag=lag, **kw)
        self.k, self.lag = k, lag

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _disc(x, self.k)
        mi = _mi(s[self.lag:], s[:-self.lag], self.k) if len(s) > self.lag + 2 else 0.0
        hmax = np.log2(self.k)
        return Belief("signal", {"mutual_information": mi, "lag": self.lag,
                                 "normalized": mi / (hmax + 1e-12), "series": x.tolist()},
                      float(max(0.0, min(1.0, mi / (hmax + 1e-12)))), self.name)


@register
class TransferEntropyNode(Node):
    """Transfer entropy TE(Y→X) = I(X_t ; Y_{t-1} | X_{t-1}) — directed information
    flow ('does channel Y help predict X beyond X's own past?'); a causality proxy.
    Needs D≥2; reports both directions + the net flow. 0 on a single channel."""
    layer = "signal"
    node_type = "transfer_entropy"

    def __init__(self, k: int = 4, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _te(self, x, y):
        k = self.k
        Xt, Xp, Yp = _disc(x[1:], k), _disc(x[:-1], k), _disc(y[:-1], k)
        # TE = H(Xt|Xp) - H(Xt|Xp,Yp)
        # H(Xt|Xp) = H(Xt,Xp) - H(Xp)
        def H(*arrs):
            dims = [k] * len(arrs)
            c = np.zeros(dims)
            for tup in zip(*arrs):
                c[tup] += 1
            return _entropy(c.ravel())
        h_xt_xp = H(Xt, Xp) - H(Xp)
        h_xt_xp_yp = H(Xt, Xp, Yp) - H(Xp, Yp)
        return float(max(0.0, h_xt_xp - h_xt_xp_yp))

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        x = X[:, 0].astype(float)
        if D < 2 or T < 12:
            return Belief("signal", {"transfer_entropy": 0.0, "direction": "n/a (D<2)",
                                     "series": x.tolist()}, 0.0, self.name)
        y = X[:, 1].astype(float)
        te_yx = self._te(x, y)            # Y -> X
        te_xy = self._te(y, x)            # X -> Y
        net = te_yx - te_xy
        return Belief("signal", {"transfer_entropy": te_yx, "te_y_to_x": te_yx,
                                 "te_x_to_y": te_xy, "net_flow": net,
                                 "direction": "Y->X" if net > 0 else "X->Y",
                                 "series": x.tolist()},
                      float(max(0.0, min(1.0, abs(net)))), self.name)


@register
class ConditionalEntropyNode(Node):
    """Conditional entropy H(x_t | x_{t-1}) — residual uncertainty after the last
    value (low = predictable, high = unpredictable)."""
    layer = "signal"
    node_type = "conditional_entropy"

    def __init__(self, k: int = 5, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _disc(x, self.k)
        k = self.k
        if len(s) < 4:
            return Belief("signal", {"conditional_entropy": 0.0, "series": x.tolist()}, 0.0, self.name)
        jc = np.zeros((k, k))
        for a, b in zip(s[:-1], s[1:]):
            jc[a, b] += 1
        h_joint = _entropy(jc.ravel())
        h_prev = _entropy(jc.sum(1))
        hc = max(0.0, h_joint - h_prev)
        return Belief("signal", {"conditional_entropy": hc,
                                 "normalized": hc / (np.log2(k) + 1e-12), "series": x.tolist()},
                      float(max(0.0, 1.0 - hc / (np.log2(k) + 1e-12))), self.name)


@register
class CrossEntropyNode(Node):
    """Cross-entropy between the first-half and second-half value distributions —
    distribution drift / surprise of recent data under the older model."""
    layer = "signal"
    node_type = "cross_entropy"

    def __init__(self, k: int = 6, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _disc(x, self.k)
        mid = len(s) // 2
        if mid < 3:
            return Belief("signal", {"cross_entropy": 0.0, "series": x.tolist()}, 0.0, self.name)
        p = np.bincount(s[:mid], minlength=self.k) + 1.0
        q = np.bincount(s[mid:], minlength=self.k) + 1.0
        p /= p.sum(); q /= q.sum()
        ce = float(-(q * np.log2(p)).sum())
        kl = float((q * np.log2(q / p)).sum())
        return Belief("signal", {"cross_entropy": ce, "kl_divergence": kl,
                                 "drift": kl, "series": x.tolist()},
                      float(max(0.0, min(1.0, kl))), self.name)


@register
class KolmogorovComplexityNode(Node):
    """Kolmogorov-complexity proxy via compression: ratio of compressed to raw size
    of the symbolized series (low = structured/compressible, ~1 = random)."""
    layer = "signal"
    node_type = "kolmogorov_complexity"

    def __init__(self, k: int = 8, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _disc(x, self.k).astype(np.uint8).tobytes()
        if len(s) < 8:
            return Belief("signal", {"kolmogorov_complexity": 1.0, "series": x.tolist()}, 0.0, self.name)
        ratio = len(zlib.compress(s, 9)) / len(s)
        ratio = float(min(1.5, ratio))
        return Belief("signal", {"kolmogorov_complexity": ratio,
                                 "compressibility": float(max(0.0, 1.0 - ratio)),
                                 "series": x.tolist()},
                      float(max(0.0, min(1.0, 1.0 - ratio))), self.name)


@register
class MDLComplexityNode(Node):
    """Minimum Description Length: pick the Markov order (0/1/2) that minimizes
    total bits = data bits under the model + parameter bits. Reports the chosen
    order + bits/symbol — a principled complexity + predictability measure."""
    layer = "signal"
    node_type = "mdl_complexity"

    def __init__(self, k: int = 4, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        s = _disc(x, self.k)
        n, k = len(s), self.k
        if n < 12:
            return Belief("signal", {"mdl_order": 0, "description_length": 0.0,
                                     "series": x.tolist()}, 0.0, self.name)
        best = (None, np.inf, 0.0)
        for order in (0, 1, 2):
            from collections import defaultdict
            ctx = defaultdict(lambda: np.zeros(k))
            for t in range(order, n):
                ctx[tuple(s[t - order:t])][s[t]] += 1
            data_bits = sum(cnt.sum() * _entropy(cnt) for cnt in ctx.values())
            n_params = len(ctx) * (k - 1)
            param_bits = 0.5 * n_params * np.log2(max(2, n))
            total = data_bits + param_bits
            if total < best[1]:
                best = (order, total, data_bits / max(1, n - order))
        return Belief("signal", {"mdl_order": int(best[0]),
                                 "description_length": float(best[1]),
                                 "bits_per_symbol": float(best[2]), "series": x.tolist()},
                      float(max(0.0, min(1.0, 1.0 - best[2] / (np.log2(k) + 1e-12)))), self.name)
