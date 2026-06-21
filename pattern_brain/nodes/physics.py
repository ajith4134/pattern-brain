"""Phase 7e — physics / PhD-tier nodes (PLAN.md §11, Block 57; owner's headline ask).

Light-stack only (numpy), each behind the generic Node interface (Rule 23) and
emitting EXISTING interlingua belief types so the bank stays conformant (Block 46):

* **Reservoir computing** (chaos prediction — the 2026 workhorse): `esn_forecaster`
  (random Echo State Network) + `deterministic_esn_forecaster` (TCRC-style,
  logistic-map reservoir, no randomness — arxiv 2501.15615). Emit `forecast`.
* **Chaos / nonlinear-dynamics diagnostics** ("is this deterministic chaos or
  noise?"): `lyapunov_exponent` (Rosenstein largest-LE), `recurrence_rate` (RQA).
* **Fractal / long-memory**: `hurst_exponent` (rescaled-range).
* **Complexity / predictability**: `sample_entropy`, `permutation_entropy`
  (Bandt-Pompe). All emit `signal`.
* **Econophysics noise cleaning**: `rmt_denoise` — Random Matrix Theory
  (Marchenko-Pastur) eigenvalue cleaning of the cross-feature correlation matrix.
  Emits `denoised`.

These operationalize "predict chaos / convert randomness into patterns / separate
signal from noise" on the generic (T, D) sequence — zero stock coupling.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register


# --------------------------------------------------------------- reservoirs
def _run_reservoir(x: np.ndarray, Win: np.ndarray, W: np.ndarray,
                   a: float = 0.3) -> np.ndarray:
    """Drive a leaky-tanh reservoir with scalar series x; return state matrix
    (one row per step, predicting the NEXT value), plus the final state."""
    N = W.shape[0]
    s = np.zeros(N)
    states = []
    for t in range(len(x)):
        u = np.array([1.0, x[t]])
        s = (1.0 - a) * s + a * np.tanh(Win @ u + W @ s)
        states.append(np.concatenate([[1.0, x[t]], s]))
    return np.asarray(states), s


def _ridge_readout(S: np.ndarray, y: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    A = S.T @ S + reg * np.eye(S.shape[1])
    return np.linalg.solve(A, S.T @ y)


def _esn_predict(x: np.ndarray, Win: np.ndarray, W: np.ndarray, a: float = 0.3):
    """One-step-ahead prediction of x[next] via reservoir + ridge readout."""
    states, _ = _run_reservoir(x, Win, W, a)
    washout = min(10, len(x) // 4)
    S, Y = states[washout:-1], x[washout + 1:]
    if len(S) < 5:
        return float(x[-1]), 0.1
    Wout = _ridge_readout(S, Y)
    # advance one more step from the last observed value -> predict x[next]
    pred = float(states[-1] @ Wout)
    # confidence from in-sample fit quality
    resid = Y - S @ Wout
    nrmse = float(np.sqrt(np.mean(resid ** 2)) / (np.std(Y) + 1e-9))
    return pred, max(0.05, min(0.9, 1.0 - nrmse))


def _scale_spectral_radius(W: np.ndarray, rho: float = 0.9) -> np.ndarray:
    sr = float(np.max(np.abs(np.linalg.eigvals(W)))) or 1.0
    return W * (rho / sr)


@register
class EchoStateNetworkNode(Node):
    """Reservoir computing (random ESN) — the standard model-free chaotic-sequence
    predictor. A fixed random reservoir + a trained linear readout."""
    layer = "sequence"
    node_type = "esn_forecaster"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if len(x) < 20:
            return Belief("forecast", {"next_vector": [float(x[-1])]}, 0.1, self.name)
        rng = np.random.default_rng(0)            # fixed -> reproducible
        N = 64
        Win = rng.uniform(-0.5, 0.5, size=(N, 2))
        W = _scale_spectral_radius(rng.uniform(-1, 1, size=(N, N)))
        pred, conf = _esn_predict(x, Win, W)
        return Belief("forecast", {"next_vector": [pred], "reservoir": N}, conf, self.name)


@register
class DeterministicESNNode(Node):
    """Deterministic reservoir computing (TCRC-style): the reservoir weights are
    filled from a logistic-map chaotic sequence instead of random draws, removing
    the randomness while keeping the dynamics (arxiv 2501.15615)."""
    layer = "sequence"
    node_type = "deterministic_esn_forecaster"

    @staticmethod
    def _logistic_fill(n: int, r: float = 3.99, x0: float = 0.137) -> np.ndarray:
        out = np.empty(n)
        v = x0
        for i in range(n):
            v = r * v * (1.0 - v)
            out[i] = v
        return out

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        if len(x) < 20:
            return Belief("forecast", {"next_vector": [float(x[-1])]}, 0.1, self.name)
        N = 64
        seq = self._logistic_fill(N * (N + 2))
        Win = (seq[: N * 2].reshape(N, 2) - 0.5)
        W = _scale_spectral_radius((seq[N * 2:].reshape(N, N) - 0.5))
        pred, conf = _esn_predict(x, Win, W)
        return Belief("forecast", {"next_vector": [pred], "reservoir": N,
                                   "deterministic": True}, conf, self.name)


# ------------------------------------------------------- chaos / complexity
def _delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i * tau: i * tau + n] for i in range(m)])


@register
class LyapunovExponentNode(Node):
    """Largest Lyapunov exponent from data (Rosenstein method, simplified) — a
    chaos detector: >0 => sensitive dependence (deterministic chaos), ~0 => regular,
    <0 => convergent. 'Predicting chaos / horizon of predictability.'"""
    layer = "signal"
    node_type = "lyapunov_exponent"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        emb = _delay_embed(x, m=3, tau=1)
        le = 0.0
        if len(emb) > 12:
            # nearest neighbor (excluding temporal neighbors), track log-divergence
            divs = []
            for i in range(len(emb) - 1):
                d = np.linalg.norm(emb - emb[i], axis=1)
                d[max(0, i - 2): i + 3] = np.inf
                j = int(np.argmin(d))
                if not np.isfinite(d[j]) or d[j] <= 0:
                    continue
                steps = min(len(emb) - i, len(emb) - j) - 1
                if steps > 1:
                    k = min(steps, 5)
                    dist = np.linalg.norm(emb[i + k] - emb[j + k]) + 1e-12
                    divs.append(np.log(dist / d[j]) / k)
            if divs:
                le = float(np.mean(divs))
        chaotic = bool(le > 0.01)
        return Belief("signal", {"lyapunov": le, "chaotic": chaotic,
                                 "series": x.tolist()},
                      max(0.0, min(1.0, abs(le))), self.name)


@register
class RecurrenceRateNode(Node):
    """Recurrence Quantification Analysis — recurrence rate (fraction of state-space
    pairs within a threshold). High RR => repetitive/predictable structure."""
    layer = "signal"
    node_type = "recurrence_rate"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        emb = _delay_embed(x, m=3, tau=1)
        rr = 0.0
        if len(emb) > 5:
            from scipy.spatial.distance import pdist
            d = pdist(emb)
            eps = 0.2 * (np.std(x) + 1e-9)
            rr = float(np.mean(d < eps)) if d.size else 0.0
        return Belief("signal", {"recurrence_rate": rr, "series": x.tolist()},
                      max(0.0, min(1.0, rr)), self.name)


@register
class HurstExponentNode(Node):
    """Hurst exponent (rescaled-range). H>0.5 => persistent/trending (long memory),
    H<0.5 => mean-reverting, H=0.5 => random walk. Fractal long-memory measure."""
    layer = "signal"
    node_type = "hurst_exponent"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        n = len(x)
        h = 0.5
        if n >= 32:
            sizes = [s for s in (8, 16, 32, 64, 128) if s <= n]
            rs = []
            for s in sizes:
                chunks = n // s
                vals = []
                for c in range(chunks):
                    seg = x[c * s:(c + 1) * s]
                    z = seg - seg.mean()
                    Z = np.cumsum(z)
                    R = Z.max() - Z.min()
                    S = seg.std()
                    if S > 0:
                        vals.append(R / S)
                if vals:
                    rs.append((np.log(s), np.log(np.mean(vals))))
            if len(rs) >= 2:
                a = np.array(rs)
                h = float(np.polyfit(a[:, 0], a[:, 1], 1)[0])
        h = max(0.0, min(1.0, h))
        regime = "trending" if h > 0.55 else "mean_reverting" if h < 0.45 else "random_walk"
        return Belief("signal", {"hurst": h, "regime": regime, "series": x.tolist()},
                      abs(h - 0.5) * 2, self.name)


@register
class SampleEntropyNode(Node):
    """Sample Entropy (SampEn) — regularity/predictability: low => regular/predictable,
    high => complex/random."""
    layer = "signal"
    node_type = "sample_entropy"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        se = 0.0
        n = len(x)
        if n >= 20:
            m, r = 2, 0.2 * (np.std(x) + 1e-9)

            def _phi(mm):
                tmpl = np.array([x[i:i + mm] for i in range(n - mm)])
                if len(tmpl) < 2:
                    return 0
                count = 0
                for i in range(len(tmpl)):
                    d = np.max(np.abs(tmpl - tmpl[i]), axis=1)
                    count += np.sum(d <= r) - 1
                return count
            B, A = _phi(m), _phi(m + 1)
            se = float(-np.log((A + 1e-9) / (B + 1e-9))) if B > 0 else 0.0
        return Belief("signal", {"sample_entropy": se, "series": x.tolist()},
                      max(0.0, min(1.0, se / 3.0)), self.name)


@register
class PermutationEntropyNode(Node):
    """Permutation Entropy (Bandt-Pompe) — complexity from ordinal patterns; robust
    to noise, fast. Near 1 => random, near 0 => regular."""
    layer = "signal"
    node_type = "permutation_entropy"

    def _predict(self, X: np.ndarray) -> Belief:
        x = X[:, 0].astype(float)
        order = 3
        pe = 0.0
        if len(x) > order:
            from math import factorial, log
            patterns = {}
            for i in range(len(x) - order + 1):
                perm = tuple(np.argsort(x[i:i + order]))
                patterns[perm] = patterns.get(perm, 0) + 1
            total = sum(patterns.values())
            probs = np.array(list(patterns.values())) / total
            ent = -(probs * np.log(probs)).sum()
            pe = float(ent / log(factorial(order)))
        return Belief("signal", {"permutation_entropy": pe, "series": x.tolist()},
                      max(0.0, min(1.0, pe)), self.name)


# ----------------------------------------------------- econophysics (RMT)
@register
class RMTDenoiseNode(Node):
    """Random Matrix Theory denoiser (Marchenko-Pastur): clean the cross-feature
    correlation matrix by keeping only eigenvalues above the MP noise bound, then
    reconstruct — separates genuine cross-asset structure from noise. Needs D>=2;
    passes through on 1-D (no correlation matrix to clean)."""
    layer = "noise"
    node_type = "rmt_denoise"
    is_transformer = False

    def _predict(self, X: np.ndarray) -> Belief:
        T, D = X.shape
        x0 = X[:, 0].astype(float)
        if D < 2 or T < D + 2:
            return Belief("denoised", {"series": x0.tolist(), "n_signal_factors": 0,
                                       "note": "needs D>=2"}, 0.2, self.name)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        C = (Z.T @ Z) / T
        vals, vecs = np.linalg.eigh(C)
        q = D / T
        lam_plus = (1.0 + np.sqrt(q)) ** 2          # MP upper edge (sigma^2=1)
        mask = vals > lam_plus
        if not mask.any():
            mask[-1] = True                          # keep the top factor at minimum
        P = vecs[:, mask]
        Zc = (Z @ P) @ P.T
        clean0 = (Zc[:, 0] * sd[0] + mu[0])
        resid = float(np.std(x0 - clean0))
        return Belief("denoised", {"series": clean0.tolist(),
                                   "n_signal_factors": int(mask.sum()),
                                   "lambda_plus": float(lam_plus), "residual": resid},
                      max(0.0, min(1.0, int(mask.sum()) / D)), self.name)
