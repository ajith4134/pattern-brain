"""Universal (self-supervised) representation encoder — VISION P1 / Stage-1 (IDEA-084).

The keystone gap in ``MODEL_NEURAL_NETWORK_VISION.md``: every node today reads either the
raw generic ``(T, D)`` series or the *deterministic* fusion from ``adapters/encoders.py``.
This module adds the missing **learned, self-supervised** encoder that compresses a window of
that ``(T, D)`` representation into one dense latent ``z`` — the shared "embedding every neuron
reads", trained with NO outcome labels (the Stage-1 self-supervised paradigm of the curriculum).

DOMAIN-AGNOSTIC (RULES.md Rule 23): this operates on ANY finite ``(T, D)`` float array. Nothing
here references candles / order books / any asset. The stock adapter feeds it; it never bends to
the data. Swap the domain → the encoder is unchanged.

================================  SPEC (CLAUDE.md Phase 1)  ================================
Algorithm: a **masked denoising auto-encoder** with an optional **temporal-contrastive** smoothness
term — the dependency-light core of the TS2Vec / PatchTST self-supervised family.

Let the input be ``X ∈ R^{T×D}`` (finite). With window ``L``:
  • Build causal windows ``w_t = vec(X[t-L+1 : t+1]) ∈ R^{m}``, ``m = L·D`` (row t uses only past+present).
  • Standardise columns of the window matrix (mean ``μ``, std ``σ``; stored, applied at transform).
  • ENCODER:   ``z = tanh(w̃_masked · W_e + b_e)``,  ``z ∈ R^{k}``  (k = latent_dim, the bottleneck k ≪ m).
  • DECODER:   ``ŵ = z · W_d + b_d``,  ``ŵ ∈ R^{m}``  (reconstruct the FULL, un-masked window).
  • Self-supervised loss (no labels):
        L = mean‖ŵ − w̃‖²            (masked-reconstruction: a fraction ``ρ`` of input entries
                                       are zeroed before encoding; the target is the full window)
          + β · mean‖z_t − z_{t-1}‖²  (temporal contrastive: adjacent windows embed near each other —
                                       TS2Vec's "nearby timestamps are positive pairs", smoothness form)
  • Trained by deterministic full-batch gradient descent with momentum (seeded), float64.

I/O contract:
  • fit(X): X array-like, shape (T, D), T ≥ window+2, finite. Returns self. No y (self-supervised).
  • transform(X) -> (T, k): per-timestep causal latent; the first ``window-1`` rows are back-filled
    with the first computable latent (no look-ahead). Rows sum to nothing special; range ≈ (-1, 1) (tanh).
  • encode(X) -> (n, k): the raw per-window latent matrix, n = T-window+1.
  • reconstruction_r2(X) -> float: held-out masked-entry R² (in [≤1]); > 0 ⇒ learned real structure.
Pre-conditions: finite X; window ≥ 2; 1 ≤ latent_dim < m. Exceptions: ValueError on shape/NaN/degenerate.
Post-conditions: transform output shape == (T, latent_dim), finite; deterministic given seed.

Edge cases handled: T == window (one window), D == 1, constant column (σ→1 guard), latent_dim ≥ m
(raise), all-NaN (raise), mask_frac 0 (pure auto-encoder) or →1 (guarded < 1).

================================  ORACLE (CLAUDE.md Phase 4)  ===============================
Verified in ``tests/test_embedding.py`` on synthetic low-rank data ``X = S·A + noise`` with known
latent factors S (rank r): (1) held-out masked-reconstruction R² > 0 and > mean-imputation; (2) a
linear map from the learned latent z recovers the TRUE factors S with R² far above a random
projection of equal dim (it learned the generators, not noise). Effectiveness vs raw features on
the Rule-32 panel (downstream probe, OOS) is the NEXT gate before any KEEP — not claimed here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _finite_2d(X) -> np.ndarray:
    """Validate → float64 (T, D), finite, non-empty."""
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"embedding input must be 2-D (T, D); got shape {A.shape}")
    if A.size == 0:
        raise ValueError("embedding input is empty")
    if not np.all(np.isfinite(A)):
        raise ValueError("embedding input contains non-finite values")
    return A


class SelfSupervisedEncoder:
    """Masked denoising auto-encoder with temporal-contrastive smoothness (VISION P1).

    Learned, label-free representation encoder. See module docstring for the full spec.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        window: int = 16,
        mask_frac: float = 0.3,
        contrastive_weight: float = 0.1,
        epochs: int = 300,
        lr: float = 0.05,
        momentum: float = 0.9,
        seed: int = 0,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if not (0.0 <= mask_frac < 1.0):
            raise ValueError("mask_frac must be in [0, 1)")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self.latent_dim = int(latent_dim)
        self.window = int(window)
        self.mask_frac = float(mask_frac)
        self.contrastive_weight = float(contrastive_weight)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self._fitted = False

    # ------------------------------------------------------------------ windowing
    def _windows(self, A: np.ndarray) -> np.ndarray:
        """(T, D) -> (n, window*D) causal window matrix, n = T-window+1."""
        T, D = A.shape
        L = self.window
        if T < L:
            raise ValueError(f"need at least window={L} rows; got T={T}")
        n = T - L + 1
        # row i = flattened A[i : i+L]
        idx = np.arange(L)[None, :] + np.arange(n)[:, None]      # (n, L)
        return A[idx].reshape(n, L * D)                          # (n, L*D)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y: Optional[np.ndarray] = None) -> "SelfSupervisedEncoder":
        A = _finite_2d(X)
        self._D = A.shape[1]
        W = self._windows(A)                                    # (n, m)
        m = W.shape[1]
        if self.latent_dim >= m:
            raise ValueError(f"latent_dim ({self.latent_dim}) must be < window*D ({m})")

        # standardise columns (σ guard for constant columns)
        self._mu = W.mean(axis=0)
        self._sd = W.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        Ws = (W - self._mu) / self._sd                          # (n, m)

        rng = np.random.default_rng(self.seed)
        k = self.latent_dim
        # small random init (Xavier-ish), float64
        We = rng.normal(0, 1.0 / np.sqrt(m), size=(m, k))
        be = np.zeros(k)
        Wd = rng.normal(0, 1.0 / np.sqrt(k), size=(k, m))
        bd = np.zeros(m)

        vWe = np.zeros_like(We); vbe = np.zeros_like(be)
        vWd = np.zeros_like(Wd); vbd = np.zeros_like(bd)

        n = Ws.shape[0]
        beta = self.contrastive_weight
        # first-difference operator applied implicitly for the contrastive term
        for _ in range(self.epochs):
            # fresh denoising mask each epoch (Bernoulli keep)
            if self.mask_frac > 0:
                keep = (rng.random(Ws.shape) >= self.mask_frac).astype(np.float64)
                Win = Ws * keep
            else:
                Win = Ws
            pre = Win @ We + be                                 # (n, k)
            z = np.tanh(pre)
            Wh = z @ Wd + bd                                    # (n, m)  reconstruction
            resid = Wh - Ws                                     # target = FULL window
            # --- gradients: reconstruction MSE (mean over all entries) ---
            dWh = (2.0 / (n * m)) * resid                       # (n, m)
            gWd = z.T @ dWh
            gbd = dWh.sum(axis=0)
            dz = dWh @ Wd.T                                     # (n, k)
            # --- temporal contrastive: beta * mean||z_t - z_{t-1}||^2 ---
            if beta > 0 and n > 1:
                diff = np.diff(z, axis=0)                       # (n-1, k)
                gc = np.zeros_like(z)
                scale = beta * (2.0 / ((n - 1) * k))
                gc[1:] += scale * diff
                gc[:-1] -= scale * diff
                dz = dz + gc
            dpre = dz * (1.0 - z * z)                           # tanh'
            gWe = Win.T @ dpre
            gbe = dpre.sum(axis=0)
            # --- momentum SGD update ---
            vWe = self.momentum * vWe - self.lr * gWe; We += vWe
            vbe = self.momentum * vbe - self.lr * gbe; be += vbe
            vWd = self.momentum * vWd - self.lr * gWd; Wd += vWd
            vbd = self.momentum * vbd - self.lr * gbd; bd += vbd

        self._We, self._be, self._Wd, self._bd = We, be, Wd, bd
        self._m = m
        self._fitted = True
        return self

    # ------------------------------------------------------------------ encode / transform
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SelfSupervisedEncoder.encode/transform called before fit()")

    def encode(self, X) -> np.ndarray:
        """(T, D) -> (n, latent_dim) per-window latent (no masking at inference)."""
        self._check_fitted()
        A = _finite_2d(X)
        if A.shape[1] != self._D:
            raise ValueError(f"expected D={self._D}; got {A.shape[1]}")
        W = self._windows(A)
        Ws = (W - self._mu) / self._sd
        return np.tanh(Ws @ self._We + self._be)

    def transform(self, X) -> np.ndarray:
        """(T, D) -> (T, latent_dim) causal per-timestep latent (back-fill leading rows)."""
        A = _finite_2d(X)
        Z = self.encode(A)                                      # (n, k)
        T = A.shape[0]
        pad = T - Z.shape[0]                                    # = window-1
        if pad > 0:
            Z = np.vstack([np.repeat(Z[:1], pad, axis=0), Z])
        return Z

    def reconstruct(self, X) -> np.ndarray:
        """(T, D) -> (n, m) reconstructed (un-masked) windows, in standardised space."""
        Z = self.encode(X)
        return Z @ self._Wd + self._bd

    def reconstruction_r2(self, X, mask_frac: Optional[float] = None, seed: int = 12345) -> float:
        """Held-out masked-entry R²: mask entries, reconstruct, score ONLY masked positions
        against the standardised truth. > 0 ⇒ the latent carries genuine cross-entry structure."""
        self._check_fitted()
        A = _finite_2d(X)
        W = self._windows(A)
        Ws = (W - self._mu) / self._sd
        rho = self.mask_frac if mask_frac is None else float(mask_frac)
        rng = np.random.default_rng(seed)
        mask = rng.random(Ws.shape) < max(rho, 1e-6)            # True = held out
        Win = Ws.copy(); Win[mask] = 0.0
        z = np.tanh(Win @ self._We + self._be)
        Wh = z @ self._Wd + self._bd
        truth = Ws[mask]; pred = Wh[mask]
        ss_res = float(np.sum((truth - pred) ** 2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot


__all__ = ["SelfSupervisedEncoder"]
