"""Universal (self-supervised) representation encoder — VISION P1 / Stage-1 (IDEA-084, IDEA-088).

The keystone gap in ``MODEL_NEURAL_NETWORK_VISION.md``: every node today reads either the
raw generic ``(T, D)`` series or the *deterministic* fusion from ``adapters/encoders.py``.
This module adds the missing **learned, self-supervised** encoder that compresses a window of
that ``(T, D)`` representation into one dense latent ``z`` — the shared "embedding every neuron
reads", trained with NO outcome labels (the Stage-1 self-supervised paradigm of the curriculum).

DOMAIN-AGNOSTIC (RULES.md Rule 23): operates on ANY finite ``(T, D)`` float array; nothing here
references candles / order books / any asset. The stock adapter feeds it; it never bends to data.

================================  TWO PRETEXT OBJECTIVES  ==================================
``objective="forecast"`` (DEFAULT, IDEA-088 — the KEEP path):
    **forecasting-aligned / next-observation prediction** (the Contrastive-Predictive-Coding
    family core). The latent ``z_t`` is trained to PREDICT the near-future standardised
    observation ``x_{t+h}`` — so it retains the directions that forecast what's coming, not the
    ones that merely reconstruct the past. This is *why* TS2Vec/PatchTST/CPC representations
    transfer downstream. Replaces the original masked-reconstruction objective, which (verified
    2026-06-24) preserved reconstructive structure but DESTROYED predictive signal vs a plain PCA.

``objective="reconstruction"`` (the original masked denoising auto-encoder):
    mask a fraction ``ρ`` of input entries, reconstruct the full window; a temporal-contrastive
    smoothness term pulls adjacent latents together. Kept for ablation / structure-recovery oracle.

================================  SPEC (CLAUDE.md Phase 1)  ================================
Input ``X ∈ R^{T×D}`` (finite). Standardise per channel (μ,σ over T; stored). Window ``L``:
  ``w_t = vec(X̃[t-L+1 : t+1]) ∈ R^{m}``, ``m = L·D`` (causal: row t uses only past+present).
  ENCODER:  ``z_t = tanh(w_t · W_e + b_e) ∈ R^{k}``  (k = latent_dim, bottleneck k ≪ m).
  FORECAST head:  ``x̂_{t+h} = z_t · W_p + b_p ∈ R^{D}``;  loss ``= mean‖x̂_{t+h} − x̃_{t+h}‖²``.
  RECON   head:  ``ŵ_t = z_t · W_d + b_d ∈ R^{m}``;  loss ``= mean‖ŵ_t − w_t‖² + β·mean‖z_t−z_{t-1}‖²``.
  Optimised by deterministic momentum gradient descent (seeded), float64.

I/O contract:
  • fit(X): X (T, D), T ≥ window+horizon+2, finite, no y. Returns self.
  • transform(X) -> (T, k): per-timestep causal latent; first ``window-1`` rows back-filled (no look-ahead).
  • encode(X) -> (n, k): per-window latent, n = T-window+1.
  • self_score(X) -> float: held-out pretext R² (forecast: future-prediction R²; recon: masked-recon R²).
Pre-conditions: finite X; window ≥ 2; horizon ≥ 1; 1 ≤ latent_dim < m. Exceptions: ValueError / RuntimeError.
Post-conditions: transform shape == (T, latent_dim), finite, deterministic given seed.

================================  ORACLE (tests/test_embedding.py)  ========================
forecast: on a predictable AR series the latent predicts the future far above the naive-mean
baseline. reconstruction: recovers the TRUE latent factors of synthetic low-rank data (R² > random
projection) + masked-recon R² > 0. Effectiveness vs PCA/raw on the Rule-32 panel (OOS vol probe,
``tools/eval_embedding.py``) is the KEEP gate — judged on the SIGNAL-BEARING target, not raw returns.
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
    """Learned, label-free representation encoder (VISION P1). Two pretext objectives:
    ``forecast`` (default, next-observation prediction) and ``reconstruction`` (masked AE).
    See module docstring for the full spec."""

    def __init__(
        self,
        latent_dim: int = 8,
        window: int = 16,
        objective: str = "forecast",
        horizon: int = 1,
        mask_frac: float = 0.15,
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
        if objective not in ("forecast", "reconstruction"):
            raise ValueError("objective must be 'forecast' or 'reconstruction'")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.latent_dim = int(latent_dim)
        self.window = int(window)
        self.objective = objective
        self.horizon = int(horizon)
        self.mask_frac = float(mask_frac)
        self.contrastive_weight = float(contrastive_weight)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self._fitted = False

    # ------------------------------------------------------------------ helpers
    def _standardize_fit(self, A: np.ndarray) -> np.ndarray:
        self._mu = A.mean(axis=0)
        self._sd = A.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        return (A - self._mu) / self._sd

    def _standardize(self, A: np.ndarray) -> np.ndarray:
        return (A - self._mu) / self._sd

    def _windows(self, As: np.ndarray) -> np.ndarray:
        """standardised (T, D) -> (n, window*D) causal window matrix, n = T-window+1."""
        T, D = As.shape
        L = self.window
        if T < L:
            raise ValueError(f"need at least window={L} rows; got T={T}")
        n = T - L + 1
        idx = np.arange(L)[None, :] + np.arange(n)[:, None]      # (n, L)
        return As[idx].reshape(n, L * D)                         # (n, L*D)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y: Optional[np.ndarray] = None) -> "SelfSupervisedEncoder":
        A = _finite_2d(X)
        self._D = D = A.shape[1]
        As = self._standardize_fit(A)
        W = self._windows(As)                                    # (n, m)
        n, m = W.shape
        if self.latent_dim >= m:
            raise ValueError(f"latent_dim ({self.latent_dim}) must be < window*D ({m})")

        rng = np.random.default_rng(self.seed)
        k, L, h = self.latent_dim, self.window, self.horizon
        We = rng.normal(0, 1.0 / np.sqrt(m), size=(m, k)); be = np.zeros(k)

        if self.objective == "forecast":
            if n - h < 5:
                raise ValueError("series too short for the requested window+horizon")
            Xin = W[:n - h]                                      # (n-h, m)
            # target = the standardised observation h steps after each window's end
            Y = As[L - 1 + h: L - 1 + h + (n - h)]               # (n-h, D)
            Wp = rng.normal(0, 1.0 / np.sqrt(k), size=(k, D)); bp = np.zeros(D)
            self._We, self._be, self._Wp, self._bp = self._train_forecast(
                Xin, Y, We, be, Wp, bp, rng)
        else:
            Wd = rng.normal(0, 1.0 / np.sqrt(k), size=(k, m)); bd = np.zeros(m)
            self._We, self._be, self._Wd, self._bd = self._train_reconstruction(
                W, We, be, Wd, bd, rng)

        self._m = m
        self._fitted = True
        return self

    # ------------------------------------------------------------------ forecast objective
    def _train_forecast(self, Xin, Y, We, be, Wp, bp, rng):
        n, m = Xin.shape
        D = Y.shape[1]
        vWe = np.zeros_like(We); vbe = np.zeros_like(be)
        vWp = np.zeros_like(Wp); vbp = np.zeros_like(bp)
        for _ in range(self.epochs):
            if self.mask_frac > 0:                               # denoising for robustness
                Win = Xin * (rng.random(Xin.shape) >= self.mask_frac)
            else:
                Win = Xin
            z = np.tanh(Win @ We + be)                          # (n, k)
            pred = z @ Wp + bp                                  # (n, D)
            resid = pred - Y
            dpred = (2.0 / (n * D)) * resid
            gWp = z.T @ dpred; gbp = dpred.sum(axis=0)
            dz = dpred @ Wp.T
            dpre = dz * (1.0 - z * z)
            gWe = Win.T @ dpre; gbe = dpre.sum(axis=0)
            vWe = self.momentum * vWe - self.lr * gWe; We += vWe
            vbe = self.momentum * vbe - self.lr * gbe; be += vbe
            vWp = self.momentum * vWp - self.lr * gWp; Wp += vWp
            vbp = self.momentum * vbp - self.lr * gbp; bp += vbp
        return We, be, Wp, bp

    # ------------------------------------------------------------------ reconstruction objective
    def _train_reconstruction(self, W, We, be, Wd, bd, rng):
        n, m = W.shape
        beta = self.contrastive_weight
        k = We.shape[1]
        vWe = np.zeros_like(We); vbe = np.zeros_like(be)
        vWd = np.zeros_like(Wd); vbd = np.zeros_like(bd)
        for _ in range(self.epochs):
            Win = W * (rng.random(W.shape) >= self.mask_frac) if self.mask_frac > 0 else W
            z = np.tanh(Win @ We + be)
            Wh = z @ Wd + bd
            resid = Wh - W
            dWh = (2.0 / (n * m)) * resid
            gWd = z.T @ dWh; gbd = dWh.sum(axis=0)
            dz = dWh @ Wd.T
            if beta > 0 and n > 1:
                diff = np.diff(z, axis=0)
                gc = np.zeros_like(z); scale = beta * (2.0 / ((n - 1) * k))
                gc[1:] += scale * diff; gc[:-1] -= scale * diff
                dz = dz + gc
            dpre = dz * (1.0 - z * z)
            gWe = Win.T @ dpre; gbe = dpre.sum(axis=0)
            vWe = self.momentum * vWe - self.lr * gWe; We += vWe
            vbe = self.momentum * vbe - self.lr * gbe; be += vbe
            vWd = self.momentum * vWd - self.lr * gWd; Wd += vWd
            vbd = self.momentum * vbd - self.lr * gbd; bd += vbd
        return We, be, Wd, bd

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
        W = self._windows(self._standardize(A))
        return np.tanh(W @ self._We + self._be)

    def transform(self, X) -> np.ndarray:
        """(T, D) -> (T, latent_dim) causal per-timestep latent (back-fill leading rows)."""
        A = _finite_2d(X)
        Z = self.encode(A)
        pad = A.shape[0] - Z.shape[0]
        if pad > 0:
            Z = np.vstack([np.repeat(Z[:1], pad, axis=0), Z])
        return Z

    # ------------------------------------------------------------------ self-supervised diagnostics
    def self_score(self, X) -> float:
        """Pretext R² on the given data (forecast: future-prediction; recon: masked-recon)."""
        self._check_fitted()
        if self.objective == "forecast":
            return self._forecast_r2(X)
        return self.reconstruction_r2(X)

    def _forecast_r2(self, X) -> float:
        A = _finite_2d(X); As = self._standardize(A)
        W = self._windows(As); n = W.shape[0]; L, h = self.window, self.horizon
        if n - h < 1:
            return float("nan")
        z = np.tanh(W[:n - h] @ self._We + self._be)
        pred = z @ self._Wp + self._bp
        Y = As[L - 1 + h: L - 1 + h + (n - h)]
        ss_res = float(np.sum((Y - pred) ** 2))
        ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    def reconstruction_r2(self, X, mask_frac: Optional[float] = None, seed: int = 12345) -> float:
        """Held-out masked-entry R² (reconstruction objective only). > 0 ⇒ learned cross-entry structure."""
        self._check_fitted()
        if self.objective != "reconstruction":
            raise RuntimeError("reconstruction_r2 only applies to objective='reconstruction'")
        A = _finite_2d(X); W = self._windows(self._standardize(A))
        rho = self.mask_frac if mask_frac is None else float(mask_frac)
        rng = np.random.default_rng(seed)
        mask = rng.random(W.shape) < max(rho, 1e-6)
        Win = W.copy(); Win[mask] = 0.0
        z = np.tanh(Win @ self._We + self._be)
        Wh = z @ self._Wd + self._bd
        truth = W[mask]; pred = Wh[mask]
        ss_res = float(np.sum((truth - pred) ** 2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot


__all__ = ["SelfSupervisedEncoder"]
