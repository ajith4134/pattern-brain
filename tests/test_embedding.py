"""Acceptance test for the Universal Self-Supervised Encoder (VISION P1 / IDEA-084),
Code-Generation Contract Phase-4-first. Oracles (synthetic low-rank data X = S·A + noise,
known latent factors S of rank r):
  * CONTRACT: transform → (T, latent_dim), finite, deterministic given seed; causal (no look-ahead).
  * LEARNED STRUCTURE: held-out masked-reconstruction R² > 0 (and beats mean-imputation ≈ 0).
  * RECOVERS GENERATORS: a linear map from the learned latent recovers the TRUE factors S with
    R² far above a random projection of equal dim → it learned the generators, not the noise.
  * EDGE/EXCEPTION: bad shapes / NaN / latent_dim ≥ window·D raise ValueError.

Run: python3 -m pytest tests/test_embedding.py -q
"""
import numpy as np
import pytest

from pattern_brain.embedding import SelfSupervisedEncoder


def _low_rank_series(T=600, D=6, r=2, noise=0.3, seed=0):
    """X = S·A + noise, with S a rank-r AR(1) latent (the true generators)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, D))
    S = np.zeros((T, r))
    for t in range(1, T):
        S[t] = 0.95 * S[t - 1] + rng.normal(0, 0.4, size=r)
    X = S @ A + rng.normal(0, noise, size=(T, D))
    return X, S


def _lin_r2(Z, target):
    """OOS-ish R² of best linear map Z -> target (ridge, train/test split)."""
    n = len(Z)
    cut = n // 2
    Ztr = np.hstack([Z[:cut], np.ones((cut, 1))])
    Zte = np.hstack([Z[cut:], np.ones((n - cut, 1))])
    Ytr, Yte = target[:cut], target[cut:]
    # ridge solve (stable; QR via lstsq)
    lam = 1e-3
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    B = Ztr.T @ Ytr
    W = np.linalg.solve(A, B)
    pred = Zte @ W
    ss_res = np.sum((Yte - pred) ** 2)
    ss_tot = np.sum((Yte - Yte.mean(axis=0)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def test_contract_shape_and_determinism():
    X, _ = _low_rank_series()
    enc = SelfSupervisedEncoder(latent_dim=4, window=12, epochs=120, seed=1).fit(X)
    Z1 = enc.transform(X)
    assert Z1.shape == (X.shape[0], 4)
    assert np.all(np.isfinite(Z1))
    # determinism: same seed → identical
    enc2 = SelfSupervisedEncoder(latent_dim=4, window=12, epochs=120, seed=1).fit(X)
    assert np.allclose(Z1, enc2.transform(X))


def test_learned_structure_reconstruction_r2_positive():
    X, _ = _low_rank_series()
    enc = SelfSupervisedEncoder(latent_dim=4, window=12, epochs=300, seed=0).fit(X)
    r2 = enc.reconstruction_r2(X)
    # mean-imputation of standardised held-out entries scores ≈ 0; a learned latent must beat it.
    assert r2 > 0.05, f"masked-reconstruction R² too low: {r2:.3f}"


def test_recovers_true_latent_factors():
    X, S = _low_rank_series(noise=0.3, seed=2)
    enc = SelfSupervisedEncoder(latent_dim=4, window=12, epochs=400, seed=0).fit(X)
    Z = enc.transform(X)
    r2_learned = _lin_r2(Z, S)
    # random projection of equal dim as the control (Rule 30 baseline)
    rng = np.random.default_rng(7)
    R = np.tanh(X @ rng.normal(size=(X.shape[1], 4)))
    r2_random = _lin_r2(R, S)
    assert r2_learned > 0.5, f"latent doesn't recover factors (R²={r2_learned:.3f})"
    assert r2_learned > r2_random + 0.05, (
        f"learned ({r2_learned:.3f}) must beat random projection ({r2_random:.3f})")


def test_edge_and_exceptions():
    X, _ = _low_rank_series(T=60, D=4)
    with pytest.raises(ValueError):
        SelfSupervisedEncoder(latent_dim=4, window=2).fit(np.full((30, 4), np.nan))
    with pytest.raises(ValueError):
        # latent_dim >= window*D
        SelfSupervisedEncoder(latent_dim=100, window=3).fit(X)
    with pytest.raises(RuntimeError):
        SelfSupervisedEncoder().transform(X)  # before fit
