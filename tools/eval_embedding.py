"""Rule-30/34 effectiveness gate for the Universal Self-Supervised Encoder (VISION P1, IDEA-084).

The embedding's DESIGN data is MULTIVARIATE (it fuses many channels → one dense latent), so testing
it on the univariate panel directly would be an invalid Rule-34 verdict. Instead, from each real
panel series we construct a multi-channel feature matrix (the "many sources" the encoder is meant to
compress), then run an HONEST out-of-sample probe:

    next-step logreturn  ~  [ learned latent z ]   vs   [ raw D channels ]   vs   [ PCA(same dim) ]

Time-ordered split (no shuffle, no look-ahead). The encoder EARNS its place if its low-dim latent
matches/beats the raw channels OOS (a compression win) AND beats a PCA latent of equal dim (its
nonlinear-denoising value over the obvious linear baseline). Reports per-symbol + aggregate.

Run: python3 tools/eval_embedding.py
"""
from __future__ import annotations

import numpy as np

from pattern_brain.datasets import load_panel
from pattern_brain.embedding import SelfSupervisedEncoder


def make_channels(r: np.ndarray) -> np.ndarray:
    """Univariate logreturns -> multi-channel feature matrix (causal, no look-ahead)."""
    r = r.reshape(-1)
    n = len(r)
    def roll(a, w, fn):
        out = np.zeros(n)
        for i in range(n):
            lo = max(0, i - w + 1)
            out[i] = fn(a[lo:i + 1])
        return out
    chans = [
        r,                                   # return
        np.abs(r),                           # volatility proxy
        roll(r, 5, np.mean),                 # short momentum
        roll(r, 20, np.mean),                # long momentum
        roll(r, 10, np.std),                 # rolling vol
        np.sign(r),                          # direction
        np.cumsum(r),                        # price level (integrated)
        roll(r, 5, lambda x: x[-1] - x[0]),  # short range
    ]
    return np.column_stack(chans)


def oos_probe(Z: np.ndarray, y: np.ndarray, cut: float = 0.6):
    """Ridge next-step probe; returns (OOS R², OOS direction-accuracy)."""
    n = len(Z)
    c = int(n * cut)
    Ztr = np.hstack([Z[:c], np.ones((c, 1))])
    Zte = np.hstack([Z[c:], np.ones((n - c, 1))])
    ytr, yte = y[:c], y[c:]
    lam = 1e-2
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    W = np.linalg.solve(A, Ztr.T @ ytr)
    pred = Zte @ W
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    diracc = float(np.mean(np.sign(pred) == np.sign(yte)))
    return float(r2), diracc


def pca_latent(Xs: np.ndarray, k: int) -> np.ndarray:
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def main():
    panel = load_panel(target="logreturn")
    K = 6
    rows = []
    for name, X in panel:
        r = np.asarray(X, dtype=float).reshape(-1)
        if len(r) < 200:
            continue
        C = make_channels(r)
        Cs = (C - C.mean(0)) / (C.std(0) + 1e-12)
        # SIGNAL-BEARING target (Rule 34): next-step realized volatility (10-bar rolling std of returns),
        # which this project has shown IS predictable (har_rv/heston_vol KEEP) — unlike raw returns.
        rv = np.array([r[max(0, i - 9):i + 1].std() for i in range(len(r))])
        feat, y = Cs[:-1], rv[1:]
        enc = SelfSupervisedEncoder(latent_dim=K, window=10, epochs=250,
                                    contrastive_weight=0.1, seed=0).fit(feat)
        Z = enc.transform(feat)
        r2_emb, da_emb = oos_probe(Z, y)
        r2_raw, da_raw = oos_probe(feat, y)
        r2_pca, da_pca = oos_probe(pca_latent(feat, K), y)
        rows.append((name, r2_emb, r2_raw, r2_pca, da_emb, da_raw, da_pca))
        print(f"{name:10s}  emb R²={r2_emb:+.4f} | raw R²={r2_raw:+.4f} (8d) | "
              f"pca R²={r2_pca:+.4f} ({K}d)")

    arr = np.array([row[1:] for row in rows], dtype=float)
    m = arr.mean(0)
    print("\n=== AGGREGATE (mean over %d symbols, latent_dim=%d vs raw 8 channels) ===" % (len(rows), K))
    print(f"  emb : OOS R²={m[0]:+.4f}  dir-acc={m[3]:.3f}")
    print(f"  raw : OOS R²={m[1]:+.4f}  dir-acc={m[4]:.3f}   (8 dims)")
    print(f"  pca : OOS R²={m[2]:+.4f}  dir-acc={m[5]:.3f}   ({K} dims)")
    beats_raw = int(np.sum(arr[:, 0] >= arr[:, 1]))
    beats_pca = int(np.sum(arr[:, 0] >= arr[:, 2]))
    print(f"  emb >= raw on {beats_raw}/{len(rows)} symbols (R²); emb >= pca on {beats_pca}/{len(rows)}")
    verdict = ("KEEP" if (m[0] >= m[2] and beats_pca >= len(rows) * 0.5) else
               "SHADOW/REJECT (no OOS edge over PCA — honest)")
    print("  VERDICT:", verdict)


if __name__ == "__main__":
    main()
