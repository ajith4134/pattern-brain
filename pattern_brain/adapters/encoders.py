"""Phase 8 — slice 2: multimodal ENCODER adapters (PLAN.md §12).

The "any raw data in" layer. Owner's decision (§12 fork 2): the project ingests
*any* raw data — time-series, tabular, text, image, audio — and each modality is
turned into the ONE generic representation the whole bank already consumes: a
finite ``(T, D)`` array (the shared space). Different modalities sampled at
different rates are fused causally (no look-ahead) onto one grid via
:func:`align_sources`.

Rule 23 (mandatory): this lives in ``adapters/`` — the sanctioned home for any
modality/domain-specific code — and is **NOT** imported by the core package, so
the core stays modality-agnostic. The dependency flows adapter → core only
(encoders reuse ``knowledge.HashingEmbedder`` for text and ``align_sources`` for
fusion). The OUTPUT is a generic ``(T, D)`` that runs through the existing bank /
Connector unchanged.

Staging (PLAN.md §12, honest): time-series + tabular + text encoders are real and
dependency-free now; image/audio are light deterministic feature encoders (real,
runnable, no heavy deps) that a pretrained CNN/audio model can replace later — the
input-router PATTERN is complete now so the architecture is multimodal-ready.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..knowledge import HashingEmbedder, Embedder
from .stock import align_sources

RawSpec = Union[np.ndarray, Sequence, Tuple[Sequence[float], object]]


def _finite_2d(X: np.ndarray) -> np.ndarray:
    """Coerce to a finite (T, D) float array (the core's Node input contract):
    1-D → (T, 1); non-finite values forward-filled then zero-filled."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"encoder output must be (T, D); got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        for j in range(X.shape[1]):                     # causal ffill per column
            col = X[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                last = 0.0
                for i in range(len(col)):
                    if np.isfinite(col[i]):
                        last = col[i]
                    else:
                        col[i] = last
                X[:, j] = col
        X[~np.isfinite(X)] = 0.0
    return X


class Encoder(ABC):
    """A modality → shared ``(T, D)`` encoder. ``modality`` names the input kind."""
    modality: str = "abstract"

    @abstractmethod
    def encode(self, raw) -> np.ndarray:
        """Return a finite ``(T, D)`` array the bank can consume."""


class TimeSeriesEncoder(Encoder):
    """Numeric time-series (T,) or (T, D) → (T, D). Optional z-score standardize."""
    modality = "timeseries"

    def __init__(self, standardize: bool = False) -> None:
        self.standardize = standardize

    def encode(self, raw) -> np.ndarray:
        X = _finite_2d(raw)
        if self.standardize:
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        return _finite_2d(X)


class TabularEncoder(Encoder):
    """A 2-D table (rows = time/sample, cols = features) → (T, D). Non-finite cells
    imputed to the column mean; optional standardize."""
    modality = "tabular"

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize

    def encode(self, raw) -> np.ndarray:
        X = np.asarray(raw, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for j in range(X.shape[1]):                     # mean-impute over FINITE cells (handles NaN and inf)
            col = X[:, j]
            finite = np.isfinite(col)
            m = float(col[finite].mean()) if finite.any() else 0.0
            col[~finite] = m
            X[:, j] = col
        if self.standardize:
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        return _finite_2d(X)


class TextEncoder(Encoder):
    """A sequence of documents (one per time-step) → (T, D) embeddings. Uses the
    dependency-free :class:`HashingEmbedder` by default (deterministic, offline);
    inject any :class:`Embedder` (nomic/bge/ollama) for higher quality."""
    modality = "text"

    def __init__(self, dim: int = 64, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or HashingEmbedder(dim=dim)
        self.dim = self.embedder.dim

    def encode(self, raw: Sequence[str]) -> np.ndarray:
        docs = list(raw)
        if not docs:
            return np.zeros((1, self.dim))
        return _finite_2d(self.embedder.embed_batch([str(d) for d in docs]))


class ImageEncoder(Encoder):
    """A sequence of images (each a 2-D/3-D numpy array) → (T, D) of light
    deterministic features (intensity mean/std, a small histogram, gradient
    energy). Real and dependency-free — replace with a pretrained CNN later."""
    modality = "image"

    def __init__(self, bins: int = 8) -> None:
        self.bins = bins

    def _feat(self, img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=float)
        if a.ndim == 3:
            a = a.mean(axis=2)                          # to grayscale
        a = np.atleast_2d(a)
        flat = a.ravel()
        rng = (float(np.min(flat)), float(np.max(flat)) + 1e-9)
        hist, _ = np.histogram(flat, bins=self.bins, range=rng, density=True)
        gx = float(np.mean(np.abs(np.diff(a, axis=1)))) if a.shape[1] > 1 else 0.0
        gy = float(np.mean(np.abs(np.diff(a, axis=0)))) if a.shape[0] > 1 else 0.0
        return np.concatenate([[float(flat.mean()), float(flat.std()), gx, gy], hist])

    def encode(self, raw: Sequence[np.ndarray]) -> np.ndarray:
        imgs = list(raw)
        if not imgs:
            return np.zeros((1, self.bins + 4))
        return _finite_2d(np.vstack([self._feat(im) for im in imgs]))


class AudioEncoder(Encoder):
    """A sequence of waveforms (each a 1-D numpy array) → (T, D) of light
    deterministic audio features (RMS, zero-crossing rate, spectral centroid, a
    few band energies). Real and dependency-free — replace with a learned audio
    encoder later."""
    modality = "audio"

    def __init__(self, bands: int = 4) -> None:
        self.bands = bands

    def _feat(self, wave: np.ndarray) -> np.ndarray:
        x = np.asarray(wave, dtype=float).ravel()
        if x.size < 2:
            return np.zeros(3 + self.bands)
        rms = float(np.sqrt(np.mean(x ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
        spec = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(x.size)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
        bands = np.array([float(b.sum()) for b in np.array_split(spec, self.bands)])
        bands = bands / (bands.sum() + 1e-12)
        return np.concatenate([[rms, zcr, centroid], bands])

    def encode(self, raw: Sequence[np.ndarray]) -> np.ndarray:
        clips = list(raw)
        if not clips:
            return np.zeros((1, 3 + self.bands))
        return _finite_2d(np.vstack([self._feat(c) for c in clips]))


# ----------------------------------------------------------------- input router
_DEFAULT_ENCODERS = {
    "timeseries": TimeSeriesEncoder,
    "tabular": TabularEncoder,
    "text": TextEncoder,
    "image": ImageEncoder,
    "audio": AudioEncoder,
}


class MultiModalEncoder:
    """The input router: takes a dict of ``{modality: raw}`` (or
    ``{name: (timestamps, raw, modality)}``), encodes each modality to ``(T_i, D_i)``
    and fuses them onto ONE shared causal grid via :func:`align_sources` — the
    single ``(T, D)`` shared representation the whole bank consumes. A single
    modality is returned as-is (no fusion needed)."""

    def __init__(self, encoders: Optional[Dict[str, Encoder]] = None) -> None:
        self.encoders = encoders or {m: cls() for m, cls in _DEFAULT_ENCODERS.items()}

    def encode_one(self, modality: str, raw) -> np.ndarray:
        if modality not in self.encoders:
            raise KeyError(f"no encoder for modality {modality!r}; have {sorted(self.encoders)}")
        return self.encoders[modality].encode(raw)

    def encode(self, inputs: Dict[str, object], step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """``inputs`` maps a name → either ``raw`` (modality inferred from the name's
        prefix or the name itself) or ``(timestamps, raw, modality)``. Returns
        ``(grid, fused (T, D))``."""
        if not inputs:
            raise ValueError("no inputs to encode")
        sources: List[Tuple[Sequence[float], np.ndarray]] = []
        for name, spec in inputs.items():
            ts = None
            if isinstance(spec, tuple) and len(spec) == 3:
                ts, raw, modality = spec
            else:
                raw, modality = spec, self._infer_modality(name)
            enc = self.encode_one(modality, raw)
            if ts is None:
                ts = np.arange(enc.shape[0], dtype=float)
            sources.append((np.asarray(ts, dtype=float), enc))
        if len(sources) == 1:
            ts, enc = sources[0]
            return ts, _finite_2d(enc)
        grid, fused = align_sources(sources, step=step)
        return grid, _finite_2d(fused)

    @staticmethod
    def _infer_modality(name: str) -> str:
        n = name.lower()
        for m in _DEFAULT_ENCODERS:
            if m in n:
                return m
        return "timeseries"


__all__ = [
    "Encoder", "TimeSeriesEncoder", "TabularEncoder", "TextEncoder",
    "ImageEncoder", "AudioEncoder", "MultiModalEncoder",
]
