"""Tests for Phase 8 slice 2 — multimodal encoder adapters (PLAN.md §12).

Run: python3 tests/test_encoders.py  (light stack — numpy).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.node import Node
from pattern_brain.adapters.encoders import (
    TimeSeriesEncoder, TabularEncoder, TextEncoder, ImageEncoder, AudioEncoder,
    MultiModalEncoder,
)

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _valid(X, msg):
    """Encoder output must satisfy the core's Node input contract exactly."""
    try:
        V = Node._validate(X)
        check(V.ndim == 2 and V.shape[0] >= 1 and V.shape[1] >= 1, f"{msg}: bad shape {V.shape}")
        return True
    except Exception as e:
        check(False, f"{msg}: failed Node._validate ({e})")
        return False


def test_each_encoder_returns_valid_TD():
    rng = np.random.default_rng(0)
    # timeseries: 1-D -> (T,1)
    ts = TimeSeriesEncoder().encode(rng.normal(size=60))
    check(ts.shape == (60, 1), f"timeseries 1-D should be (60,1), got {ts.shape}")
    _valid(ts, "timeseries")
    # tabular with NaNs -> imputed, valid
    tab = np.column_stack([rng.normal(size=40), rng.normal(size=40)])
    tab[5, 0] = np.nan; tab[7, 1] = np.inf
    enc = TabularEncoder().encode(tab)
    check(enc.shape == (40, 2) and np.all(np.isfinite(enc)), "tabular NaN/inf not cleaned")
    _valid(enc, "tabular")
    print("  timeseries + tabular encoders return finite (T,D) (NaN/inf imputed)")


def test_text_encoder_deterministic_and_distinct():
    enc = TextEncoder(dim=16)
    a = enc.encode(["bitcoin is bullish today", "market crashed hard"])
    b = enc.encode(["bitcoin is bullish today", "market crashed hard"])
    check(a.shape == (2, 16), f"text encode shape wrong: {a.shape}")
    check(np.allclose(a, b), "text encoder must be deterministic for the same input")
    check(not np.allclose(a[0], a[1]), "different documents must embed differently")
    _valid(a, "text")
    print(f"  text encoder: deterministic + distinct embeddings, shape {a.shape}")


def test_image_and_audio_encoders():
    rng = np.random.default_rng(1)
    imgs = [rng.random((12, 12)), rng.random((12, 12, 3))]      # grayscale + RGB
    ie = ImageEncoder(bins=8).encode(imgs)
    check(ie.shape[0] == 2 and np.all(np.isfinite(ie)), f"image encode bad: {ie.shape}")
    _valid(ie, "image")
    waves = [rng.normal(size=400), np.sin(np.linspace(0, 50, 400))]
    ae = AudioEncoder(bands=4).encode(waves)
    check(ae.shape[0] == 2 and np.all(np.isfinite(ae)), f"audio encode bad: {ae.shape}")
    _valid(ae, "audio")
    print(f"  image {ie.shape} + audio {ae.shape} encoders produce finite feature rows")


def test_multimodal_fusion_feeds_the_bank():
    """The router fuses two modalities at different lengths onto one causal grid,
    and the unified (T,D) runs UNCHANGED through a real bank node (Rule 21)."""
    rng = np.random.default_rng(2)
    inputs = {
        "timeseries_price": np.cumsum(rng.normal(size=50)),       # 50 steps
        "text_news": [f"headline {i%3}" for i in range(40)],      # 40 steps, dim 8
    }
    mm = MultiModalEncoder({"timeseries": TimeSeriesEncoder(),
                            "text": TextEncoder(dim=8)})
    grid, fused = mm.encode(inputs, step=1.0)
    check(fused.shape[0] == 40, f"fusion should align to the shorter overlap (40), got {fused.shape[0]}")
    check(fused.shape[1] == 1 + 8, f"fused D should be 1(ts)+8(text)=9, got {fused.shape[1]}")
    _valid(fused, "fused")
    # feed the shared representation into a real registered bank node
    from pattern_brain.registry import create
    belief = create("naive_mean_forecast").predict(fused)
    check(belief.type == "forecast" and np.isfinite(belief.confidence),
          "fused multimodal (T,D) failed to run through a bank node")
    print(f"  multimodal fusion -> shared (T,D)={fused.shape}; bank node consumed it "
          f"(belief={belief.type})")


def test_single_modality_passthrough():
    mm = MultiModalEncoder()
    grid, out = mm.encode({"timeseries": np.arange(30.0)})
    check(out.shape == (30, 1), f"single-modality passthrough shape wrong: {out.shape}")
    print("  single modality returns its encoding directly (no fusion)")


def test_rule23_core_does_not_import_encoders():
    """Core must stay modality-agnostic: importing the package must NOT pull in
    the encoder adapters (dependency flows adapter -> core only)."""
    for m in [k for k in list(sys.modules) if "pattern_brain" in k]:
        del sys.modules[m]
    import pattern_brain  # noqa: F401
    check("pattern_brain.adapters.encoders" not in sys.modules,
          "core import must NOT load adapters.encoders (Rule 23 separation)")
    print("  Rule 23: core does not import the encoder adapters")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 2: multimodal encoder adapters: tests")
    print("=" * 70)
    test_each_encoder_returns_valid_TD()
    test_text_encoder_deterministic_and_distinct()
    test_image_and_audio_encoders()
    test_multimodal_fusion_feeds_the_bank()
    test_single_modality_passthrough()
    test_rule23_core_does_not_import_encoders()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
