"""Denoise / enhancement: output contract and binarize behaviour."""
from __future__ import annotations

import numpy as np

from document_classification.denoise import denoise_classic


def _photo(h=200, w=150):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_grayscale_output_contract():
    img = _photo()
    out = denoise_classic(img, binarize=False)
    assert out.dtype == np.uint8
    assert out.ndim == 2  # grayscale
    assert out.shape == img.shape[:2]


def test_binarize_is_near_black_and_white():
    img = _photo()
    out = denoise_classic(img, binarize=True)
    assert out.shape == img.shape[:2]
    # Adaptive threshold yields (almost) only 0 and 255.
    extremes = np.isin(out, [0, 255]).mean()
    assert extremes > 0.95
