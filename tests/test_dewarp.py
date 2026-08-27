"""Dewarp geometry: corner ordering, perspective warp, and detection fallback."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from document_classification.dewarp import (
    dewarp,
    find_corners_opencv,
    find_corners_segment,
    order_corners,
    warp_to_corners,
)


def test_order_corners_from_scrambled_input():
    # A square given out of order should come back as TL, TR, BR, BL.
    tl, tr, br, bl = [0, 0], [10, 0], [10, 10], [0, 10]
    scrambled = np.array([br, tl, bl, tr], dtype=np.float32)
    ordered = order_corners(scrambled)
    np.testing.assert_allclose(ordered, np.array([tl, tr, br, bl], dtype=np.float32))


def test_warp_to_corners_produces_rectangle_size():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    corners = np.array([[10, 10], [190, 10], [190, 90], [10, 90]], dtype=np.float32)
    out = warp_to_corners(img, corners)
    # width ~180, height ~80
    assert out.shape[0] == 80 and out.shape[1] == 180
    assert out.dtype == np.uint8


def _synthetic_document(canvas=(600, 800), inset=60):
    """A bright rectangular 'page' on a dark background."""
    h, w = canvas
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    cv2.rectangle(img, (inset, inset), (w - inset, h - inset), (240, 240, 240), -1)
    return img


def test_find_corners_opencv_detects_page():
    img = _synthetic_document()
    corners = find_corners_opencv(img)
    assert corners is not None
    assert corners.shape == (4, 2)
    # Detected quad should be close to the drawn rectangle corners.
    ordered = order_corners(corners)
    assert ordered[:, 0].min() < 120 and ordered[:, 0].max() > 680


def test_find_corners_segment_on_coloured_background():
    # A cream page on a saturated blue background: edge/contour detection is weak,
    # but colour segmentation should isolate the page.
    h, w = 900, 1200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (200, 120, 40)  # BGR: strong blue-ish background
    cv2.rectangle(img, (250, 200), (950, 700), (210, 225, 235), -1)  # cream page
    corners = find_corners_segment(img)
    assert corners is not None
    assert corners.shape == (4, 2)
    ordered = order_corners(corners)
    # Recovered quad should roughly bound the drawn page, not the whole frame.
    area = cv2.contourArea(ordered.astype(np.float32))
    assert 0.2 * h * w < area < 0.85 * h * w


def test_dewarp_returns_valid_result():
    img = _synthetic_document()
    res = dewarp(img, base_dir=Path("."))
    assert res["method"] in {"cnn", "opencv", "full-frame"}
    assert res["corners"].shape == (4, 2)
    assert res["flattened"].ndim == 3 and res["flattened"].size > 0
