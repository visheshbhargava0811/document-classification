"""End-to-end pipeline (without the CLIP classifier, so it's fast and offline)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from document_classification.pipeline import (
    ImageDecodeError,
    ScanResult,
    decode_image_bytes,
    pdf_to_images,
    process_image,
)


def _synthetic_document(canvas=(600, 800), inset=60):
    h, w = canvas
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    cv2.rectangle(img, (inset, inset), (w - inset, h - inset), (240, 240, 240), -1)
    return img


def test_process_image_without_classification():
    img = _synthetic_document()
    res = process_image(img, classify_type=False)
    assert isinstance(res, ScanResult)
    assert res.corners.shape == (4, 2)
    assert res.cleaned.ndim == 2  # grayscale scan
    assert res.flattened.ndim == 3
    assert res.doc_type == "unknown"  # no classification requested
    assert res.dewarp_method in {"cnn", "opencv", "full-frame"}


def test_decode_image_bytes_roundtrip():
    img = _synthetic_document(canvas=(64, 64), inset=10)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    decoded = decode_image_bytes(buf.tobytes())
    assert decoded.shape == img.shape


def test_decode_image_bytes_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"not an image")


def test_decode_image_bytes_rejects_empty():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"")


def test_pdf_to_images_renders_pages():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    pages = pdf_to_images(data, dpi=72)
    assert len(pages) == 2
    assert all(p.ndim == 3 and p.dtype == np.uint8 for p in pages)


def test_pdf_to_images_rejects_non_pdf():
    with pytest.raises(ImageDecodeError):
        pdf_to_images(b"not a pdf")
