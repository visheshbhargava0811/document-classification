"""Export helpers: extract OCR text and build a searchable PDF from a scan.

These use Tesseract (the optional ``ocr`` extra). Each raises ``OcrUnavailable``
with a friendly message when the ``pytesseract`` package or the ``tesseract``
binary is missing, so callers can degrade gracefully.
"""
from __future__ import annotations

import numpy as np


class OcrUnavailable(RuntimeError):
    """Raised when OCR (pytesseract + the tesseract binary) is not available."""


_OCR_HINT = (
    "OCR needs Tesseract. Install the extra with `uv sync --extra ocr` and the "
    "binary (`brew install tesseract` on macOS, `apt-get install tesseract-ocr` "
    "on Debian/Ubuntu)."
)


def _pil(image: np.ndarray):
    from PIL import Image

    return Image.fromarray(image)


def _require_pytesseract():
    try:
        import pytesseract  # noqa: F401

        return pytesseract
    except Exception as e:  # pragma: no cover - import guard
        raise OcrUnavailable(_OCR_HINT) from e


def extract_text(image: np.ndarray) -> str:
    """Return OCR-extracted plain text from a (grayscale or RGB) image."""
    pytesseract = _require_pytesseract()
    try:
        return pytesseract.image_to_string(_pil(image))
    except Exception as e:
        raise OcrUnavailable(_OCR_HINT) from e


def searchable_pdf(image: np.ndarray) -> bytes:
    """Return PDF bytes: the image with an invisible, selectable OCR text layer."""
    pytesseract = _require_pytesseract()
    try:
        return pytesseract.image_to_pdf_or_hocr(_pil(image), extension="pdf")
    except Exception as e:
        raise OcrUnavailable(_OCR_HINT) from e
