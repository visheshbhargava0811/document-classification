"""End-to-end document pipeline: dewarp -> denoise -> classify."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import classify as _classify
from . import denoise as _denoise
from . import dewarp as _dewarp

# Where optional .pt weight files are looked up (project root by default).
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2]


@dataclass
class ScanResult:
    original: np.ndarray            # BGR
    flattened: np.ndarray           # BGR (perspective-corrected)
    cleaned: np.ndarray             # grayscale scan
    corners: np.ndarray             # 4x2, TL TR BR BL
    dewarp_method: str
    denoise_method: str
    classification: dict = field(default_factory=dict)

    @property
    def doc_type(self) -> str:
        return self.classification.get("label", "unknown")

    @property
    def category(self) -> str:
        return self.classification.get("category", "")

    @property
    def confidence(self) -> float:
        return float(self.classification.get("confidence", 0.0))

    @property
    def category_confidence(self) -> float:
        return float(self.classification.get("category_confidence", 0.0))

    @property
    def uncertain(self) -> bool:
        return bool(self.classification.get("uncertain", False))

    @property
    def display_type(self) -> str:
        """Human-facing type: category-only when the specific type is uncertain."""
        if not self.classification:
            return "unknown"
        if self.uncertain and self.category:
            return f"{self.category} (type uncertain)"
        return self.doc_type


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ImageDecodeError(ValueError):
    """Raised when input bytes/file cannot be decoded as an image."""


# Guard against absurdly large uploads (decompression bombs / accidental huge files).
MAX_IMAGE_PIXELS = 60_000_000  # ~60 MP


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(
            f"Could not decode '{path.name}' as an image. "
            "Supported: JPG, PNG, BMP, WEBP. (For PDFs, render pages to images first.)"
        )
    _check_size(img)
    return img


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR array, with friendly errors. Used by the UI."""
    if not data:
        raise ImageDecodeError("The uploaded file is empty.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(
            "Could not decode this file as an image (JPG, PNG, BMP, or WEBP)."
        )
    _check_size(img)
    return img


def _check_size(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    if h * w > MAX_IMAGE_PIXELS:
        raise ImageDecodeError(
            f"Image is too large ({w}x{h}, {h * w / 1e6:.0f} MP). "
            f"Please downscale to under {MAX_IMAGE_PIXELS // 1_000_000} MP."
        )


def process_image(
    image_bgr: np.ndarray,
    weights_dir: Path | None = None,
    classify_type: bool = True,
    device: str | None = None,
    binarize: bool = False,
) -> ScanResult:
    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    device = device or _pick_device()

    dw = _dewarp.dewarp(image_bgr, base_dir=weights_dir, device=device)
    dn = _denoise.denoise(dw["flattened"], base_dir=weights_dir, device=device, binarize=binarize)

    classification: dict = {}
    if classify_type:
        # Classify on the color, perspective-corrected page (CLIP likes photos).
        flat_rgb = cv2.cvtColor(dw["flattened"], cv2.COLOR_BGR2RGB)
        classification = _classify.classify(flat_rgb)

    return ScanResult(
        original=image_bgr,
        flattened=dw["flattened"],
        cleaned=dn["cleaned"],
        corners=dw["corners"],
        dewarp_method=dw["method"],
        denoise_method=dn["method"],
        classification=classification,
    )


def process_path(path: str | Path, **kwargs) -> ScanResult:
    return process_image(load_image(path), **kwargs)
