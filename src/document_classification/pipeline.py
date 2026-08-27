"""End-to-end document pipeline: dewarp -> denoise -> classify."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def process_image(
    image_bgr: np.ndarray,
    weights_dir: Optional[Path] = None,
    classify_type: bool = True,
    device: Optional[str] = None,
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
