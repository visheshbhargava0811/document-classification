"""Stage 2: clean / enhance the flattened document.

Uses the trained U-Net when ``model.pt`` is available, otherwise a classic
OpenCV enhancement (denoise + adaptive threshold) that turns the page into a
crisp scan-like image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DENOISE_WEIGHTS_CANDIDATES = ("model.pt", "weights/model.pt")


def resolve_denoise_weights(base_dir: Path) -> Optional[str]:
    for cand in DENOISE_WEIGHTS_CANDIDATES:
        p = base_dir / cand
        if p.exists():
            return str(p)
    return None


def denoise_classic(flattened_bgr: np.ndarray, binarize: bool = False) -> np.ndarray:
    """Scanner-style enhancement.

    Removes uneven lighting, shadows, and colour tint by estimating the page
    background and dividing it out, then gently denoises and boosts contrast.
    Returns a clean grayscale "scan" by default (white paper, smooth text, solid
    logos). Set ``binarize=True`` for a pure black-and-white document instead.
    """
    gray = cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2GRAY)

    # Kernel scaled to the image so it works on both small and large scans.
    k = max(11, (min(gray.shape[:2]) // 20) | 1)  # odd, >= 11

    # Estimate the page background: dilation keeps local bright (paper) values and
    # erases dark text; a median blur then smooths it into an illumination map.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    bg = cv2.medianBlur(bg, k)

    # Divide out the background -> flat white page, tint and shadows removed,
    # while text and logos keep their true darkness.
    norm = cv2.divide(gray, bg, scale=255)

    # Gentle, edge-preserving denoise (much lighter than before).
    norm = cv2.fastNlMeansDenoising(norm, None, h=7, templateWindowSize=7, searchWindowSize=21)

    # Local contrast boost so faint print stays legible without crushing solids.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cleaned = clahe.apply(norm)

    if not binarize:
        # Lift the paper to white: map the 80th-percentile brightness (the page
        # itself) up to pure white so residual shadow haze reads as clean paper,
        # while keeping darker ink and logos intact.
        wp = float(np.percentile(cleaned, 80))
        cleaned = np.clip(cleaned.astype(np.float32) * (255.0 / max(wp, 1.0)), 0, 255).astype(np.uint8)

    if binarize:
        cleaned = cv2.adaptiveThreshold(
            cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=k, C=15,
        )
    return cleaned


def denoise_unet(flattened_bgr: np.ndarray, weights_path: str, device: str = "cpu") -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import transforms

    from .models import Unet

    model = Unet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    gray = cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    orig_size = pil.size  # (w, h)
    tfm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    tensor = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze(0).cpu()
    out_np = (out.squeeze().numpy() * 255).astype(np.uint8)
    return np.array(Image.fromarray(out_np).resize(orig_size))


def denoise(flattened_bgr: np.ndarray, base_dir: Path, device: str = "cpu",
            binarize: bool = False) -> dict:
    weights = resolve_denoise_weights(base_dir)
    if weights:
        try:
            cleaned = denoise_unet(flattened_bgr, weights, device=device)
            return {"cleaned": cleaned, "method": "unet"}
        except Exception:
            pass
    return {"cleaned": denoise_classic(flattened_bgr, binarize=binarize), "method": "classic"}
