"""Stage 1: locate the document and flatten it to a top-down view.

Two strategies:

* ``DocumentCornerModel`` CNN, if the trained weights file is present.
* Classic OpenCV contour detection as a robust, always-available fallback.

Both produce four ordered corner points (TL, TR, BR, BL) which are fed to a
perspective transform to un-skew the page.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CORNER_WEIGHTS_CANDIDATES = (
    "dl_project_doc_extract_weights.pt",
    "weights/dl_project_doc_extract_weights.pt",
)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL regardless of input order."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _dest_size(corners: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    return max(width, 1), max(height, 1)


def warp_to_corners(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply a perspective transform so the 4 corners become a flat rectangle."""
    corners = order_corners(corners)
    width, height = _dest_size(corners)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image_bgr, M, (width, height))


# --------------------------------------------------------------------------- #
# Classic OpenCV corner detection
# --------------------------------------------------------------------------- #
def find_corners_opencv(image_bgr: np.ndarray) -> np.ndarray | None:
    """Find the document quadrilateral using edges + contours. May return None."""
    h, w = image_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1 else image_bgr.copy()
    resize_factor = small.shape[1] / w  # small_w / orig_w

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    img_area = small.shape[0] * small.shape[1]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img_area:
            return approx.reshape(4, 2).astype(np.float32) / resize_factor
    return None


# --------------------------------------------------------------------------- #
# CNN corner detection (optional weights)
# --------------------------------------------------------------------------- #
def find_corners_cnn(image_bgr: np.ndarray, weights_path: str, device: str = "cpu") -> np.ndarray | None:
    import torch
    from torchvision import transforms

    from .models import DocumentCornerModel

    model = DocumentCornerModel().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    tfm = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    tensor = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).cpu().numpy().flatten()

    # Map [x_bl,x_br,x_tl,x_tr,y_bl,y_br,y_tl,y_tr] -> TL,TR,BR,BL (notebook order)
    corners = np.array(
        [
            [out[2], out[6]],  # TL
            [out[3], out[7]],  # TR
            [out[1], out[5]],  # BR
            [out[0], out[4]],  # BL
        ],
        dtype=np.float32,
    )
    # The model was trained on SmartDoc frames (~1920x1080). Rescale predictions
    # from that reference frame onto the actual image dimensions.
    h, w = image_bgr.shape[:2]
    corners[:, 0] *= w / 1920.0
    corners[:, 1] *= h / 1080.0

    # Sanity check: corners should sit inside (a small margin around) the image.
    margin = 0.15
    if (
        corners[:, 0].min() < -margin * w
        or corners[:, 0].max() > (1 + margin) * w
        or corners[:, 1].min() < -margin * h
        or corners[:, 1].max() > (1 + margin) * h
    ):
        return None
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


def resolve_corner_weights(base_dir: Path) -> str | None:
    for cand in CORNER_WEIGHTS_CANDIDATES:
        p = base_dir / cand
        if p.exists():
            return str(p)
    return None


def dewarp(image_bgr: np.ndarray, base_dir: Path, device: str = "cpu") -> dict:
    """Return dict with flattened image, the corners used, and which method won."""
    weights = resolve_corner_weights(base_dir)
    corners: np.ndarray | None = None
    method = "none"

    if weights:
        try:
            corners = find_corners_cnn(image_bgr, weights, device=device)
            if corners is not None:
                method = "cnn"
        except Exception:
            corners = None

    if corners is None:
        corners = find_corners_opencv(image_bgr)
        if corners is not None:
            method = "opencv"

    if corners is None:
        # No document boundary found: use the full frame.
        h, w = image_bgr.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        method = "full-frame"

    flattened = warp_to_corners(image_bgr, corners)
    return {"flattened": flattened, "corners": order_corners(corners), "method": method}
