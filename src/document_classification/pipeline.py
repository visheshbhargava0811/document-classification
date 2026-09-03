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
    raw_text: str = ""               # case-preserving OCR text (for extraction/summary)
    ocr_json: dict = field(default_factory=dict)       # structured OCR: engine + blocks
    ocr_json_en: dict = field(default_factory=dict)    # same blocks, translated to English
    understanding: dict = field(default_factory=dict)  # summary + key_points + fields
    invoice: dict = field(default_factory=dict)        # structured invoice fields (for Excel)

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
    ocr_engine: str | None = None,
    understand: bool = False,
    translate: bool = False,
    invoice: bool = False,
) -> ScanResult:
    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    device = device or _pick_device()

    dw = _dewarp.dewarp(image_bgr, base_dir=weights_dir, device=device)
    dn = _denoise.denoise(dw["flattened"], base_dir=weights_dir, device=device, binarize=binarize)

    flat_rgb = cv2.cvtColor(dw["flattened"], cv2.COLOR_BGR2RGB)

    classification: dict = {}
    if classify_type:
        # Classify on the color, perspective-corrected page (CLIP likes photos).
        classification = _classify.classify(flat_rgb, ocr_engine=ocr_engine)

    raw_text = ""
    ocr_json: dict = {}
    ocr_json_en: dict = {}
    understanding: dict = {}
    invoice_data: dict = {}
    if understand or translate or invoice:
        # One OCR pass yields both the case-preserving text (for summarisation) and
        # the structured per-region JSON (engine + blocks with bboxes/confidence).
        ocr = _classify.ocr_extract(flat_rgb, engine=ocr_engine)
        raw_text = ocr.get("text", "")
        ocr_json = {"engine": ocr.get("engine", ""), "blocks": ocr.get("blocks", [])}
        if translate and ocr_json["blocks"]:
            from . import translate as _translate

            ocr_json_en = _translate.translate_ocr_json(ocr_json) or {}
        if understand and raw_text.strip():
            from . import understand as _understand

            understanding = _understand.understand(
                raw_text, doc_type=classification.get("label")
            )
        if invoice and ocr_json.get("blocks"):
            from . import invoice as _invoice

            invoice_data = _invoice.extract_invoice(
                ocr_json, doc_type=classification.get("label")
            ) or {}

    return ScanResult(
        original=image_bgr,
        flattened=dw["flattened"],
        cleaned=dn["cleaned"],
        corners=dw["corners"],
        dewarp_method=dw["method"],
        denoise_method=dn["method"],
        classification=classification,
        raw_text=raw_text,
        ocr_json=ocr_json,
        ocr_json_en=ocr_json_en,
        understanding=understanding,
        invoice=invoice_data,
    )


def process_path(path: str | Path, **kwargs) -> ScanResult:
    return process_image(load_image(path), **kwargs)


@dataclass
class DocumentResult:
    """One multi-page document: per-page scans + a single document-level type.

    A PDF is one document, so ``classification`` is decided over all pages at
    once (see :func:`document_classification.classify.classify_document`), while
    ``pages`` keeps every page's cleaned/flattened scan for export and a per-page
    breakdown lives in ``classification["pages"]``.
    """
    pages: list[ScanResult]
    classification: dict = field(default_factory=dict)
    raw_text: str = ""
    ocr_json: dict = field(default_factory=dict)       # {engine, pages:[blocks...]}
    ocr_json_en: dict = field(default_factory=dict)    # translated {engine, pages:[blocks...]}
    understanding: dict = field(default_factory=dict)
    invoice: dict = field(default_factory=dict)        # structured invoice fields (for Excel)

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
        if not self.classification:
            return "unknown"
        if self.uncertain and self.category:
            return f"{self.category} (type uncertain)"
        return self.doc_type


# Cap combined OCR text fed to the summariser so a huge PDF stays affordable.
UNDERSTAND_TEXT_CAP = 20_000


def process_document(
    images: list[np.ndarray],
    weights_dir: Path | None = None,
    classify_type: bool = True,
    device: str | None = None,
    binarize: bool = False,
    ocr_engine: str | None = None,
    understand: bool = False,
    translate: bool = False,
    invoice: bool = False,
) -> DocumentResult:
    """Process a multi-page document (list of BGR page images) into ONE result.

    Each page is dewarped + denoised individually (scans are inherently
    per-page), but the document **type is decided once** over all pages. When
    ``understand`` is set, the pages' OCR text is concatenated and summarised as
    a single document.
    """
    if not images:
        raise ImageDecodeError("The document has no pages to process.")

    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    device = device or _pick_device()

    pages: list[ScanResult] = []
    flat_rgb_pages: list[np.ndarray] = []
    for img in images:
        # Dewarp + denoise only; the type is classified at the document level.
        page = process_image(
            img, weights_dir=weights_dir, classify_type=False, device=device,
            binarize=binarize, ocr_engine=ocr_engine, understand=False,
        )
        pages.append(page)
        flat_rgb_pages.append(cv2.cvtColor(page.flattened, cv2.COLOR_BGR2RGB))

    classification: dict = {}
    if classify_type:
        classification = _classify.classify_document(flat_rgb_pages, ocr_engine=ocr_engine)

    raw_text = ""
    ocr_json: dict = {}
    ocr_json_en: dict = {}
    understanding: dict = {}
    invoice_data: dict = {}
    if understand or translate or invoice:
        texts: list[str] = []
        page_blocks: list[dict] = []
        page_blocks_en: list[dict] = []
        engine_used = ""
        any_translated = False
        for flat_rgb in flat_rgb_pages:
            ocr = _classify.ocr_extract(flat_rgb, engine=ocr_engine)
            engine_used = engine_used or ocr.get("engine", "")
            texts.append(ocr.get("text", ""))
            page = {"engine": ocr.get("engine", ""), "blocks": ocr.get("blocks", [])}
            page_blocks.append(page)
            if translate and page["blocks"]:
                from . import translate as _translate

                page_en = _translate.translate_ocr_json(page)
                if page_en:
                    page_blocks_en.append({"engine": page_en.get("engine", ""),
                                           "blocks": page_en.get("blocks", [])})
                    any_translated = True
                else:  # untranslated page — keep it aligned with the original
                    page_blocks_en.append(page)
            elif translate:
                page_blocks_en.append(page)
        raw_text = "\n\n".join(t for t in texts if t.strip())
        ocr_json = {"engine": engine_used, "pages": page_blocks}
        if any_translated:
            ocr_json_en = {"engine": f"{engine_used}+gemini-translate",
                           "target_language": "en", "pages": page_blocks_en}
        if understand and raw_text.strip():
            from . import understand as _understand

            understanding = _understand.understand(
                raw_text[:UNDERSTAND_TEXT_CAP],
                doc_type=classification.get("label"),
            )
        if invoice and ocr_json.get("pages"):
            from . import invoice as _invoice

            invoice_data = _invoice.extract_invoice(
                ocr_json, doc_type=classification.get("label")
            ) or {}

    return DocumentResult(
        pages=pages,
        classification=classification,
        raw_text=raw_text,
        ocr_json=ocr_json,
        ocr_json_en=ocr_json_en,
        understanding=understanding,
        invoice=invoice_data,
    )


def pdf_to_images(data: bytes, dpi: int = 200, max_pages: int = 500) -> list[np.ndarray]:
    """Render a PDF (given as bytes) to a list of BGR page images via PyMuPDF.

    PyMuPDF ships as a self-contained wheel (no system dependency), so this works
    on Streamlit Community Cloud without extra packages.
    """
    import pymupdf  # (a.k.a. fitz)

    images: list[np.ndarray] = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ImageDecodeError("Could not open this file as a PDF.") from e
    with doc:
        for page in doc:
            if len(images) >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:  # grayscale
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            _check_size(arr)
            images.append(arr)
    if not images:
        raise ImageDecodeError("The PDF has no rendered pages.")
    return images
