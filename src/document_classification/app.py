"""Streamlit web app: scan document photos/PDFs and classify their type.

Features: single or batch upload, multi-page PDFs, camera capture, a clean
grayscale or black-and-white scan, hierarchical type classification, and
downloads (cleaned PNG, searchable PDF, extracted text, JSON, batch ZIP).

Run with:
    streamlit run src/document_classification/app.py
or:
    document-classification-web   (see README)
"""
from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np
import streamlit as st

from document_classification.exports import OcrUnavailable, extract_text, searchable_pdf
from document_classification.pipeline import (
    ImageDecodeError,
    decode_image_bytes,
    pdf_to_images,
    process_image,
)

st.set_page_config(page_title="Document Scanner + Classifier", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Processing (cached per page image + options)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _process_page(png_bytes: bytes, binarize: bool, want_exports: bool) -> dict:
    bgr = decode_image_bytes(png_bytes)
    result = process_image(bgr, binarize=binarize)
    out = {
        "flattened": cv2.cvtColor(result.flattened, cv2.COLOR_BGR2RGB),
        "cleaned": result.cleaned,
        "corners": result.corners.tolist(),
        "dewarp_method": result.dewarp_method,
        "denoise_method": result.denoise_method,
        "classification": result.classification,
        "original": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "text": None,
        "pdf": None,
    }
    if want_exports:
        try:
            out["text"] = extract_text(result.cleaned)
            out["pdf"] = searchable_pdf(result.cleaned)
        except OcrUnavailable:
            pass
    return out


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _gather_pages(uploaded, camera) -> list[tuple[str, bytes]]:
    """Flatten uploads (+ camera) into (label, png/image-bytes) pages, expanding PDFs."""
    pages: list[tuple[str, bytes]] = []
    files = list(uploaded or [])
    if camera is not None:
        files.append(camera)
    for f in files:
        name = getattr(f, "name", "camera.png")
        data = f.getvalue()
        if name.lower().endswith(".pdf"):
            imgs = pdf_to_images(data)
            for i, img in enumerate(imgs, 1):
                pages.append((f"{name} · page {i}", _png_bytes(img)))
        else:
            pages.append((name, data))
    return pages


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _draw_corners(img_rgb: np.ndarray, corners) -> np.ndarray:
    out = img_rgb.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0),
                  thickness=max(2, img_rgb.shape[1] // 300))
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), max(4, img_rgb.shape[1] // 150), (0, 200, 0), -1)
    return out


def _result_payload(label: str, res: dict) -> dict:
    cls = res["classification"]
    return {
        "input": label,
        "category": cls.get("category", ""),
        "category_confidence": round(cls.get("category_confidence", 0.0), 4),
        "document_type": cls.get("label", "unknown"),
        "confidence": round(cls.get("confidence", 0.0), 4),
        "uncertain": cls.get("uncertain", False),
        "classifier": cls.get("method", "n/a"),
        "top_k": cls.get("top_k", []),
        "dewarp_method": res["dewarp_method"],
        "denoise_method": res["denoise_method"],
    }


def _render_detail(label: str, res: dict) -> None:
    cls = res["classification"]
    category = cls.get("category", "")
    lbl = cls.get("label", "unknown")
    conf = cls.get("confidence", 0.0)
    cat_conf = cls.get("category_confidence", 0.0)
    uncertain = cls.get("uncertain", False)
    method = cls.get("method", "n/a")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", category or "—", help=f"{cat_conf*100:.1f}% confident")
    c2.metric("Document type", lbl if not uncertain else "—",
              help="Specific type is uncertain — trust the category." if uncertain else None)
    c3.metric("Confidence", f"{conf*100:.1f}%")
    c4.metric("Classifier", method)

    if uncertain:
        st.info(f"The category **{category}** is a confident call, but the specific "
                f"type is ambiguous here — treat the top guess below as a hint.")

    if cls.get("top_k"):
        st.write("**Top guesses**")
        for item in cls["top_k"]:
            cat = item.get("category", "")
            prefix = f"{cat} › " if cat else ""
            st.progress(min(1.0, float(item["score"])),
                        text=f"{prefix}{item['label']} — {item['score']*100:.1f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Detected page**")
        st.image(_draw_corners(res["original"], res["corners"]), use_container_width=True)
        st.caption(f"Corner detection: {res['dewarp_method']}")
    with col2:
        st.markdown("**2. Flattened**")
        st.image(res["flattened"], use_container_width=True)
    with col3:
        st.markdown("**3. Cleaned scan**")
        st.image(res["cleaned"], use_container_width=True, clamp=True)
        st.caption(f"Enhancement: {res['denoise_method']}")

    _render_downloads(label, res)


def _safe_stem(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label)
    return keep.strip("_") or "scan"


def _render_downloads(label: str, res: dict) -> None:
    stem = _safe_stem(label)
    scan_png = _png_bytes(res["cleaned"])
    cols = st.columns(4)
    cols[0].download_button("⬇️ Scan (PNG)", scan_png, file_name=f"{stem}.png",
                            mime="image/png", key=f"png_{stem}")
    cols[1].download_button("⬇️ Result (JSON)",
                            json.dumps(_result_payload(label, res), indent=2),
                            file_name=f"{stem}.json", mime="application/json",
                            key=f"json_{stem}")
    if res.get("pdf"):
        cols[2].download_button("⬇️ Searchable PDF", res["pdf"],
                                file_name=f"{stem}.pdf", mime="application/pdf",
                                key=f"pdf_{stem}")
    if res.get("text"):
        cols[3].download_button("⬇️ Text (TXT)", res["text"],
                                file_name=f"{stem}.txt", mime="text/plain",
                                key=f"txt_{stem}")


def _batch_zip(results: list[tuple[str, dict]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, res in results:
            zf.writestr(f"{_safe_stem(label)}.png", _png_bytes(res["cleaned"]))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("📄 Document Scanner + Classifier")
st.caption(
    "Upload photos or PDFs of documents. Each page is detected, perspective-"
    "corrected, cleaned into a scan, and classified into a category and type."
)

with st.sidebar:
    st.header("Options")
    binarize = st.toggle(
        "Pure black & white",
        value=False,
        help="Off: clean grayscale scan (recommended). On: hard black-and-white — "
             "smaller files, but can break logos/photos.",
    )
    ocr_ok = _ocr_available()
    want_exports = st.toggle(
        "OCR text + searchable PDF",
        value=ocr_ok,
        disabled=not ocr_ok,
        help="Extract selectable text and build a searchable PDF (needs Tesseract)."
             if ocr_ok else "Install Tesseract to enable OCR exports.",
    )
    if not ocr_ok:
        st.caption("⚠️ Tesseract not found — OCR exports disabled.")

uploaded = st.file_uploader(
    "Upload documents (images or PDF)",
    type=["jpg", "jpeg", "png", "bmp", "webp", "pdf"],
    accept_multiple_files=True,
)
with st.expander("📷 …or take a photo"):
    camera = st.camera_input("Capture a document")

if not uploaded and camera is None:
    st.info("Upload one or more receipts, forms, invoices, IDs, PDFs, etc. to begin.")
    st.stop()

try:
    pages = _gather_pages(uploaded, camera)
except ImageDecodeError as e:
    st.error(str(e))
    st.stop()

results: list[tuple[str, dict]] = []
errors: list[tuple[str, str]] = []
progress = st.progress(0.0, text="Processing…")
for i, (label, data) in enumerate(pages, 1):
    try:
        results.append((label, _process_page(data, binarize, want_exports)))
    except ImageDecodeError as e:
        errors.append((label, str(e)))
    except Exception as e:  # pragma: no cover - defensive UI guard
        errors.append((label, f"Unexpected error: {e}"))
    progress.progress(i / len(pages), text=f"Processing… ({i}/{len(pages)})")
progress.empty()

for label, msg in errors:
    st.error(f"**{label}**: {msg}")

if not results:
    st.stop()

# Single document -> detailed view. Multiple -> summary table + per-doc expanders.
if len(results) == 1:
    st.subheader("Result")
    _render_detail(*results[0])
else:
    st.subheader(f"Results — {len(results)} pages")
    st.dataframe(
        [
            {
                "Document": label,
                "Category": r["classification"].get("category", ""),
                "Type": (r["classification"].get("label", "")
                         if not r["classification"].get("uncertain") else "(uncertain)"),
                "Confidence": f"{r['classification'].get('confidence', 0.0)*100:.0f}%",
            }
            for label, r in results
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button("⬇️ Download all cleaned scans (ZIP)", _batch_zip(results),
                       file_name="scans.zip", mime="application/zip")
    for label, r in results:
        with st.expander(label):
            _render_detail(label, r)
