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
from contextlib import contextmanager

import cv2
import numpy as np
import streamlit as st

from document_classification.exports import OcrUnavailable, extract_text, searchable_pdf
from document_classification.invoice import invoice_workbook
from document_classification.pipeline import (
    ImageDecodeError,
    decode_image_bytes,
    pdf_to_images,
    process_document,
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


@st.cache_resource(show_spinner=False)
def _surya_available() -> bool:
    try:
        import surya  # noqa: F401

        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _gemini_available() -> bool:
    try:
        from document_classification.understand import gemini_available

        return gemini_available()
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def _process_page(png_bytes: bytes, binarize: bool, want_exports: bool,
                  ocr_engine: str, want_understanding: bool,
                  want_translate: bool = False, want_invoice: bool = False) -> dict:
    bgr = decode_image_bytes(png_bytes)
    result = process_image(bgr, binarize=binarize, ocr_engine=ocr_engine,
                           understand=want_understanding, translate=want_translate,
                           invoice=want_invoice)
    out = {
        "flattened": cv2.cvtColor(result.flattened, cv2.COLOR_BGR2RGB),
        "cleaned": result.cleaned,
        "corners": result.corners.tolist(),
        "dewarp_method": result.dewarp_method,
        "denoise_method": result.denoise_method,
        "classification": result.classification,
        "original": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "raw_text": result.raw_text,
        "ocr_json": result.ocr_json,
        "ocr_json_en": result.ocr_json_en,
        "understanding": result.understanding,
        "invoice": result.invoice,
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


@st.cache_data(show_spinner=False)
def _process_document(kind: str, pages_bytes: tuple[bytes, ...], binarize: bool,
                      want_exports: bool, ocr_engine: str,
                      want_understanding: bool, want_translate: bool = False,
                      want_invoice: bool = False) -> dict:
    """Process one document. Images defer to :func:`_process_page`; a multi-page
    PDF is dewarped/denoised per page but classified as ONE document.

    Returns the same ``res`` shape as :func:`_process_page` (so the existing
    detail renderer works), using page 1 as the representative stage view and
    adding ``doc_pages`` thumbnails plus the per-page breakdown in
    ``classification["pages"]``.
    """
    if kind != "pdf" or len(pages_bytes) == 1:
        return _process_page(pages_bytes[0], binarize, want_exports, ocr_engine,
                             want_understanding, want_translate, want_invoice)

    images = [decode_image_bytes(b) for b in pages_bytes]
    doc = process_document(images, binarize=binarize, ocr_engine=ocr_engine,
                           understand=want_understanding, translate=want_translate,
                           invoice=want_invoice)
    first = doc.pages[0]
    # Flatten the per-page OCR JSON into one {engine, blocks} for the UI/exports.
    all_blocks: list[dict] = []
    engine = doc.ocr_json.get("engine", "")
    for pj in doc.ocr_json.get("pages", []):
        all_blocks.extend(pj.get("blocks", []))
    all_blocks_en: list[dict] = []
    engine_en = doc.ocr_json_en.get("engine", "")
    for pj in doc.ocr_json_en.get("pages", []):
        all_blocks_en.extend(pj.get("blocks", []))
    out = {
        "flattened": cv2.cvtColor(first.flattened, cv2.COLOR_BGR2RGB),
        "cleaned": first.cleaned,
        "corners": first.corners.tolist(),
        "dewarp_method": first.dewarp_method,
        "denoise_method": first.denoise_method,
        "classification": doc.classification,
        "original": cv2.cvtColor(first.original, cv2.COLOR_BGR2RGB),
        "raw_text": doc.raw_text,
        "ocr_json": {"engine": engine, "blocks": all_blocks} if all_blocks else {},
        "ocr_json_en": ({"engine": engine_en, "target_language": "en",
                         "blocks": all_blocks_en} if all_blocks_en else {}),
        "understanding": doc.understanding,
        "invoice": doc.invoice,
        "text": None,
        "pdf": None,
        "n_pages": len(doc.pages),
        "doc_pages": [p.cleaned for p in doc.pages],
    }
    if want_exports:
        try:
            out["text"] = "\n\n".join(extract_text(p.cleaned) for p in doc.pages)
        except OcrUnavailable:
            pass
    return out


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _gather_documents(uploaded, camera) -> list[tuple[str, str, tuple[bytes, ...]]]:
    """Group uploads (+ camera) into documents: (label, kind, page-bytes tuple).

    A PDF is one document made of its page images (classified as a whole); each
    image/camera shot is its own single-page document. ``kind`` is "pdf" or
    "image".
    """
    documents: list[tuple[str, str, tuple[bytes, ...]]] = []
    files = list(uploaded or [])
    if camera is not None:
        files.append(camera)
    for f in files:
        name = getattr(f, "name", "camera.png")
        data = f.getvalue()
        if name.lower().endswith(".pdf"):
            imgs = pdf_to_images(data)
            documents.append((name, "pdf", tuple(_png_bytes(img) for img in imgs)))
        else:
            documents.append((name, "image", (data,)))
    return documents


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@contextmanager
def _collapsible(title: str, nested: bool):
    """A collapsible section that degrades to a bordered container when nested.

    Streamlit forbids an ``st.expander`` inside another expander. In the batch
    view each document's detail is rendered inside a per-document expander, so
    inner sections (per-page breakdown, OCR regions) must not open their own
    expander there — they render as an always-open bordered block instead.
    """
    if nested:
        box = st.container(border=True)
        box.markdown(f"**{title}**")
        with box:
            yield
    else:
        with st.expander(title):
            yield


def _draw_corners(img_rgb: np.ndarray, corners) -> np.ndarray:
    out = img_rgb.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0),
                  thickness=max(2, img_rgb.shape[1] // 300))
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), max(4, img_rgb.shape[1] // 150), (0, 200, 0), -1)
    return out


def _truncate(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _result_payload(label: str, res: dict) -> dict:
    cls = res["classification"]
    payload = {
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
    understanding = res.get("understanding")
    if understanding:
        payload["understanding"] = {
            "engine": understanding.get("engine", ""),
            "summary": understanding.get("summary", ""),
            "key_points": understanding.get("key_points", []),
            "fields": understanding.get("fields", {}),
        }
    ocr_json_en = res.get("ocr_json_en") or {}
    if ocr_json_en.get("blocks"):
        payload["ocr_json_en"] = ocr_json_en
    return payload


def _render_detail(label: str, res: dict, nested: bool = False) -> None:
    cls = res["classification"]
    category = cls.get("category", "")
    lbl = cls.get("label", "unknown")
    conf = cls.get("confidence", 0.0)
    cat_conf = cls.get("category_confidence", 0.0)
    uncertain = cls.get("uncertain", False)
    method = cls.get("method", "n/a")

    n_pages = res.get("n_pages")
    if n_pages:
        st.caption(f"📄 {n_pages}-page document — classified as a whole "
                   f"(page 1 shown below as representative).")

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

    _render_page_breakdown(res, nested)
    _render_understanding(res, nested)
    _render_invoice(res)
    _render_downloads(label, res)


def _render_page_breakdown(res: dict, nested: bool = False) -> None:
    """For a multi-page document, show each page's own guess + scan thumbnail."""
    pages = res.get("classification", {}).get("pages") or []
    thumbs = res.get("doc_pages") or []
    if len(pages) <= 1 and len(thumbs) <= 1:
        return
    with _collapsible(f"📄 Per-page breakdown ({res.get('n_pages', len(thumbs))} pages)", nested):
        st.caption("Each page's independent guess — the headline type above is the "
                   "whole-document decision (averaged across all pages).")
        if pages:
            st.dataframe(
                [
                    {
                        "Page": p["page"],
                        "Category": p.get("category", ""),
                        "Type": p["label"] if not p.get("uncertain") else "(uncertain)",
                        "Confidence": f"{p.get('confidence', 0.0)*100:.0f}%",
                    }
                    for p in pages
                ],
                use_container_width=True, hide_index=True,
            )
        if thumbs:
            cols = st.columns(min(4, len(thumbs)))
            for i, thumb in enumerate(thumbs):
                with cols[i % len(cols)]:
                    st.image(thumb, caption=f"Page {i + 1}", use_container_width=True,
                             clamp=True)


def _fmt_field(key: str, value) -> str:
    label = key.replace("_", " ").title()
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return f"**{label}:** {value}"


def _render_invoice(res: dict) -> None:
    """Show the extracted invoice header fields + line-items table, when computed."""
    inv = res.get("invoice") or {}
    if not inv:
        return
    st.markdown("### 🧾 Invoice data")
    currency = inv.get("currency") or ""
    total = inv.get("total_amount") or ""
    if total and currency:
        total = f"{currency} {total}".strip()
    header = [
        ("Invoice No", inv.get("invoice_no")),
        ("Date", inv.get("date")),
        ("Vendor", inv.get("vendor_name")),
        ("Customer", inv.get("customer_name")),
        ("GSTIN", inv.get("gstin")),
        ("Total", total),
    ]
    cols = st.columns(2)
    for i, (k, v) in enumerate(h for h in header if h[1]):
        cols[i % 2].markdown(f"**{k}:** {v}")

    # Cheap validations: do the lines add up, and is the GSTIN checksum valid?
    validation = inv.get("validation") or {}
    rec = validation.get("reconciliation") or {}
    if rec.get("status") == "ok":
        st.success(f"✅ Line amounts reconcile with the total ({rec.get('lines_sum')}).")
    elif rec.get("status") == "mismatch":
        st.warning(f"⚠️ Line amounts sum to {rec.get('lines_sum')} but the total says "
                   f"{rec.get('total')} (Δ {rec.get('diff')}). A line may be misread or missing.")
    g = validation.get("gstin") or {}
    if g.get("status") == "valid":
        st.success("✅ GSTIN checksum valid.")
    elif g.get("status") == "invalid":
        st.warning(f"⚠️ GSTIN checksum invalid ({g.get('reason')}) — likely an OCR error; verify it.")

    particulars = inv.get("particulars") or []
    if particulars:
        st.dataframe(
            [{"Item (raw)": p.get("item_raw", ""), "Item (English)": p.get("item", ""),
              "ERP Code": p.get("erp_code", ""), "Category": p.get("category", ""),
              "Qty": p.get("qty", 1), "Amount": p.get("amount", ""),
              "Confidence": p.get("confidence", "")} for p in particulars],
            use_container_width=True, hide_index=True,
        )
    st.caption("Extracted by Gemini from Surya OCR regions, then resolved against the "
               "item knowledge graph (canonical name + ERP code + category); source kept "
               "as \"raw\". Use the **Download invoices (Excel)** button above (flat, one "
               "row per line item). Low-confidence rows need review — verify amounts.")


def _render_understanding(res: dict, nested: bool = False) -> None:
    """Show the summary + extracted structured fields, when computed."""
    u = res.get("understanding") or {}
    summary = (u.get("summary") or "").strip()
    fields = u.get("fields") or {}
    key_points = u.get("key_points") or []
    if not summary and not fields:
        return

    st.markdown("### 🧠 Summary & extracted data")
    if summary:
        st.write(summary)
    if key_points:
        for pt in key_points:
            st.markdown(f"- {pt}")
    if fields:
        st.markdown("**Extracted fields**")
        # Two-column key/value grid.
        items = list(fields.items())
        cols = st.columns(2)
        for i, (k, v) in enumerate(items):
            cols[i % 2].markdown(_fmt_field(k, v))
    engine = u.get("engine", "")
    if engine:
        pretty = "Gemini" if engine.startswith("gemini") else "offline heuristics"
        st.caption(f"Generated by {pretty}. Verify important values against the document.")

    ocr_json = res.get("ocr_json") or {}
    blocks = ocr_json.get("blocks") or []
    if blocks:
        with _collapsible(f"🔎 OCR regions — {ocr_json.get('engine', 'ocr')} "
                          f"({len(blocks)} blocks)", nested):
            st.caption("Structured OCR output: each detected region with its text, "
                       "bounding box, confidence and label. Copy the code below or "
                       "use the download button — both are exact.")
            # st.code (not st.json) so copied text matches the file byte-for-byte;
            # the interactive JSON tree annotates whole-number floats when copied.
            st.code(json.dumps(ocr_json, indent=2, ensure_ascii=False), language="json")

    ocr_json_en = res.get("ocr_json_en") or {}
    blocks_en = ocr_json_en.get("blocks") or []
    if blocks_en:
        with _collapsible(f"🌐 OCR regions, English — {ocr_json_en.get('engine', 'ocr')} "
                          f"({len(blocks_en)} blocks)", nested):
            st.caption("The same OCR regions with non-English text translated to "
                       "English inline. Each block keeps its bounding box and "
                       "confidence; the source string is preserved as "
                       "\"text_original\". Machine translation — verify important "
                       "values against the document.")
            st.code(json.dumps(ocr_json_en, indent=2, ensure_ascii=False), language="json")


def _safe_stem(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label)
    return keep.strip("_") or "scan"


def _render_downloads(label: str, res: dict) -> None:
    stem = _safe_stem(label)
    scan_png = _png_bytes(res["cleaned"])
    ocr_json = res.get("ocr_json") or {}
    has_ocr_json = bool(ocr_json.get("blocks"))
    ocr_json_en = res.get("ocr_json_en") or {}
    has_ocr_json_en = bool(ocr_json_en.get("blocks"))
    cols = st.columns(6)
    cols[0].download_button("⬇️ Scan (PNG)", scan_png, file_name=f"{stem}.png",
                            mime="image/png", key=f"png_{stem}")
    cols[1].download_button("⬇️ Result (JSON)",
                            json.dumps(_result_payload(label, res), indent=2),
                            file_name=f"{stem}.json", mime="application/json",
                            key=f"json_{stem}")
    if has_ocr_json:
        engine = ocr_json.get("engine", "ocr")
        cols[2].download_button(
            f"⬇️ OCR JSON ({engine})",
            json.dumps(ocr_json, indent=2, ensure_ascii=False),
            file_name=f"{stem}_ocr.json", mime="application/json",
            key=f"ocrjson_{stem}",
            help="Raw structured output of the OCR model: each detected region with "
                 "its text, bounding box, confidence and label.",
        )
    if has_ocr_json_en:
        cols[3].download_button(
            "⬇️ OCR JSON (English)",
            json.dumps(ocr_json_en, indent=2, ensure_ascii=False),
            file_name=f"{stem}_ocr_en.json", mime="application/json",
            key=f"ocrjsonen_{stem}",
            help="The OCR JSON with non-English text translated to English inline. "
                 "Same regions/bounding boxes; each block keeps the source string as "
                 "\"text_original\".",
        )
    if res.get("pdf"):
        cols[4].download_button("⬇️ Searchable PDF", res["pdf"],
                                file_name=f"{stem}.pdf", mime="application/pdf",
                                key=f"pdf_{stem}")
    if res.get("text"):
        cols[5].download_button("⬇️ Text (TXT)", res["text"],
                                file_name=f"{stem}.txt", mime="text/plain",
                                key=f"txt_{stem}")


def _batch_zip(results: list[tuple[str, dict]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, res in results:
            stem = _safe_stem(label)
            thumbs = res.get("doc_pages")
            if thumbs and len(thumbs) > 1:
                for i, page in enumerate(thumbs, 1):
                    zf.writestr(f"{stem}_p{i}.png", _png_bytes(page))
            else:
                zf.writestr(f"{stem}.png", _png_bytes(res["cleaned"]))
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

    gemini_on = _gemini_available()
    any_ocr = ocr_ok or _surya_available()
    want_understanding = st.toggle(
        "Summarize & extract data",
        value=True,
        disabled=not any_ocr,
        help="Read the document's text and produce a plain-language summary plus "
             "structured fields (receipt total, date, invoice number, …). Uses "
             "Google Gemini when a key is set, otherwise a fully offline fallback."
             if any_ocr else "Needs an OCR engine (Tesseract or Surya) to read the text first.",
    )
    if want_understanding:
        st.caption("✨ Gemini summaries active." if gemini_on
                   else "ℹ️ No GEMINI_API_KEY — using offline summary + extraction.")

    want_translate = st.toggle(
        "Translate OCR to English",
        value=False,
        disabled=not (any_ocr and gemini_on),
        help="Produce a second OCR JSON where non-English text (Hindi, Marwari, "
             "Kannada, …) is translated to English inline — same regions, same "
             "bounding boxes, English substituted for the source text. Requires "
             "a GEMINI_API_KEY."
             if any_ocr else "Needs an OCR engine to read the text first.",
    )
    if want_translate and not gemini_on:
        st.caption("ℹ️ Translation needs a GEMINI_API_KEY.")

    want_invoice = st.toggle(
        "Extract invoice → Excel",
        value=False,
        disabled=not (any_ocr and gemini_on),
        help="Read each invoice/bill (any language) and extract invoice no, date, "
             "party name & address, total, and a line-item table (item + quantity, "
             "with item names translated to English). Download all invoices as one "
             "Excel workbook — one sheet per invoice. Requires a GEMINI_API_KEY."
             if any_ocr else "Needs an OCR engine to read the text first.",
    )
    if want_invoice and not gemini_on:
        st.caption("ℹ️ Invoice extraction needs a GEMINI_API_KEY.")

    # Classification OCR engine (fuses keyword signals into CLIP). Surya is an
    # opt-in, higher-accuracy engine; offered only when installed.
    if _surya_available():
        ocr_engine = st.radio(
            "Classification OCR engine",
            options=["tesseract", "surya"],
            format_func=lambda e: "Tesseract (fast)" if e == "tesseract"
            else "Surya (accurate, heavier)",
            help="Engine used to read text for keyword fusion. Surya is more "
                 "accurate on skewed/low-quality photos but slower and heavier. "
                 "Searchable-PDF export always uses Tesseract.",
        )
    else:
        ocr_engine = "tesseract"

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
    documents = _gather_documents(uploaded, camera)
except ImageDecodeError as e:
    st.error(str(e))
    st.stop()

# Invoice extraction depends on high-quality OCR regions, so use Surya when it's
# installed regardless of the classification-engine radio (Tesseract mangles
# handwritten regional text and yields nothing to extract).
proc_engine = ocr_engine
if want_invoice and _surya_available() and ocr_engine != "surya":
    proc_engine = "surya"
    st.caption("🧾 Invoice mode: using Surya OCR for accurate region extraction.")

results: list[tuple[str, dict]] = []
errors: list[tuple[str, str]] = []
progress = st.progress(0.0, text="Processing…")
for i, (label, kind, pages_bytes) in enumerate(documents, 1):
    try:
        results.append((label, _process_document(kind, pages_bytes, binarize,
                                                  want_exports, proc_engine,
                                                  want_understanding, want_translate,
                                                  want_invoice)))
    except ImageDecodeError as e:
        errors.append((label, str(e)))
    except Exception as e:  # pragma: no cover - defensive UI guard
        errors.append((label, f"Unexpected error: {e}"))
    progress.progress(i / len(documents), text=f"Processing… ({i}/{len(documents)})")
progress.empty()

for label, msg in errors:
    st.error(f"**{label}**: {msg}")

if not results:
    st.stop()

# Invoice → Excel: collect every document that yielded structured invoice data
# into one workbook (one sheet per invoice) and offer it as a single download.
invoices = [(label, r["invoice"]) for label, r in results if r.get("invoice")]
if want_invoice:
    if invoices:
        st.download_button(
            f"⬇️ Download invoices (Excel) — {len(invoices)} invoice"
            f"{'s' if len(invoices) != 1 else ''}, one row per line item",
            invoice_workbook(invoices),
            file_name="invoices.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="invoices_xlsx",
        )
    else:
        st.info("No invoice data could be extracted (Gemini may be busy — try again, "
                "or check that the uploads are invoices/bills).")

# Single document -> detailed view. Multiple -> summary table + per-doc expanders.
if len(results) == 1:
    st.subheader("Result")
    _render_detail(*results[0])
else:
    st.subheader(f"Results — {len(results)} documents")
    st.dataframe(
        [
            {
                "Document": label,
                "Category": r["classification"].get("category", ""),
                "Type": (r["classification"].get("label", "")
                         if not r["classification"].get("uncertain") else "(uncertain)"),
                "Confidence": f"{r['classification'].get('confidence', 0.0)*100:.0f}%",
                "Summary": _truncate((r.get("understanding") or {}).get("summary", "")),
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
            _render_detail(label, r, nested=True)
