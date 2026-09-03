"""Command-line interface: scan images/PDFs and classify their document type."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .exports import OcrUnavailable, extract_text, searchable_pdf
from .pipeline import (
    ImageDecodeError,
    ScanResult,
    pdf_to_images,
    process_document,
    process_image,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_inputs(paths: list[Path]) -> list[Path]:
    """Expand directories to their image/PDF files; keep files as given."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() == ".pdf"
            ))
        else:
            out.append(p)
    return out


def _load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(f"Could not decode '{path.name}' as an image.")
    return img


def _print_human(payload: dict) -> None:
    cat = f" [{payload['category']}]" if payload.get("category") else ""
    n_pages = payload.get("outputs", {}).get("pages")
    suffix = f"  ({n_pages} pages, classified as one document)" if n_pages else ""
    print(f"\n{payload['input']}{suffix}")
    print(f"Category      : {payload['category'] or 'n/a'} "
          f"({payload['category_confidence']*100:.1f}%)")
    flag = "  (uncertain — trust the category)" if payload["uncertain"] else ""
    print(f"Document type : {payload['document_type']}{cat}  "
          f"({payload['confidence']*100:.1f}% via {payload['classifier']}){flag}")
    if payload["top_k"]:
        print("Top guesses   :")
        for item in payload["top_k"]:
            icat = f" [{item.get('category','')}]" if item.get("category") else ""
            print(f"   - {item['label']:<22}{icat:<24} {item['score']*100:5.1f}%")
    u = payload.get("understanding")
    if u and (u.get("summary") or u.get("fields")):
        if u.get("summary"):
            print(f"Summary       : {u['summary']}")
        for pt in u.get("key_points") or []:
            print(f"   • {pt}")
        if u.get("fields"):
            pairs = ", ".join(f"{k}={v}" for k, v in u["fields"].items())
            print(f"Fields        : {pairs}")
    print(f"Scan          : {payload['outputs']['scan']}")


def _handle_one(label: str, img, args, outdir: Path) -> dict:
    result: ScanResult = process_image(
        img, classify_type=not args.no_classify, binarize=args.bw,
        ocr_engine=args.ocr_engine, understand=args.summarize,
        translate=args.translate, invoice=args.invoice,
    )
    flat_path = outdir / f"{label}_flattened.png"
    clean_path = outdir / f"{label}_scan.png"
    cv2.imwrite(str(flat_path), result.flattened)
    cv2.imwrite(str(clean_path), result.cleaned)

    outputs = {"flattened": str(flat_path), "scan": str(clean_path)}
    if args.searchable_pdf or args.text:
        try:
            if args.searchable_pdf:
                pdf_path = outdir / f"{label}.pdf"
                pdf_path.write_bytes(searchable_pdf(result.cleaned))
                outputs["searchable_pdf"] = str(pdf_path)
            if args.text:
                txt_path = outdir / f"{label}.txt"
                txt_path.write_text(extract_text(result.cleaned))
                outputs["text"] = str(txt_path)
        except OcrUnavailable as e:
            print(f"warning: {e}", file=sys.stderr)

    if (args.summarize or args.translate) and result.ocr_json.get("blocks"):
        ocr_path = outdir / f"{label}_ocr.json"
        ocr_path.write_text(json.dumps(result.ocr_json, indent=2, ensure_ascii=False))
        outputs["ocr_json"] = str(ocr_path)
    if result.ocr_json_en.get("blocks"):
        ocr_en_path = outdir / f"{label}_ocr_en.json"
        ocr_en_path.write_text(json.dumps(result.ocr_json_en, indent=2, ensure_ascii=False))
        outputs["ocr_json_en"] = str(ocr_en_path)

    return {
        "input": label,
        "category": result.category,
        "category_confidence": round(result.category_confidence, 4),
        "document_type": result.doc_type,
        "confidence": round(result.confidence, 4),
        "uncertain": result.uncertain,
        "display_type": result.display_type,
        "top_k": result.classification.get("top_k", []),
        "classifier": result.classification.get("method", "n/a"),
        "dewarp_method": result.dewarp_method,
        "denoise_method": result.denoise_method,
        "understanding": result.understanding or None,
        "invoice": result.invoice or None,
        "outputs": outputs,
    }


def _handle_pdf(path: Path, args, outdir: Path) -> dict:
    """Process every page of a PDF but classify it as ONE document."""
    images = pdf_to_images(path.read_bytes())
    doc = process_document(
        images, classify_type=not args.no_classify, binarize=args.bw,
        ocr_engine=args.ocr_engine, understand=args.summarize,
        translate=args.translate, invoice=args.invoice,
    )
    stem = path.stem
    page_scans: list[str] = []
    for i, page in enumerate(doc.pages, 1):
        flat_path = outdir / f"{stem}_p{i}_flattened.png"
        clean_path = outdir / f"{stem}_p{i}_scan.png"
        cv2.imwrite(str(flat_path), page.flattened)
        cv2.imwrite(str(clean_path), page.cleaned)
        page_scans.append(str(clean_path))

    outputs = {"pages": len(doc.pages), "scan": page_scans[0] if page_scans else "",
               "page_scans": page_scans}
    if args.text:
        try:
            txt_path = outdir / f"{stem}.txt"
            txt_path.write_text("\n\n".join(extract_text(p.cleaned) for p in doc.pages))
            outputs["text"] = str(txt_path)
        except OcrUnavailable as e:
            print(f"warning: {e}", file=sys.stderr)
    if args.searchable_pdf:
        print("warning: --searchable-pdf is per-page and not produced for PDFs; "
              "the PDF is classified as one document.", file=sys.stderr)
    if (args.summarize or args.translate) and doc.ocr_json.get("pages"):
        ocr_path = outdir / f"{stem}_ocr.json"
        ocr_path.write_text(json.dumps(doc.ocr_json, indent=2, ensure_ascii=False))
        outputs["ocr_json"] = str(ocr_path)
    if doc.ocr_json_en.get("pages"):
        ocr_en_path = outdir / f"{stem}_ocr_en.json"
        ocr_en_path.write_text(json.dumps(doc.ocr_json_en, indent=2, ensure_ascii=False))
        outputs["ocr_json_en"] = str(ocr_en_path)

    return {
        "input": stem,
        "category": doc.category,
        "category_confidence": round(doc.category_confidence, 4),
        "document_type": doc.doc_type,
        "confidence": round(doc.confidence, 4),
        "uncertain": doc.uncertain,
        "display_type": doc.display_type,
        "top_k": doc.classification.get("top_k", []),
        "classifier": doc.classification.get("method", "n/a"),
        "pages": doc.classification.get("pages", []),
        "understanding": doc.understanding or None,
        "invoice": doc.invoice or None,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docscan",
        description="Dewarp, denoise and classify document photos or PDFs.",
    )
    parser.add_argument("inputs", nargs="+", help="Image/PDF files or directories")
    parser.add_argument("-o", "--outdir", default="output",
                        help="Directory for output files (default: output/)")
    parser.add_argument("--no-classify", action="store_true",
                        help="Skip document-type classification")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument("--bw", action="store_true",
                        help="Pure black & white scan instead of clean grayscale")
    parser.add_argument("--ocr-engine", choices=["tesseract", "surya"], default=None,
                        help="OCR engine for classification keyword fusion "
                             "(default: tesseract, or $DOC_OCR_ENGINE). 'surya' needs "
                             "the optional surya-ocr extra; it falls back to Tesseract.")
    parser.add_argument("--searchable-pdf", action="store_true",
                        help="Also write a searchable PDF (OCR text layer; needs Tesseract)")
    parser.add_argument("--text", action="store_true",
                        help="Also write extracted OCR text (.txt; needs Tesseract)")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize the document and extract structured fields "
                             "(receipt total, date, …). Uses Gemini when $GEMINI_API_KEY "
                             "is set, otherwise an offline extractive fallback.")
    parser.add_argument("--translate", action="store_true",
                        help="Also write a second OCR JSON with non-English text "
                             "translated to English inline (same regions/bboxes; the "
                             "source string is kept as \"text_original\"). Needs "
                             "$GEMINI_API_KEY.")
    parser.add_argument("--invoice", action="store_true",
                        help="Extract structured invoice data (invoice no, date, name, "
                             "address, total, and a line-item table with item names "
                             "translated to English) and write all inputs to one "
                             "'invoices.xlsx' (one sheet per invoice). Needs $GEMINI_API_KEY.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        parser.error(f"Not found: {', '.join(str(m) for m in missing)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    exit_code = 0
    for path in _iter_inputs(paths):
        try:
            if path.suffix.lower() == ".pdf":
                payloads.append(_handle_pdf(path, args, outdir))
            else:
                payloads.append(_handle_one(path.stem, _load_image(path), args, outdir))
        except ImageDecodeError as e:
            print(f"error: {e}", file=sys.stderr)
            exit_code = 2

    if args.invoice:
        invoices = [(p["input"], p["invoice"]) for p in payloads if p.get("invoice")]
        if invoices:
            from .invoice import invoice_workbook

            xlsx_path = outdir / "invoices.xlsx"
            xlsx_path.write_bytes(invoice_workbook(invoices))
            print(f"Invoices Excel: {xlsx_path} ({len(invoices)} sheet(s))",
                  file=sys.stderr)
        else:
            print("warning: no invoice data extracted (Gemini unavailable or inputs "
                  "were not invoices).", file=sys.stderr)

    if args.json:
        print(json.dumps(payloads if len(payloads) != 1 else payloads[0], indent=2))
    else:
        for payload in payloads:
            _print_human(payload)
        print()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
