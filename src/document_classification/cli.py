"""Command-line interface: scan an image and classify its document type."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .pipeline import process_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docscan",
        description="Dewarp, denoise and classify a document photo.",
    )
    parser.add_argument("image", help="Path to the input document photo")
    parser.add_argument(
        "-o", "--outdir", default="output", help="Directory for output images (default: output/)"
    )
    parser.add_argument(
        "--no-classify", action="store_true", help="Skip document-type classification"
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    parser.add_argument(
        "--bw", action="store_true",
        help="Pure black & white scan instead of clean grayscale (smaller, can break logos)",
    )
    args = parser.parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        parser.error(f"Image not found: {in_path}")

    result = process_path(in_path, classify_type=not args.no_classify, binarize=args.bw)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem
    flat_path = outdir / f"{stem}_flattened.png"
    clean_path = outdir / f"{stem}_scan.png"
    cv2.imwrite(str(flat_path), result.flattened)
    cv2.imwrite(str(clean_path), result.cleaned)

    payload = {
        "input": str(in_path),
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
        "outputs": {"flattened": str(flat_path), "scan": str(clean_path)},
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        cat = f" [{payload['category']}]" if payload.get("category") else ""
        print(f"\nCategory      : {payload['category'] or 'n/a'} "
              f"({payload['category_confidence']*100:.1f}%)")
        flag = "  (uncertain — trust the category)" if payload["uncertain"] else ""
        print(f"Document type : {payload['document_type']}{cat}  "
              f"({payload['confidence']*100:.1f}% via {payload['classifier']}){flag}")
        if payload["top_k"]:
            print("Top guesses   :")
            for item in payload["top_k"]:
                icat = f" [{item.get('category','')}]" if item.get("category") else ""
                print(f"   - {item['label']:<22}{icat:<24} {item['score']*100:5.1f}%")
        print(f"Dewarp        : {payload['dewarp_method']}")
        print(f"Denoise       : {payload['denoise_method']}")
        print(f"Flattened     : {flat_path}")
        print(f"Scan          : {clean_path}\n")


if __name__ == "__main__":
    sys.exit(main())
