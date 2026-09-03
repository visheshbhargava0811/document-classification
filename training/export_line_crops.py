"""Export OCR line-crops from invoices for handwriting-annotation.

This turns your real invoices into a labelling-ready dataset for fine-tuning a
handwriting recognizer (TrOCR / PaddleOCR) — step 2 of the accuracy plan (see
``dataset-and-training-plan.md``). For each page it runs the pipeline (dewarp +
denoise) and Surya OCR, then crops every detected text region to its own PNG and
writes a ``manifest.csv`` whose ``transcription`` column is pre-filled with
Surya's guess. Import the manifest into Label Studio (or edit the CSV directly)
and correct each transcription — that corrected text is your ground truth.

Because the target is Hindi, keep only crops whose Surya text contains Devanagari
by default (``--script devanagari``); use ``--script any`` to keep everything.

Usage:
    .venv/bin/python training/export_line_crops.py "/path/to/invoices/*.pdf" -o training/crops
    .venv/bin/python training/export_line_crops.py inv1.pdf inv2.jpg --script any
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2

# Make the src/ package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_classification import classify as _classify  # noqa: E402
from document_classification.pipeline import (  # noqa: E402
    load_image,
    pdf_to_images,
    process_image,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MIN_SIDE = 8  # skip degenerate boxes


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def _expand(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in inputs:
        hits = [Path(p) for p in glob.glob(pattern)] or [Path(pattern)]
        for p in hits:
            if p.is_dir():
                paths.extend(sorted(f for f in p.iterdir()
                                    if f.suffix.lower() in IMAGE_EXTS | {".pdf"}))
            else:
                paths.append(p)
    return paths


def _pages(path: Path):
    """Yield (page_index, BGR image) for a PDF or single image."""
    if path.suffix.lower() == ".pdf":
        yield from enumerate(pdf_to_images(path.read_bytes()), 1)
    else:
        yield 1, load_image(path)


def export(inputs: list[str], out_dir: Path, engine: str, script: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    n = 0
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop", "transcription", "source", "page", "bbox", "confidence"])
        for path in _expand(inputs):
            if not path.exists():
                print(f"skip (not found): {path}", file=sys.stderr)
                continue
            stem = path.stem.replace(" ", "_")
            for page_idx, bgr in _pages(path):
                # Same preprocessing the app uses, so crops match what OCR sees.
                res = process_image(bgr, classify_type=False)
                rgb = cv2.cvtColor(res.flattened, cv2.COLOR_BGR2RGB)
                ocr = _classify.ocr_extract(rgb, engine=engine)
                for b in ocr.get("blocks", []):
                    text = str(b.get("text") or "").strip()
                    if not text:
                        continue
                    if script == "devanagari" and not _has_devanagari(text):
                        continue
                    box = b.get("bbox")
                    if not (isinstance(box, (list, tuple)) and len(box) == 4):
                        continue
                    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
                    x0, y0 = max(0, x0), max(0, y0)
                    if x1 - x0 < _MIN_SIDE or y1 - y0 < _MIN_SIDE:
                        continue
                    crop = rgb[y0:y1, x0:x1]
                    name = f"{stem}_p{page_idx}_{n:04d}.png"
                    cv2.imwrite(str(out_dir / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    writer.writerow([name, text, path.name, page_idx,
                                     f"{x0},{y0},{x1},{y1}", b.get("confidence", "")])
                    n += 1
                print(f"{path.name} p{page_idx}: kept {n} crops so far", file=sys.stderr)
    print(f"\nWrote {n} crops + {manifest}")
    print("Next: import manifest.csv into Label Studio (image + pre-filled "
          "transcription) and correct each line — that becomes your ground truth.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Export OCR line-crops for annotation.")
    ap.add_argument("inputs", nargs="+", help="PDF/image files, globs, or directories")
    ap.add_argument("-o", "--out", default="training/crops", help="Output dir (default: training/crops)")
    ap.add_argument("--engine", choices=["surya", "tesseract"], default="surya",
                    help="OCR engine for detection + pre-fill (default: surya)")
    ap.add_argument("--script", choices=["devanagari", "any"], default="devanagari",
                    help="Keep only Devanagari crops (default) or every crop")
    args = ap.parse_args()
    raise SystemExit(export(args.inputs, Path(args.out), args.engine, args.script))


if __name__ == "__main__":
    main()
