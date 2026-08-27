"""Evaluate the classifier on a small, hand-labelled image set.

This measures accuracy on *your curated set* — it is a regression guard and a
sanity check, not a public benchmark. Provide labelled images one of two ways:

1. Folder layout (category = folder name)::

       eval/images/Financial/receipt1.jpg
       eval/images/Healthcare/rx.png

2. CSV manifest ``eval/labels.csv`` with columns ``path,category[,type]``
   (paths relative to the repo root).

Run::

    uv run python eval/evaluate.py                 # full pipeline (dewarp+clean+classify)
    uv run python eval/evaluate.py --raw           # classify raw images (faster)

Outputs top-1 category and top-1 type accuracy and writes a category confusion
matrix to ``eval/results/confusion_matrix.png``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from document_classification.classify import TAXONOMY, classify  # noqa: E402
from document_classification.pipeline import process_image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_labels() -> list[tuple[Path, str, str | None]]:
    """Return (path, category, type_or_None) triples from CSV or folder layout."""
    items: list[tuple[Path, str, str | None]] = []
    csv_path = REPO / "eval" / "labels.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                p = (REPO / row["path"]).resolve()
                items.append((p, row["category"].strip(), (row.get("type") or "").strip() or None))
        return items

    images_dir = REPO / "eval" / "images"
    if images_dir.exists():
        for category_dir in sorted(images_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for img in sorted(category_dir.rglob("*")):
                if img.suffix.lower() in IMAGE_EXTS:
                    items.append((img, category, None))
    return items


def _predict(path: Path, raw: bool) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {path}")
    if raw:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return classify(rgb)
    return process_image(bgr, classify_type=True).classification


def _confusion_png(categories: list[str], matrix: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(categories)), categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(categories)), categories, fontsize=8)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Category confusion matrix")
    for i in range(len(categories)):
        for j in range(len(categories)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nConfusion matrix -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="store_true",
                    help="Classify the raw image (skip dewarp/denoise) — faster")
    args = ap.parse_args()

    items = _load_labels()
    if not items:
        print("No eval images found. Add labelled images under eval/images/<Category>/ "
              "or an eval/labels.csv manifest. See eval/README.md.")
        return 0

    categories = list(TAXONOMY.keys())
    cat_index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=int)

    cat_correct = type_correct = type_total = 0
    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_correct: dict[str, int] = defaultdict(int)

    for path, true_cat, true_type in items:
        pred = _predict(path, args.raw)
        pred_cat = pred.get("category", "")
        pred_type = pred.get("label", "")
        per_cat_total[true_cat] += 1
        if pred_cat == true_cat:
            cat_correct += 1
            per_cat_correct[true_cat] += 1
        if true_type is not None:
            type_total += 1
            type_correct += int(pred_type == true_type)
        if true_cat in cat_index and pred_cat in cat_index:
            matrix[cat_index[true_cat], cat_index[pred_cat]] += 1
        flag = "✓" if pred_cat == true_cat else "✗"
        print(f"{flag} {path.name:<28} true={true_cat:<22} pred={pred_cat} ({pred_type})")

    n = len(items)
    print(f"\nTop-1 category accuracy : {cat_correct}/{n} = {cat_correct/n*100:.1f}%")
    if type_total:
        print(f"Top-1 type accuracy     : {type_correct}/{type_total} = "
              f"{type_correct/type_total*100:.1f}%")
    print("\nPer-category:")
    for c in categories:
        if per_cat_total[c]:
            print(f"  {c:<22} {per_cat_correct[c]}/{per_cat_total[c]}")

    _confusion_png(categories, matrix, REPO / "eval" / "results" / "confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
