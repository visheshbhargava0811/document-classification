# Document Scanner + Classifier

Turn a **photo of a document** into a clean, flattened scan **and** predict what
kind of document it is — receipt, handwritten note, invoice, form, ID card,
resume, and more.

This grew out of a deep-learning course project (corner detection + dewarping +
denoising) and adds a **document-type classifier** plus a real, runnable app
(web UI, CLI, and Python API).

## Pipeline

```
photo  ─►  1. detect page & flatten  ─►  2. clean / denoise  ─►  3. classify type
             (CNN or OpenCV)              (U-Net or classic)      (CLIP zero-shot)
```

1. **Dewarp** — finds the document's four corners and perspective-corrects it to
   a flat, top-down view.
   - Uses the trained `DocumentCornerModel` CNN if `dl_project_doc_extract_weights.pt`
     is present; otherwise a robust **OpenCV contour** detector.
2. **Denoise / enhance** — turns the page into a crisp scan.
   - Uses the trained `Unet` if `model.pt` is present; otherwise **classic**
     OpenCV enhancement (non-local-means + adaptive threshold).
3. **Classify** — **CLIP zero-shot**: the page is matched against natural-language
   type descriptions, so no labeled training data is required. Falls back to an
   **OCR keyword** heuristic (Tesseract) when CLIP can't be loaded (fully offline).

> The `.pt` model weights and datasets are **optional**. The app is fully
> functional without them thanks to the classic-CV fallbacks. Drop the trained
> weight files into the project root (or a `weights/` folder) to use the neural
> versions — the architectures match the originals so the files load as-is.

## Install

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # core app
uv sync --extra ocr    # + Tesseract OCR fallback (needs the `tesseract` binary)
```

On macOS the OCR binary is `brew install tesseract`.

## Usage

### Web app (recommended)

```bash
uv run document-classification-web
# then open http://localhost:8501
```

Upload a document photo and you'll see the detected page, the flattened image,
the cleaned scan, the predicted type with confidence, and a download button.

### Command line

```bash
uv run docscan path/to/photo.jpg
uv run docscan path/to/photo.jpg --json          # machine-readable
uv run docscan path/to/photo.jpg -o out/ --no-classify
```

Outputs the flattened image and cleaned scan to `output/` and prints the
predicted document type.

### Python API

```python
from document_classification import process_path

result = process_path("photo.jpg")
print(result.category, result.doc_type, result.confidence)   # e.g. "Financial" "Receipt" 0.96
# result.flattened / result.cleaned are numpy images; result.classification has top-k
```

## Document types recognized

Types are organised into a two-level taxonomy — each result reports both the
**category** and the specific **type**:

- **Financial** — Invoice, Receipt, Bill, Bank Statement, Cheque, Tax Document, Payslip
- **Business** — Purchase Order, Quotation, Contract, Report, Business Card, Resume / CV, Spreadsheet / Table
- **Government / Identity** — Passport, Driver License, ID Card, Government Form
- **Healthcare** — Prescription, Medical Bill, Lab Report, Medical Form
- **Logistics** — Shipping Label, Packing Slip, Delivery Note, Waybill
- **Education** — Exam, Assignment, Notes, Transcript, Certificate
- **Forms** — Application, Registration, Survey, Checklist
- **General** — Letter, Newspaper, Book, Menu, Flyer, Handwritten Note, Photograph, Other

Edit the `TAXONOMY` tree in `src/document_classification/classify.py` to add or
remove categories and types — no retraining needed (zero-shot). Note that
fine-grained distinctions (e.g. Invoice vs Bill vs Purchase Order) are heuristic.

## Project layout

```
src/document_classification/
  models.py     # DocumentCornerModel (CNN) + Unet — match the original notebooks
  dewarp.py     # stage 1: corner detection (CNN / OpenCV) + perspective warp
  denoise.py    # stage 2: U-Net / classic enhancement
  classify.py   # stage 3: CLIP zero-shot + OCR fallback
  pipeline.py   # orchestration -> ScanResult
  cli.py        # `docscan` command
  app.py        # Streamlit web UI
  web.py        # launcher for the web command
```

The original research notebooks (`project_*.ipynb`) and report remain in the
repo for reference.

## Notes

- First run downloads the CLIP model (`openai/clip-vit-base-patch32`, ~600 MB),
  cached afterward.
- Classification quality is best on a clearly framed document; the CLIP labels
  are heuristic, not a guarantee.
