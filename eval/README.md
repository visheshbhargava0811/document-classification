# Evaluation

A tiny, honest accuracy check on a **curated** set of labelled images. This is a
regression guard and sanity check — not a public benchmark.

## Provide labelled images

**Option A — folder layout** (the category is the folder name):

```
eval/images/Financial/receipt_01.jpg
eval/images/Financial/invoice_02.png
eval/images/Healthcare/prescription_01.jpg
eval/images/General/letter_01.png
```

Category folder names must match the taxonomy top level: `Financial`,
`Business`, `Government / Identity`, `Healthcare`, `Logistics`, `Education`,
`Forms`, `General`.

**Option B — CSV manifest** at `eval/labels.csv` (paths relative to the repo
root; `type` optional, enables type-level accuracy):

```csv
path,category,type
eval/images/misc/r1.jpg,Financial,Receipt
eval/images/misc/rx.png,Healthcare,Prescription
```

Where to get sample images: your own phone photos of documents, or a few images
from public document-image datasets. Keep the set small and representative.

## Run

```bash
uv run python eval/evaluate.py          # full pipeline: dewarp + clean + classify
uv run python eval/evaluate.py --raw    # classify the raw image (faster)
```

Prints top-1 **category** accuracy (and top-1 **type** accuracy when the CSV
supplies types) and writes a category confusion matrix to
`eval/results/confusion_matrix.png`.

`eval/results/` is git-ignored; the curated inputs and this script are tracked.
