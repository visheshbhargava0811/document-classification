"""Document-type classification.

The flattened document is matched against natural-language descriptions with
**CLIP zero-shot** (no training data needed). Two techniques sharpen the raw
zero-shot signal:

* **Hierarchical inference** — leaf scores are aggregated into the 8 top-level
  categories; the category is decided first (robust) and the specific type is
  then chosen within it.
* **CLIP + OCR fusion** — when Tesseract is available, keyword signals nudge the
  CLIP scores to separate visually-similar but textually-distinct types
  (Invoice vs Bill vs Purchase Order, Cheque, Payslip, Menu, ...).

If CLIP cannot be loaded (fully offline, no cached weights) the classifier falls
back to a pure OCR keyword heuristic. Nothing here imports Streamlit, so the CLI
and Python API stay framework-free; the CLIP model and its text embeddings are
cached with ``lru_cache`` and survive across Streamlit reruns.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np

# Hierarchical taxonomy: category -> {leaf label -> prompt phrasings}.
# An ensemble of phrasings per label improves CLIP zero-shot accuracy. Edit this
# tree to add/remove categories or types — no retraining needed (zero-shot).
TAXONOMY: dict[str, dict[str, list[str]]] = {
    "Financial": {
        "Invoice": ["an invoice", "a business invoice with line items, quantities and an amount due"],
        "Receipt": ["a store receipt", "a printed shopping receipt with prices and a total"],
        "Bill": ["a utility bill", "a bill showing charges and an amount payable"],
        "Bank Statement": ["a bank statement", "a bank account statement listing transactions and a balance"],
        "Cheque": ["a bank cheque", "a printed cheque with payee, amount and a signature line"],
        "Tax Document": ["a tax document", "a tax return or tax form such as a W-2 or 1099"],
        "Payslip": ["a payslip", "a salary pay stub showing earnings and deductions"],
    },
    "Business": {
        "Purchase Order": ["a purchase order", "a purchase order document with a PO number and ordered items"],
        "Quotation": ["a price quotation", "a sales quote or estimate with prices"],
        "Contract": ["a legal contract", "a signed agreement document with clauses"],
        "Report": ["a business report", "a formal report document with sections and headings"],
        "Business Card": ["a business card"],
        "Resume / CV": ["a resume", "a curriculum vitae listing work experience"],
        "Spreadsheet / Table": ["a spreadsheet", "a document that is mostly a data table"],
    },
    "Government / Identity": {
        "Passport": ["a passport", "a passport identity page with a photo"],
        "Driver License": ["a driver's license", "a driving licence ID card"],
        "ID Card": ["an identity card", "a government-issued ID card"],
        "Government Form": ["a government form", "an official government application form"],
    },
    "Healthcare": {
        "Prescription": ["a medical prescription", "a doctor's prescription with medicine and dosage"],
        "Medical Bill": ["a medical bill", "a hospital bill or medical invoice"],
        "Lab Report": ["a medical lab report", "a laboratory test results report"],
        "Medical Form": ["a medical form", "a patient intake or medical history form"],
    },
    "Logistics": {
        "Shipping Label": ["a shipping label", "a parcel shipping label with an address and a barcode"],
        "Packing Slip": ["a packing slip", "a packing list enclosed with a shipment"],
        "Delivery Note": ["a delivery note", "a delivery docket confirming goods delivered"],
        "Waybill": ["a waybill", "an air waybill or consignment note for freight"],
    },
    "Education": {
        "Exam": ["an exam paper", "a test or examination question paper"],
        "Assignment": ["a school assignment", "a homework assignment sheet"],
        "Notes": ["study notes", "a page of lecture or study notes"],
        "Transcript": ["an academic transcript", "a school grade report or marksheet"],
        "Certificate": ["a certificate or diploma", "an award certificate"],
    },
    "Forms": {
        "Application": ["an application form", "a filled-in application form"],
        "Registration": ["a registration form", "a sign-up or enrollment form"],
        "Survey": ["a survey form", "a questionnaire with questions to answer"],
        "Checklist": ["a checklist", "a document with a list of checkbox items"],
    },
    "General": {
        "Letter": ["a printed formal letter", "a typed letter document"],
        "Newspaper": ["a newspaper page", "a printed news article with columns"],
        "Book": ["a page from a book", "a printed book page with paragraphs"],
        "Menu": ["a restaurant menu", "a menu listing dishes and prices"],
        "Flyer": ["a promotional flyer", "an advertising flyer or poster"],
        "Handwritten Note": ["a handwritten note", "a page of handwritten text"],
        "Photograph": ["a photograph", "a natural photo that is not a document"],
        "Other": ["a document", "a general document that does not fit other categories"],
    },
}

# Derived flat views used by the classifier.
# label -> list of prompt phrasings
DOCUMENT_TYPES: dict[str, list[str]] = {
    label: prompts for leaves in TAXONOMY.values() for label, prompts in leaves.items()
}
# label -> category
CATEGORY_OF: dict[str, str] = {
    label: category for category, leaves in TAXONOMY.items() for label in leaves
}

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Templates wrapped around each phrase to stabilise zero-shot scores (ensemble).
PROMPT_TEMPLATES = (
    "a photo of {}",
    "a scanned image of {}",
    "a document that is {}",
    "{}",
)

# Fusion / confidence tuning.
OCR_BOOST = 1.5            # logit units added per matched keyword (capped below)
OCR_MAX_HITS = 4          # cap keyword hits so OCR can nudge but not dominate CLIP
LEAF_CONF_MIN = 0.30      # below this top-leaf prob -> report category only
LEAF_MARGIN_MIN = 0.10    # top1-top2 leaf prob gap below this -> report category only


# --------------------------------------------------------------------------- #
# CLIP model + cached text embeddings
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_clip():
    """Load and cache the CLIP model + processor. Returns None on failure."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        model.eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return model, processor, torch
    except Exception:
        return None


@lru_cache(maxsize=1)
def _text_features():
    """Encode all taxonomy prompts once and cache the normalised embeddings.

    Returns ``(text_emb, owners, logit_scale)`` or None. ``text_emb`` is a
    tensor of shape ``[n_prompts, d]``; ``owners[i]`` is the leaf label that
    prompt *i* belongs to. The prompts are fixed, so this runs a single time.
    """
    loaded = _load_clip()
    if loaded is None:
        return None
    model, processor, torch = loaded

    prompts: list[str] = []
    owners: list[str] = []
    for label, phrases in DOCUMENT_TYPES.items():
        for phrase in phrases:
            for template in PROMPT_TEMPLATES:
                prompts.append(template.format(phrase))
                owners.append(label)

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Canonical CLIP text embedding (stable across transformers versions):
        # project the text-model pooled output, then L2-normalise.
        pooled = model.text_model(**inputs).pooler_output
        text_emb = model.text_projection(pooled)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    logit_scale = float(model.logit_scale.exp().item())
    return text_emb, owners, logit_scale


def _clip_label_logits(image_rgb: np.ndarray) -> Optional[dict[str, float]]:
    """Per-leaf CLIP logit (max-pooled over the prompt ensemble). None if no CLIP."""
    loaded = _load_clip()
    feats = _text_features()
    if loaded is None or feats is None:
        return None
    model, processor, torch = loaded
    text_emb, owners, logit_scale = feats
    from PIL import Image

    inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt")
    with torch.no_grad():
        pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        img_emb = model.visual_projection(pooled)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    logits = (logit_scale * img_emb @ text_emb.T).squeeze(0).tolist()

    per_label: dict[str, float] = {}
    for owner, logit in zip(owners, logits):
        per_label[owner] = max(per_label.get(owner, -1e9), logit)
    return per_label


# --------------------------------------------------------------------------- #
# Hierarchical + confidence-aware assembly
# --------------------------------------------------------------------------- #
def _softmax(values: np.ndarray) -> np.ndarray:
    v = values - values.max()
    e = np.exp(v)
    return e / e.sum()


def _finalize(label_logits: dict[str, float], method: str, top_k: int) -> dict:
    """Turn per-leaf logits into a hierarchical, confidence-aware result.

    Decides the category first (max-pooled over its leaves, softmax across the 8
    categories), then the best leaf within that category. Flags ``uncertain``
    when the leaf choice is weak, in which case callers should present the
    category rather than a shaky specific type.
    """
    labels = list(DOCUMENT_TYPES.keys())
    logits = np.array([label_logits.get(lbl, -1e9) for lbl in labels], dtype=np.float64)
    leaf_probs = _softmax(logits)
    leaf_prob = dict(zip(labels, leaf_probs))

    # Category distribution: each category scored by its strongest leaf.
    categories = list(TAXONOMY.keys())
    cat_logit = np.array(
        [max(label_logits.get(lbl, -1e9) for lbl in TAXONOMY[cat]) for cat in categories],
        dtype=np.float64,
    )
    cat_probs = _softmax(cat_logit)
    top_cat_idx = int(cat_probs.argmax())
    top_category = categories[top_cat_idx]
    category_confidence = float(cat_probs[top_cat_idx])

    # Best leaf within the chosen category.
    in_cat = list(TAXONOMY[top_category].keys())
    top_label = max(in_cat, key=lambda lbl: leaf_prob[lbl])
    leaf_confidence = float(leaf_prob[top_label])

    # Global ranking for transparency / uncertainty margin.
    ranked = sorted(labels, key=lambda lbl: leaf_prob[lbl], reverse=True)
    margin = float(leaf_prob[ranked[0]] - leaf_prob[ranked[1]]) if len(ranked) > 1 else 1.0
    uncertain = leaf_confidence < LEAF_CONF_MIN or margin < LEAF_MARGIN_MIN

    return {
        "method": method,
        "label": top_label,
        "category": top_category,
        "confidence": leaf_confidence,
        "category_confidence": category_confidence,
        "uncertain": uncertain,
        "top_k": [
            {"label": lbl, "category": CATEGORY_OF.get(lbl, ""), "score": float(leaf_prob[lbl])}
            for lbl in ranked[:top_k]
        ],
    }


def classify_clip(image_rgb: np.ndarray, top_k: int = 5) -> Optional[dict]:
    """Pure-CLIP classification (no OCR fusion). None if CLIP is unavailable."""
    label_logits = _clip_label_logits(image_rgb)
    if label_logits is None:
        return None
    return _finalize(label_logits, method="clip", top_k=top_k)


# --------------------------------------------------------------------------- #
# OCR keyword signals (offline-safe; also used to fuse with CLIP)
# --------------------------------------------------------------------------- #
# Keyword hints. Labels must match TAXONOMY leaves.
_KEYWORDS = {
    "Receipt": ["total", "subtotal", "change", "cash", "qty", "receipt"],
    "Invoice": ["invoice", "bill to", "amount due", "invoice no", "due date"],
    "Bill": ["amount payable", "billing", "utility", "meter", "due date"],
    "Bank Statement": ["balance", "statement", "account number", "transaction"],
    "Cheque": ["pay to the order", "cheque", "check no", "payee", "void"],
    "Payslip": ["gross pay", "net pay", "earnings", "deductions", "pay period"],
    "Tax Document": ["tax", "w-2", "1099", "taxable", "irs", "return"],
    "Purchase Order": ["purchase order", "p.o.", "po number", "ship to"],
    "Quotation": ["quotation", "quote", "estimate", "valid until"],
    "Contract": ["agreement", "party", "hereby", "terms and conditions", "signature"],
    "Resume / CV": ["experience", "education", "skills", "curriculum vitae", "resume"],
    "Prescription": ["rx", "prescription", "tablet", "mg", "dosage"],
    "Medical Bill": ["patient", "hospital", "diagnosis", "amount due", "medical"],
    "Lab Report": ["result", "reference range", "specimen", "test", "laboratory"],
    "Shipping Label": ["ship to", "tracking", "barcode", "carrier", "postage"],
    "Menu": ["menu", "starters", "mains", "dessert", "appetizer"],
    "Transcript": ["gpa", "grade", "semester", "credits", "transcript"],
    "Application": ["please fill", "date of birth", "signature", "application"],
    "Survey": ["questionnaire", "please rate", "strongly agree", "survey"],
    "Letter": ["dear", "sincerely", "regards", "yours faithfully"],
}


def _ocr_text(image_rgb: np.ndarray) -> str:
    try:
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.fromarray(image_rgb)).lower()
    except Exception:
        return ""


def _ocr_label_scores(text: str) -> dict[str, int]:
    """Keyword-hit count per leaf (capped), plus a handwritten-note hint."""
    scores: dict[str, int] = {}
    for label, kws in _KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits:
            scores[label] = min(hits, OCR_MAX_HITS)
    if _handwritten_ratio(text) > 0.5:
        scores["Handwritten Note"] = scores.get("Handwritten Note", 0) + 3
    return scores


def classify_ocr(image_rgb: np.ndarray, top_k: int = 5) -> dict:
    """Offline classifier using only OCR keyword signals."""
    text = _ocr_text(image_rgb)
    if not text.strip():
        return {
            "method": "ocr",
            "label": "unknown (no text detected)",
            "category": "",
            "confidence": 0.0,
            "category_confidence": 0.0,
            "uncertain": True,
            "top_k": [],
        }

    scores = _ocr_label_scores(text)
    if not scores:
        return {
            "method": "ocr",
            "label": "Letter",
            "category": CATEGORY_OF.get("Letter", ""),
            "confidence": 0.0,
            "category_confidence": 0.0,
            "uncertain": True,
            "top_k": [],
        }

    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label = ranked[0][0]
    return {
        "method": "ocr",
        "label": top_label,
        "category": CATEGORY_OF.get(top_label, ""),
        "confidence": ranked[0][1] / total,
        "category_confidence": ranked[0][1] / total,
        "uncertain": ranked[0][1] < 2,
        "top_k": [
            {"label": k, "category": CATEGORY_OF.get(k, ""), "score": v / total}
            for k, v in ranked[:top_k]
        ],
    }


def _handwritten_ratio(text: str) -> float:
    """Very rough proxy: OCR gibberish rate hints at handwriting."""
    words = re.findall(r"[a-zA-Z]+", text)
    if len(words) < 3:
        return 0.0
    short_or_odd = sum(1 for w in words if len(w) <= 2)
    return short_or_odd / len(words)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def classify(image_rgb: np.ndarray, top_k: int = 5, prefer: str = "clip",
             use_ocr: bool = True) -> dict:
    """Classify the document type.

    Uses CLIP when available, fuses OCR keyword signals into the CLIP scores when
    Tesseract is installed, and falls back to a pure OCR heuristic if CLIP cannot
    be loaded. The result always carries both a ``category`` and a specific
    ``label`` plus an ``uncertain`` flag (True when the specific type is a weak
    guess and only the category should be trusted).
    """
    if prefer == "clip":
        label_logits = _clip_label_logits(image_rgb)
        if label_logits is not None:
            method = "clip"
            if use_ocr:
                ocr_scores = _ocr_label_scores(_ocr_text(image_rgb))
                if ocr_scores:
                    for lbl, hits in ocr_scores.items():
                        label_logits[lbl] = label_logits.get(lbl, 0.0) + OCR_BOOST * hits
                    method = "clip+ocr"
            return _finalize(label_logits, method=method, top_k=top_k)
    return classify_ocr(image_rgb, top_k=top_k)
