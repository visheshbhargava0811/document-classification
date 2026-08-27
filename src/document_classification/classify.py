"""Document-type classification via CLIP zero-shot.

No training data is required: the flattened document image is compared against a
set of natural-language type descriptions and the best match is returned. If the
CLIP model cannot be loaded (e.g. fully offline with no cached weights), a
lightweight OCR keyword heuristic is used instead.
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


def classify_clip(image_rgb: np.ndarray, top_k: int = 5) -> Optional[dict]:
    loaded = _load_clip()
    if loaded is None:
        return None
    model, processor, torch = loaded
    from PIL import Image

    labels = list(DOCUMENT_TYPES.keys())
    prompts, prompt_owner = [], []
    for label in labels:
        for phrase in DOCUMENT_TYPES[label]:
            prompts.append(f"a photo of {phrase}")
            prompt_owner.append(label)

    pil = Image.fromarray(image_rgb)
    inputs = processor(text=prompts, images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # image-text similarity logits, shape (1, n_prompts)
        logits = outputs.logits_per_image.squeeze(0)

    # Max-pool the prompt ensemble per label, then softmax across labels.
    label_logits = {}
    for label, logit in zip(prompt_owner, logits.tolist()):
        label_logits[label] = max(label_logits.get(label, -1e9), logit)
    keys = list(label_logits.keys())
    vals = torch.tensor([label_logits[k] for k in keys])
    probs = torch.softmax(vals, dim=0).tolist()

    ranked = sorted(zip(keys, probs), key=lambda kv: kv[1], reverse=True)
    return {
        "method": "clip",
        "label": ranked[0][0],
        "category": CATEGORY_OF.get(ranked[0][0], ""),
        "confidence": float(ranked[0][1]),
        "top_k": [
            {"label": k, "category": CATEGORY_OF.get(k, ""), "score": float(p)}
            for k, p in ranked[:top_k]
        ],
    }


# --------------------------------------------------------------------------- #
# OCR keyword fallback (offline-safe)
# --------------------------------------------------------------------------- #
# Keyword hints for the offline OCR fallback. Labels must match TAXONOMY leaves.
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


def classify_ocr(image_rgb: np.ndarray, top_k: int = 5) -> dict:
    text = ""
    try:
        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.fromarray(image_rgb)).lower()
    except Exception:
        pass

    scores = {}
    for label, kws in _KEYWORDS.items():
        scores[label] = sum(1 for kw in kws if kw in text)

    handwritten_ratio = _handwritten_ratio(text)
    if handwritten_ratio > 0.5:
        scores["Handwritten Note"] = scores.get("Handwritten Note", 0) + 3

    if not text.strip():
        return {
            "method": "ocr",
            "label": "unknown (no text detected)",
            "category": "",
            "confidence": 0.0,
            "top_k": [],
        }

    total = sum(scores.values()) or 1
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label = ranked[0][0] if ranked[0][1] > 0 else "Letter"
    return {
        "method": "ocr",
        "label": top_label,
        "category": CATEGORY_OF.get(top_label, ""),
        "confidence": ranked[0][1] / total,
        "top_k": [
            {"label": k, "category": CATEGORY_OF.get(k, ""), "score": v / total}
            for k, v in ranked[:top_k] if v > 0
        ],
    }


def _handwritten_ratio(text: str) -> float:
    """Very rough proxy: OCR gibberish rate hints at handwriting."""
    words = re.findall(r"[a-zA-Z]+", text)
    if len(words) < 3:
        return 0.0
    short_or_odd = sum(1 for w in words if len(w) <= 2)
    return short_or_odd / len(words)


def classify(image_rgb: np.ndarray, top_k: int = 5, prefer: str = "clip") -> dict:
    """Classify the document type. Falls back to OCR if CLIP is unavailable."""
    if prefer == "clip":
        result = classify_clip(image_rgb, top_k=top_k)
        if result is not None:
            return result
    return classify_ocr(image_rgb, top_k=top_k)
