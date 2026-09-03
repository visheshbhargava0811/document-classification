"""Classifier logic that doesn't need CLIP: assembly, OCR scoring, fusion, fallback."""
from __future__ import annotations

import numpy as np

from document_classification import classify as C


def test_finalize_picks_category_and_leaf():
    logits = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    logits["Receipt"] = 30.0
    logits["Bill"] = 10.0
    res = C._finalize(logits, method="clip", top_k=5)
    assert res["category"] == "Financial"
    assert res["label"] == "Receipt"
    assert res["uncertain"] is False
    assert res["top_k"][0]["label"] == "Receipt"
    assert 0.0 <= res["confidence"] <= 1.0


def test_finalize_flags_uncertain_when_leaves_are_close():
    logits = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    # Two tied leaves in the same category -> ~zero margin -> uncertain.
    logits["Invoice"] = 8.0
    logits["Bill"] = 8.0
    res = C._finalize(logits, method="clip", top_k=5)
    assert res["category"] == "Financial"
    assert res["uncertain"] is True


def test_ocr_label_scores_detect_keywords():
    text = "invoice no 123 bill to acme amount due $50 due date today"
    scores = C._ocr_label_scores(text)
    assert scores.get("Invoice", 0) >= 1
    assert "Invoice" in scores


def test_ocr_scores_are_capped():
    text = " ".join(["total", "subtotal", "cash", "qty", "receipt", "change"])  # 6 receipt hits
    scores = C._ocr_label_scores(text)
    assert scores["Receipt"] <= C.OCR_MAX_HITS


def test_classify_falls_back_to_ocr_when_clip_unavailable(monkeypatch):
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: None)
    monkeypatch.setattr(C, "_ocr_text", lambda img, engine=None: "total subtotal cash receipt change")
    res = C.classify(np.zeros((8, 8, 3), np.uint8))
    assert res["method"] == "ocr"
    assert res["label"] == "Receipt"
    assert res["category"] == "Financial"


def test_classify_fuses_ocr_into_clip(monkeypatch):
    # CLIP marginally prefers Bill; OCR keywords for Invoice should flip it.
    base = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    base["Bill"] = 12.0
    base["Invoice"] = 11.0
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: dict(base))
    monkeypatch.setattr(C, "_ocr_text",
                        lambda img, engine=None: "invoice no 7 bill to acme amount due due date")
    res = C.classify(np.zeros((8, 8, 3), np.uint8), use_ocr=True)
    assert res["method"] == "clip+ocr"
    assert res["label"] == "Invoice"


def test_classify_no_ocr_signal_stays_clip(monkeypatch):
    base = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    base["Receipt"] = 25.0
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: dict(base))
    monkeypatch.setattr(C, "_ocr_text", lambda img, engine=None: "")  # no text -> no fusion
    res = C.classify(np.zeros((8, 8, 3), np.uint8), use_ocr=True)
    assert res["method"] == "clip"
    assert res["label"] == "Receipt"


def test_ocr_text_surya_falls_back_to_tesseract(monkeypatch):
    # Surya unavailable/empty -> _ocr_text should fall back to the tesseract path.
    monkeypatch.setattr(C, "_surya_text", lambda img, lower=True: "")
    monkeypatch.setattr(C, "_tesseract_text", lambda img, lower=True: "fallback text")
    assert C._ocr_text(np.zeros((8, 8, 3), np.uint8), engine="surya") == "fallback text"


def test_ocr_text_surya_used_when_it_returns_text(monkeypatch):
    monkeypatch.setattr(C, "_surya_text", lambda img, lower=True: "surya text")
    monkeypatch.setattr(C, "_tesseract_text", lambda img, lower=True: "should not be used")
    assert C._ocr_text(np.zeros((8, 8, 3), np.uint8), engine="surya") == "surya text"


def test_classify_method_tags_surya_engine(monkeypatch):
    base = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    base["Bill"] = 12.0
    base["Invoice"] = 11.0
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: dict(base))
    monkeypatch.setattr(C, "_ocr_text",
                        lambda img, engine=None: "invoice no 7 amount due due date")
    res = C.classify(np.zeros((8, 8, 3), np.uint8), use_ocr=True, ocr_engine="surya")
    assert res["method"] == "clip+ocr:surya"
    assert res["label"] == "Invoice"


# --------------------------------------------------------------------------- #
# Document-level classification (a PDF is one document with one type)
# --------------------------------------------------------------------------- #
def _page_logits(label: str, value: float = 30.0):
    logits = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    logits[label] = value
    return logits


def test_classify_document_averages_to_dominant_category(monkeypatch):
    # 3 Exam pages + 1 Cheque page: the document should be Education overall,
    # not flip to the single minority page's type.
    per_page = [_page_logits("Exam"), _page_logits("Exam"),
                _page_logits("Exam"), _page_logits("Cheque")]
    it = iter(per_page)
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: next(it))
    monkeypatch.setattr(C, "_ocr_text", lambda img, engine=None: "")
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in per_page]
    res = C.classify_document(imgs)
    assert res["category"] == "Education"
    assert res["label"] == "Exam"
    assert res["method"] == "clip-doc"
    # Per-page breakdown is preserved for transparency.
    assert [p["page"] for p in res["pages"]] == [1, 2, 3, 4]
    assert res["pages"][3]["category"] == "Financial"  # the minority page


def test_classify_document_minority_page_does_not_flip(monkeypatch):
    # One strong Receipt page cannot outweigh four Contract pages once averaged.
    per_page = [_page_logits("Receipt", 60.0)] + [_page_logits("Contract", 30.0)] * 4
    it = iter(per_page)
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: next(it))
    monkeypatch.setattr(C, "_ocr_text", lambda img, engine=None: "")
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in per_page]
    res = C.classify_document(imgs)
    assert res["category"] == "Business"
    assert res["label"] == "Contract"


def test_classify_document_ocr_fusion_tags_surya(monkeypatch):
    per_page = [_page_logits("Exam"), _page_logits("Exam")]
    it = iter(per_page)
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: next(it))
    monkeypatch.setattr(C, "_ocr_text",
                        lambda img, engine=None: "transcript gpa grade semester credits")
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in per_page]
    res = C.classify_document(imgs, ocr_engine="surya")
    assert res["method"] == "clip+ocr:surya-doc"


def test_classify_document_falls_back_to_ocr_without_clip(monkeypatch):
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: None)
    monkeypatch.setattr(C, "_ocr_text",
                        lambda img, engine=None: "total subtotal cash receipt change")
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    res = C.classify_document(imgs)
    assert res["method"] == "ocr-doc"
    assert res["category"] == "Financial"
    assert res["label"] == "Receipt"
