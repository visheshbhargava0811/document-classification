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
    monkeypatch.setattr(C, "_ocr_text", lambda img: "total subtotal cash receipt change")
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
    monkeypatch.setattr(C, "_ocr_text", lambda img: "invoice no 7 bill to acme amount due due date")
    res = C.classify(np.zeros((8, 8, 3), np.uint8), use_ocr=True)
    assert res["method"] == "clip+ocr"
    assert res["label"] == "Invoice"


def test_classify_no_ocr_signal_stays_clip(monkeypatch):
    base = {lbl: 0.0 for lbl in C.DOCUMENT_TYPES}
    base["Receipt"] = 25.0
    monkeypatch.setattr(C, "_clip_label_logits", lambda img: dict(base))
    monkeypatch.setattr(C, "_ocr_text", lambda img: "")  # no text -> no fusion
    res = C.classify(np.zeros((8, 8, 3), np.uint8), use_ocr=True)
    assert res["method"] == "clip"
    assert res["label"] == "Receipt"
