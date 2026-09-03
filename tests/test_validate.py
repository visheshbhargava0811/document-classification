"""Tests for the cheap invoice validations: amount reconciliation + GSTIN checksum."""
from document_classification.validate import (
    _gstin_check_digit,
    parse_amount,
    reconcile,
    reconciliation_label,
    validate_gstin,
)


def test_parse_amount_handles_messy_ocr():
    assert parse_amount("₹ 1,250.50") == 1250.5
    assert parse_amount("780-") == 780.0
    assert parse_amount("2310/-") == 2310.0
    assert parse_amount(1200) == 1200.0
    assert parse_amount(None) is None
    assert parse_amount("abc") is None


def test_reconcile_ok_and_mismatch():
    parts = [{"amount": "780-"}, {"amount": "170-"}, {"amount": "320-"}, {"amount": "1200-"}]
    ok = reconcile("2470-", parts)
    assert ok["status"] == "ok" and ok["lines_sum"] == 2470.0 and ok["diff"] == 0.0
    bad = reconcile("2500-", parts)
    assert bad["status"] == "mismatch" and bad["diff"] == -30.0
    assert "MISMATCH" in reconciliation_label(bad)
    assert reconciliation_label(ok) == "OK"


def test_reconcile_unknown_without_total_or_amounts():
    assert reconcile(None, [{"amount": "10"}])["status"] == "unknown"
    assert reconcile("100", [{"item": "x"}])["status"] == "unknown"


def test_gstin_valid_example():
    # A well-known valid public GSTIN example.
    r = validate_gstin("27AAPFU0939F1ZV")
    assert r["status"] == "valid" and r["valid"] is True


def test_gstin_normalizes_case_and_space():
    assert validate_gstin("  27aapfu0939f1zv ")["valid"] is True


def test_gstin_checksum_roundtrip_and_flip():
    prefix = "27AAPFU0939F1Z"
    cd = _gstin_check_digit(prefix)
    assert validate_gstin(prefix + cd)["status"] == "valid"
    # Flipping the check digit must fail on checksum, not format.
    wrong = "A" if cd != "A" else "B"
    r = validate_gstin(prefix + wrong)
    assert r["valid"] is False and r["reason"] == "checksum"


def test_gstin_absent_and_malformed():
    assert validate_gstin("")["status"] == "absent"
    assert validate_gstin("GARBAGE")["reason"] == "format"
    # A garbled OCR GSTIN from a real scan flags invalid (needs review), not valid.
    assert validate_gstin("08AAKCRT631M21R")["valid"] is False
