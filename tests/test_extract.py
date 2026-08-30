"""Tests for the offline, deterministic extraction heuristics (no API key needed)."""
from document_classification import extract as E

RECEIPT = """WHOLE FOODS MARKET
Date: 08/14/2026
Bananas        $3.49
Almond Milk    $4.29
Coffee Beans   $12.99
SUBTOTAL       $20.77
TAX            $1.66
TOTAL DUE      $22.43
Visa ****1234
"""


def test_to_float_handles_separators():
    assert E._to_float("1,234.56") == 1234.56
    assert E._to_float("1.234,56") == 1234.56
    assert E._to_float("22.43") == 22.43
    assert E._to_float("1,000") == 1000.0


def test_find_total_prefers_labelled_total_row():
    # Must pick the "TOTAL DUE" amount, not the larger-looking subtotal math.
    assert E.find_total(RECEIPT) == 22.43


def test_find_total_falls_back_to_largest_amount():
    text = "Coffee $3.00\nCake $5.50\nCash $10.00"
    assert E.find_total(text) == 10.00


def test_find_dates():
    assert "08/14/2026" in E.find_dates(RECEIPT)
    assert E.find_dates("Signed on 5 March 2025 by hand") == ["5 March 2025"]


def test_extract_fields_receipt():
    fields = E.extract_fields(RECEIPT, "Receipt")
    assert fields["total"] == 22.43
    assert fields["currency"] == "$"
    assert fields["dates"] == ["08/14/2026"]
    assert fields["line_count"] >= 3


def test_extract_fields_invoice_number_and_email():
    text = "Invoice No: INV-2026-0042\nQuestions? billing@acme.com\nAmount Due $500.00"
    fields = E.extract_fields(text, "Invoice")
    assert fields["invoice_number"].startswith("INV-2026")
    assert "billing@acme.com" in fields["emails"]
    assert fields["total"] == 500.00


def test_extractive_summary_picks_sentences():
    text = (
        "Green spaces reduce stress in cities. Green spaces improve air quality. "
        "The weather today is mild. Funding for green spaces remains an obstacle. "
        "Planners value green infrastructure highly."
    )
    summary = E.extractive_summary(text, max_sentences=2)
    assert summary
    # The most representative sentences mention the dominant term "green spaces".
    assert "green" in summary.lower()
    assert len(summary) < len(text)


def test_extractive_summary_short_text_returned_whole():
    text = "One sentence only."
    assert E.extractive_summary(text) == "One sentence only."
