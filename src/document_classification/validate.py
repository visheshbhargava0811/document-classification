"""Cheap, high-value validation of an extracted invoice: catch errors for free.

Two checks that need no extra data and no model call:

* **Amount reconciliation** — do the line-item amounts add up to the stated
  grand total? Amounts survive OCR far better than handwritten item *names*, so
  this reliably flags a dropped/misread line or a hallucinated one.
* **GSTIN checksum** — the 15th character of a GSTIN is a mod-36 check digit over
  the first 14, so a single OCR error in the tax id is detectable. On handwritten
  scans most GSTINs are garbled and will (correctly) flag as invalid — that's the
  signal: "verify this id", not "trust it".

:func:`validate_invoice` returns a small dict the caller attaches to the invoice;
the Excel/app surface it as ``Total Check`` / ``GSTIN Check``.
"""
from __future__ import annotations

import re

# Currency noise to strip before reading a number: symbols, Rs/INR words, commas,
# and the trailing "-" / "/-" used on Indian handwritten bills.
_CURRENCY = re.compile(r"₹|rs\.?|inr|\$", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GST_MOD = 36
# 2 state digits, 10-char PAN, 1 entity char, 'Z', 1 checksum char.
_GST_FORMAT = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")


def parse_amount(value) -> float | None:
    """Read a monetary value from messy OCR text (``'₹ 1,250.50'``, ``'780-'``,
    ``'2310/-'``) → float, or ``None`` if there's no number."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = _CURRENCY.sub("", str(value)).replace(",", "")
    m = _NUMBER.search(s)
    return float(m.group()) if m else None


def reconcile(total_amount, particulars: list[dict], *, rel_tol: float = 0.005) -> dict:
    """Compare the sum of line amounts to the grand total.

    Returns ``{status, total, lines_sum, diff, n_lines, n_amounts}`` where status
    is ``ok`` (they agree within tolerance), ``mismatch`` (they don't), or
    ``unknown`` (no total or no parseable line amounts). Tolerance is 1 unit or
    ``rel_tol`` of the total, whichever is larger, to allow rounding.
    """
    total = parse_amount(total_amount)
    parsed = [parse_amount(p.get("amount")) for p in (particulars or [])]
    amounts = [a for a in parsed if a is not None]
    result = {"status": "unknown", "total": total, "lines_sum": None,
              "diff": None, "n_lines": len(parsed), "n_amounts": len(amounts)}
    if total is None or not amounts:
        return result
    lines_sum = round(sum(amounts), 2)
    diff = round(lines_sum - total, 2)
    tol = max(1.0, rel_tol * abs(total))
    result.update(lines_sum=lines_sum, diff=diff,
                  status="ok" if abs(diff) <= tol else "mismatch")
    return result


def _gstin_check_digit(first14: str) -> str | None:
    """GSTN's mod-36 check digit over the first 14 chars, or None if a char is
    outside 0-9/A-Z."""
    factor = 2
    total = 0
    for ch in reversed(first14):
        cp = _GST_CHARS.find(ch)
        if cp < 0:
            return None
        addend = factor * cp
        factor = 1 if factor == 2 else 2
        addend = (addend // _GST_MOD) + (addend % _GST_MOD)
        total += addend
    return _GST_CHARS[(_GST_MOD - (total % _GST_MOD)) % _GST_MOD]


def validate_gstin(gstin) -> dict:
    """Check a GSTIN's format and mod-36 checksum.

    Returns ``{status, valid, reason, normalized}`` with status ``valid`` /
    ``invalid`` / ``absent``. A garbled OCR GSTIN flags ``invalid`` (reason
    ``format`` or ``checksum``) — treat that as "verify", not "reject the row".
    """
    if not gstin or not str(gstin).strip():
        return {"status": "absent", "valid": None, "reason": "absent", "normalized": ""}
    norm = re.sub(r"\s+", "", str(gstin)).upper()
    if not _GST_FORMAT.match(norm):
        return {"status": "invalid", "valid": False, "reason": "format", "normalized": norm}
    expected = _gstin_check_digit(norm[:14])
    if expected is not None and expected == norm[14]:
        return {"status": "valid", "valid": True, "reason": "ok", "normalized": norm}
    return {"status": "invalid", "valid": False, "reason": "checksum", "normalized": norm}


def validate_invoice(invoice: dict) -> dict:
    """Bundle the checks for one extracted invoice."""
    return {
        "reconciliation": reconcile(invoice.get("total_amount"), invoice.get("particulars")),
        "gstin": validate_gstin(invoice.get("gstin")),
    }


def reconciliation_label(rec: dict) -> str:
    """Short human/Excel label for a reconciliation result."""
    status = (rec or {}).get("status")
    if status == "ok":
        return "OK"
    if status == "mismatch":
        diff = rec.get("diff")
        return f"MISMATCH Δ{diff:g}" if diff is not None else "MISMATCH"
    return ""
