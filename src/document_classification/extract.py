"""Deterministic, offline structured extraction from OCR text.

No API key, no network, no model download — pure regex + layout heuristics on the
case-preserving OCR text produced by :func:`classify.ocr_raw_text`. This is both a
standalone feature (a receipt upload shows its total and date immediately) and the
graceful fallback for :mod:`understand` when Gemini is unavailable.

The functions are intentionally conservative: they return only what they can find
with high confidence and leave a field absent rather than guessing.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
# Money like $1,234.56 / 1.234,56 € / £9.99 / 12.00. Capture the numeric part so
# we can compare magnitudes; keep it deliberately strict to avoid phone numbers.
_CURRENCY = r"(?:[$£€]|USD|EUR|GBP|INR|Rs\.?|₹)"
_AMOUNT_RE = re.compile(
    rf"(?P<cur>{_CURRENCY})?\s*(?P<num>\d{{1,3}}(?:[.,]\d{{3}})*(?:[.,]\d{{2}})|\d+\.\d{{2}})\s*(?P<cur2>{_CURRENCY})?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"                                  # 2026-08-29
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"                   # 29/08/2026
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"                 # 29 August 2026
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}"               # August 29, 2026
    r")\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_INVOICE_RE = re.compile(
    # \b stops "inv" from splitting the word "invoice" itself; the lookahead
    # requires the captured id to contain a digit (so "No" / "oice" don't match).
    r"\b(?:invoice|inv|bill|order|ref(?:erence)?)\b[\s:#.\-]*(?:no\.?|number)?[\s:#.\-]*"
    r"([A-Za-z0-9](?=[A-Za-z0-9/-]*\d)[A-Za-z0-9/-]{2,})",
    re.IGNORECASE,
)

# Line labels, most-specific first — "total due" beats "subtotal" beats "total".
_TOTAL_LABELS = (
    "grand total", "total due", "amount due", "balance due", "total amount",
    "total", "amount payable", "amount paid", "to pay",
)


def _to_float(num: str) -> float | None:
    """Parse a captured numeric string ('1,234.56' / '1.234,56') into a float."""
    s = num.strip()
    if "," in s and "." in s:
        # Whichever separator is last is the decimal separator.
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        # Comma as decimal only when it looks like ,dd at the end.
        s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _amounts(text: str) -> list[float]:
    out: list[float] = []
    for m in _AMOUNT_RE.finditer(text):
        val = _to_float(m.group("num"))
        if val is not None:
            out.append(val)
    return out


def find_total(text: str) -> float | None:
    """Best guess at the payable total: the amount on the most specific total row.

    Falls back to the largest amount on the page when no labelled row is found —
    on a receipt the total is almost always the biggest number.
    """
    for label in _TOTAL_LABELS:
        for line in text.splitlines():
            if label in line.lower():
                vals = _amounts(line)
                if vals:
                    return max(vals)  # a "total" line rarely has two competing sums
    all_vals = _amounts(text)
    return max(all_vals) if all_vals else None


def find_dates(text: str) -> list[str]:
    seen: list[str] = []
    for m in _DATE_RE.finditer(text):
        d = m.group(1)
        if d not in seen:
            seen.append(d)
    return seen


def _first(pattern: re.Pattern, text: str, group: int = 0) -> str | None:
    m = pattern.search(text)
    return m.group(group) if m else None


def extract_fields(text: str, doc_type: str | None = None) -> dict:
    """Extract structured fields from OCR text, tuned by document type.

    Returns a dict with whatever could be found confidently. Common keys:
    ``total``, ``currency``, ``dates``, ``emails``, ``phone``, ``invoice_number``,
    ``line_count``. Absent keys mean "not found", never "zero".
    """
    text = text or ""
    kind = (doc_type or "").lower()
    fields: dict = {}

    dates = find_dates(text)
    if dates:
        fields["dates"] = dates

    emails = list(dict.fromkeys(_EMAIL_RE.findall(text)))
    if emails:
        fields["emails"] = emails

    money_like = any(c in kind for c in ("receipt", "invoice", "bill", "statement", "cheque")) or bool(
        re.search(_CURRENCY, text)
    )
    if money_like:
        total = find_total(text)
        if total is not None:
            fields["total"] = round(total, 2)
        cur = _first(re.compile(_CURRENCY, re.IGNORECASE), text)
        if cur:
            fields["currency"] = cur.upper() if cur.isalpha() else cur
        inv = _first(_INVOICE_RE, text, group=1)
        if inv:
            fields["invoice_number"] = inv
        # Rough line-item count: non-empty lines that carry an amount.
        item_lines = [ln for ln in text.splitlines() if _amounts(ln)]
        if item_lines:
            fields["line_count"] = len(item_lines)

    if "id" in kind or "passport" in kind or "licen" in kind:
        phone = _first(_PHONE_RE, text)
        if phone:
            fields["phone"] = phone.strip()

    return fields


# --------------------------------------------------------------------------- #
# Offline extractive summary (dependency-free fallback for understand.py)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_STOP = frozenset(
    "the a an and or but of to in on at for with is are was were be been being this that "
    "these those it its as by from you your we our they their he she his her i me my".split()
)


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """A frequency-scored extractive summary: pick the most representative sentences.

    Cheap TextRank-style baseline (no dependencies): score each sentence by the
    summed frequency of its non-stopword terms, then return the top sentences in
    their original order. Good enough to be meaningful; not abstractive.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    freq: dict[str, int] = {}
    for w in _WORD_RE.findall(text.lower()):
        if w not in _STOP and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return " ".join(sentences[:max_sentences])

    def score(sent: str) -> float:
        words = [w for w in _WORD_RE.findall(sent.lower()) if w in freq]
        return sum(freq[w] for w in words) / (len(words) + 1)  # length-normalised

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    keep = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in keep)
