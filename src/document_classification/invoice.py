"""Extract structured invoice data from OCR regions and serialise it for ERP/Excel.

Flow (an item-procurement solution): Surya OCR turns the scanned invoice into a
JSON of text *regions* (each with its ``text`` and a ``bbox`` = ``[x0,y0,x1,y1]``
pixel box), and Gemini reads that **structured** JSON — not a flattened string —
so it can use the spatial layout to map the document: the left column is the
particulars/description, the right column is the amount, and regions sharing a
vertical band belong to the same line-item row.

One Gemini call both **parses and translates** (kindest to the free-tier quota),
but it is told to keep every identifier (invoice no, GSTIN, amounts, dates)
verbatim and to preserve each item's original text as ``item_raw`` alongside a
canonical English ``item`` name — so translation can never corrupt an identifier,
and a human can always check the English against the source. Each line also gets
a ``confidence`` so low-quality rows (handwritten regional text OCRs poorly) can
be flagged for review rather than silently trusted. An ``erp_code`` column is
emitted but left blank until a customer item-master is wired in.

Like :mod:`understand`, this leans on Gemini and degrades gracefully: with no
``GEMINI_API_KEY`` (or ``google-genai`` missing, or the call failing after
retries) :func:`extract_invoice` returns ``None`` and the caller omits the export.
"""
from __future__ import annotations

import io
import json
import re
import time

from .understand import DEFAULT_MODEL, MAX_INPUT_CHARS, _api_key, _parse_json

# Gemini's flash models intermittently return 503 UNAVAILABLE under load and, on
# the free tier, 429 RESOURCE_EXHAUSTED when the per-minute request cap is hit — a
# single miss would silently drop the whole invoice, so retry with backoff and
# honour the server's suggested wait when it gives one.
_RETRIES = 4
_BACKOFF = 3.0  # seconds, doubled each attempt
_MAX_WAIT = 30.0  # cap any single backoff/suggested wait

# Cap the regions/characters we hand the model so a dense multi-page batch stays
# affordable and within context.
MAX_REGIONS = 300

_RETRY_HINT = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)

_CONFIDENCE = {"high", "medium", "low"}

_SYSTEM = (
    "You are a precise invoice-parsing assistant for an item-procurement/ERP system. You "
    "are given the OCR output of a scanned invoice/bill as a JSON array of text REGIONS, "
    "each with its recognised text and a pixel bounding box box = [x0, y0, x1, y1]. Use the "
    "geometry: regions with a SMALL x are the left 'particulars'/description column; regions "
    "with a LARGE x are the right 'amount' column; regions that overlap in their y-range "
    "belong to the SAME line-item row, so pair a description with the amount beside it. Text "
    "may be in Hindi, Marwari, Gujarati, Kannada or other languages, mixed with English, and "
    "contains OCR errors (handwritten regional text is especially noisy — the numbers are far "
    "more reliable than the words). Keep every IDENTIFIER exactly as written: invoice number, "
    "date, GSTIN, phone, and all monetary amounts — never translate or 'correct' these. For "
    "each line item, preserve the ORIGINAL description text as item_raw and ALSO give a "
    "canonical English item name (translate the meaning, e.g. चश्मा -> \"Spectacles\", दाल -> "
    "\"Lentils\"; normalise spelling/number so variants map to one name — never transliterate). "
    "Rate each line's confidence by how legible the source was. Never invent values not "
    "supported by the text; use null when a field is absent. Reply with ONLY a JSON object."
)

_SCHEMA_HINT = (
    'Return JSON exactly of this shape:\n'
    '{\n'
    '  "invoice_no": "<invoice/bill number as written, or null>",\n'
    '  "date": "<invoice date as written, or null>",\n'
    '  "vendor_name": "<seller / letterhead business name in English, or null>",\n'
    '  "vendor_address": "<seller address in English, or null>",\n'
    '  "customer_name": "<the billed party (\'Name & Address\') in English, or null>",\n'
    '  "gstin": "<GST/tax id if present (~15-char alphanumeric), or null>",\n'
    '  "total_amount": "<grand total as written (digits, keep decimals), or null>",\n'
    '  "currency": "<currency symbol or code if shown, e.g. ₹, Rs, INR, or null>",\n'
    '  "particulars": [\n'
    '    {"item_raw": "<original description text, verbatim>",\n'
    '     "item": "<canonical English item name>",\n'
    '     "qty": <integer quantity>,\n'
    '     "amount": "<line amount as written, or null>",\n'
    '     "confidence": "high|medium|low"}\n'
    '  ]\n'
    '}\n'
    "Rules for particulars: one object per distinct line item, top to bottom; pair each "
    "description with the amount in the same horizontal band. If a line shows no quantity, "
    "use qty = 1 (whole number). Do NOT include the grand-total, amount-in-words or tax-"
    "summary rows as items. If a description is illegible, set item_raw to whatever text "
    "there is, item to \"(illegible)\", and confidence to \"low\". Do not wrap the JSON in "
    "markdown fences."
)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt: the server's hint if present
    (e.g. '... Please retry in 3.16s'), else exponential backoff. Capped."""
    m = _RETRY_HINT.search(str(exc))
    if m:
        try:
            return min(float(m.group(1)) + 0.5, _MAX_WAIT)
        except ValueError:
            pass
    return min(_BACKOFF * (2 ** attempt), _MAX_WAIT)


def _regions(ocr_json: dict) -> list[dict]:
    """Flatten an OCR-JSON (single-page ``blocks`` or multi-page ``pages``) to a
    list of ``{text, box}`` regions, dropping empties and tagging the page."""
    out: list[dict] = []
    if not isinstance(ocr_json, dict):
        return out

    def _add(blocks, page):
        for b in blocks or []:
            text = str(b.get("text") or "").strip()
            if not text:
                continue
            box = b.get("bbox")
            entry = {"text": text}
            if isinstance(box, (list, tuple)) and len(box) == 4:
                entry["box"] = [int(round(float(v))) for v in box]
            if page is not None:
                entry["page"] = page
            out.append(entry)

    if ocr_json.get("blocks"):
        _add(ocr_json["blocks"], None)
    else:
        for pi, page in enumerate(ocr_json.get("pages") or [], 1):
            _add(page.get("blocks"), pi)
    return out


def _gemini_invoice(regions: list[dict], doc_type: str | None) -> dict | None:
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None

    key = _api_key()
    if not key:
        return None

    numbered = [{"i": i, **r} for i, r in enumerate(regions[:MAX_REGIONS])]
    payload = json.dumps(numbered, ensure_ascii=False)[:MAX_INPUT_CHARS]
    prompt = (
        f"{_SCHEMA_HINT}\n\n"
        f"Detected document type: {doc_type or 'invoice'}\n\n"
        f"OCR regions (JSON):\n{payload}"
    )
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM,
        response_mime_type="application/json",
        temperature=0.1,
    )
    for attempt in range(_RETRIES):
        try:
            resp = client.models.generate_content(
                model=DEFAULT_MODEL, contents=prompt, config=config
            )
            data = _parse_json(getattr(resp, "text", "") or "")
            if isinstance(data, dict):
                return data
            return None  # got a reply, but not the expected shape — don't retry
        except Exception as exc:
            if attempt < _RETRIES - 1:
                time.sleep(_retry_delay(exc, attempt))
    return None  # exhausted retries (e.g. persistent 503/429)


def _coerce_qty(value) -> int:
    """A missing/blank quantity means one unit; parse leading digits otherwise."""
    if value is None or value == "":
        return 1
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return int(value) if value else 1
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 1


def _clean_particulars(raw) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = str(entry.get("item") or "").strip()
        item_raw = str(entry.get("item_raw") or "").strip()
        amount = "" if entry.get("amount") in (None, "") else str(entry.get("amount")).strip()
        if not item and not item_raw and not amount:
            continue
        conf = str(entry.get("confidence") or "").strip().lower()
        if conf not in _CONFIDENCE:
            conf = "low" if (item == "(illegible)" or not item) else "medium"
        out.append({
            "item_raw": item_raw,
            "item": item or "(illegible)",
            "qty": _coerce_qty(entry.get("qty")),
            "amount": amount,
            "erp_code": "",   # filled by the item-graph resolver when matched
            "category": "",   # procurement category from the graph hierarchy
            "confidence": conf,
        })
    return out


def extract_invoice(ocr_json: dict, doc_type: str | None = None) -> dict | None:
    """Extract structured invoice fields from a structured OCR JSON, or ``None``.

    ``ocr_json`` is the ``{engine, blocks}`` (single page) or ``{engine, pages}``
    (multi-page) produced by :func:`classify.ocr_extract`. Returns invoice-level
    fields plus ``particulars: [{item_raw, item, qty, amount, erp_code,
    confidence}]``. Item names are translated to canonical English while the
    source text is kept as ``item_raw``; identifiers are preserved verbatim. A
    line with no quantity gets ``qty = 1``. ``None`` is returned when there are no
    regions or Gemini is unavailable (no key / package / persistent error).
    """
    regions = _regions(ocr_json)
    if not regions:
        return None
    data = _gemini_invoice(regions, doc_type)
    if data is None:
        return None

    particulars = _clean_particulars(data.get("particulars"))
    # Resolve each line against the item knowledge graph (canonical name + ERP
    # code + category); a weak match is left as the model's English for review.
    from .item_graph import normalize_particulars

    particulars = normalize_particulars(particulars)

    def _field(key: str) -> str:
        val = data.get(key)
        return "" if val in (None, "") else str(val).strip()

    result = {
        "invoice_no": _field("invoice_no"),
        "date": _field("date"),
        "vendor_name": _field("vendor_name"),
        "vendor_address": _field("vendor_address"),
        "customer_name": _field("customer_name"),
        "gstin": _field("gstin"),
        "total_amount": _field("total_amount"),
        "currency": _field("currency"),
        "particulars": particulars,
    }
    # Nothing usable extracted at all — treat as a miss so the UI can say so.
    scalar_keys = ("invoice_no", "date", "vendor_name", "customer_name", "total_amount")
    if not any(result[k] for k in scalar_keys) and not particulars:
        return None
    # Cheap, no-model validation: do the lines add up, and is the GSTIN checksum ok?
    from .validate import validate_invoice

    result["validation"] = validate_invoice(result)
    return result


# Flat, one-row-per-line-item schema for ERP ingestion. Invoice-level metadata is
# repeated on every row; erp_code stays blank until an item-master is wired in.
_COLUMNS = [
    "Invoice No", "Date", "Vendor", "Vendor Address", "Customer", "GSTIN",
    "Item (raw)", "Item (English)", "ERP Code", "Category", "Qty", "Amount", "Total",
    "Confidence", "Total Check", "GSTIN Check",
]


def _rows(invoices: list[tuple[str, dict]]) -> list[list]:
    from .validate import reconciliation_label

    rows: list[list] = []
    for label, inv in invoices:
        inv = inv or {}
        currency = inv.get("currency") or ""
        total = inv.get("total_amount") or ""
        if total and currency:
            total = f"{currency} {total}".strip()
        validation = inv.get("validation") or {}
        total_check = reconciliation_label(validation.get("reconciliation"))
        gstin_status = (validation.get("gstin") or {}).get("status") or ""
        gstin_check = "" if gstin_status == "absent" else gstin_status  # valid/invalid
        base = [
            inv.get("invoice_no") or label or "",
            inv.get("date") or "",
            inv.get("vendor_name") or "",
            inv.get("vendor_address") or "",
            inv.get("customer_name") or "",
            inv.get("gstin") or "",
        ]
        tail = [total_check, gstin_check]
        particulars = inv.get("particulars") or []
        if not particulars:
            rows.append(base + ["", "", "", "", "", "", total, ""] + tail)
            continue
        for p in particulars:
            amount = p.get("amount") or ""
            if amount and currency:
                amount = f"{currency} {amount}".strip()
            rows.append(base + [
                p.get("item_raw", ""),
                p.get("item", ""),
                p.get("erp_code", ""),
                p.get("category", ""),
                p.get("qty", 1),
                amount,
                total,
                p.get("confidence", ""),
            ] + tail)
    return rows


def invoice_workbook(invoices: list[tuple[str, dict]]) -> bytes:
    """Build an ERP-ready .xlsx (bytes): one flat sheet, **one row per line item**,
    with invoice-level metadata repeated across each invoice's rows.

    ``invoices`` is a list of ``(label, invoice_dict)`` where ``invoice_dict`` is
    an :func:`extract_invoice` result. Low-confidence rows keep their flag in the
    Confidence column so they can be filtered for review.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Line items"

    bold = Font(bold=True)
    top = Alignment(vertical="top", wrap_text=True)
    for col, head in enumerate(_COLUMNS, start=1):
        c = ws.cell(row=1, column=col, value=head)
        c.font = bold

    for r, row in enumerate(_rows(invoices), start=2):
        for col, val in enumerate(row, start=1):
            ws.cell(row=r, column=col, value=val).alignment = top

    widths = [12, 12, 22, 26, 24, 18, 26, 22, 12, 14, 6, 12, 12, 11, 14, 11]
    for i, w in enumerate(widths):
        ws.column_dimensions[ws.cell(row=1, column=i + 1).column_letter].width = w
    ws.freeze_panes = "A2"
    if ws.max_row >= 1:
        ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(_COLUMNS)).column_letter}{ws.max_row}"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
