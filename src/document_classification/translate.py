"""Translate the structured OCR JSON to English, block-for-block, via Gemini.

Surya (or Tesseract) produces one machine-readable JSON: a list of detected
regions, each with ``text`` in whatever language the document is written in
(Hindi, Marwari, Kannada, mixed English, …). This module produces a *second*
JSON of the same shape where every region's ``text`` has been translated to
English **inline** — the block's bbox, confidence, label and reading order are
preserved, so the two JSONs line up region-for-region. The pre-translation
string is kept on each block as ``text_original`` for verification.

Like :mod:`understand`, this leans on Google Gemini and degrades gracefully:
when no ``GEMINI_API_KEY`` is set (or ``google-genai`` isn't installed, or the
call fails) it returns ``None`` and the caller simply omits the translated JSON.
"""
from __future__ import annotations

import json

# Reuse understand.py's env/key loading, JSON parsing and model selection so
# translation is configured exactly like the rest of the Gemini features.
from .understand import DEFAULT_MODEL, _api_key, _parse_json

# Keep a single request affordable and within the model's context: a dense
# receipt is a few dozen regions, but a 50-page PDF batch must be capped.
MAX_BLOCKS = 400
MAX_INPUT_CHARS = 24_000

_SYSTEM = (
    "You are a precise translator for OCR output. You receive a JSON array of text "
    "segments extracted region-by-region from a scanned document. Segments may be in "
    "any language (e.g. Hindi, Marwari, Kannada) and may mix scripts with English, "
    "numbers and symbols. Translate each segment to natural English. Rules: keep "
    "numbers, currency amounts, dates, phone numbers, codes and proper-noun brand "
    "names exactly as written; if a segment is already English or is purely numeric/"
    "symbolic, return it unchanged; never merge, split, drop, add or reorder segments. "
    "OCR text may contain recognition errors — translate sensibly but do not invent "
    "content. Reply with ONLY a JSON object."
)

_SCHEMA_HINT = (
    'Return JSON exactly of this shape:\n'
    '{"segments": [{"i": <int index from the input>, '
    '"en": "<English translation of that segment>", '
    '"lang": "<BCP-47 code of the segment\'s source language, e.g. hi, kn, en>"}, ...]}\n'
    "Include exactly one object per input segment, keeping the same i values. "
    "Do not wrap the JSON in markdown fences."
)


def _texts(blocks: list[dict]) -> list[str]:
    return [str(b.get("text") or "") for b in blocks]


def _gemini_translate(texts: list[str]) -> tuple[dict[int, str], dict[int, str]] | None:
    """Translate a list of segments. Returns ({i: english}, {i: lang}) or None on failure."""
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None

    key = _api_key()
    if not key:
        return None

    # Index only the non-empty segments; empty ones stay empty without a round trip.
    indexed = [{"i": i, "text": t} for i, t in enumerate(texts) if t.strip()]
    if not indexed:
        return {}, {}

    payload = json.dumps(indexed, ensure_ascii=False)[:MAX_INPUT_CHARS]
    prompt = f"{_SCHEMA_HINT}\n\nSegments:\n{payload}"
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        data = _parse_json(getattr(resp, "text", "") or "")
    except Exception:
        return None

    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None

    english: dict[int, str] = {}
    langs: dict[int, str] = {}
    for seg in data["segments"]:
        if not isinstance(seg, dict):
            continue
        try:
            i = int(seg["i"])
        except (KeyError, TypeError, ValueError):
            continue
        en = seg.get("en")
        if isinstance(en, str):
            english[i] = en
        lang = seg.get("lang")
        if isinstance(lang, str) and lang.strip():
            langs[i] = lang.strip().lower()
    if not english:
        return None
    return english, langs


def translate_ocr_json(ocr_json: dict) -> dict | None:
    """Return an English-translated copy of a structured OCR JSON, or ``None``.

    Input is the ``{"engine": str, "blocks": [...]}`` produced by
    :func:`classify.ocr_extract`. The output has the same shape and the same
    blocks in the same order, with each block's ``text`` replaced by its English
    translation and the source string preserved as ``text_original``. A ``lang``
    code is added per block when the model reported one. ``None`` is returned when
    there is nothing to translate or Gemini is unavailable — the caller then just
    omits the translated JSON.
    """
    if not isinstance(ocr_json, dict):
        return None
    blocks = ocr_json.get("blocks") or []
    if not blocks:
        return None

    texts = _texts(blocks)[:MAX_BLOCKS]
    result = _gemini_translate(texts)
    if result is None:
        return None
    english, langs = result

    out_blocks: list[dict] = []
    any_translated = False
    for i, block in enumerate(blocks):
        new = dict(block)  # preserve bbox / confidence / label / reading_order
        original = str(block.get("text") or "")
        translated = english.get(i, original)
        if translated != original:
            new["text_original"] = original
            any_translated = True
        new["text"] = translated
        if i in langs:
            new["lang"] = langs[i]
        out_blocks.append(new)

    if not any_translated:
        # Everything was already English (or nothing came back) — no second JSON needed.
        return None

    return {
        "engine": f"{ocr_json.get('engine', 'ocr')}+gemini-translate:{DEFAULT_MODEL}",
        "target_language": "en",
        "blocks": out_blocks,
    }
