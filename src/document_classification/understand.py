"""Turn OCR text into something meaningful: a summary + structured fields.

The good stuff (abstractive summary, receipt totals as clean JSON, "what is this
document about") comes from Google Gemini. When no ``GEMINI_API_KEY`` is set — or
the ``google-genai`` package isn't installed, or the call fails — we degrade
gracefully to the fully offline :mod:`extract` heuristics so the feature never
hard-fails on Streamlit Cloud.

Enable Gemini by installing the extra and setting a key::

    pip install '.[understand]'
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Model is ``gemini-2.5-flash`` by default; override with ``DOC_GEMINI_MODEL``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import extract as _extract

DEFAULT_MODEL = os.environ.get("DOC_GEMINI_MODEL", "gemini-3.6-flash")

# Cap what we send: OCR of a dense page is a few KB, but a 50-page PDF batch is not.
MAX_INPUT_CHARS = 12_000

_SYSTEM = (
    "You are a precise document-understanding assistant. You are given raw OCR text "
    "extracted from a scanned document, plus the document type detected by a vision "
    "model. OCR text may contain recognition errors — infer sensibly but never invent "
    "facts that are not supported by the text. Reply with ONLY a JSON object."
)

_SCHEMA_HINT = (
    'Return JSON exactly of this shape:\n'
    '{\n'
    '  "summary": "<2-4 sentence plain-language summary of what this document is and says>",\n'
    '  "key_points": ["<short bullet>", ...],   // 2-6 items, most important first\n'
    '  "fields": { <structured key/value pairs relevant to the document type> }\n'
    '}\n'
    "For a receipt or invoice, put total, currency, date, merchant, and item_count in "
    '"fields" when present. For an essay/letter/note, "fields" may be an empty object. '
    "Use null for anything you cannot determine. Do not wrap the JSON in markdown fences."
)


_ENV_LOADED = False


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a project ``.env`` into ``os.environ`` (once).

    Dependency-free so the app works on a bare install; existing environment
    variables always win over the file. Walks up from this file to find ``.env``.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.is_file():
            try:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)  # don't clobber real env vars
            except Exception:
                pass
            break


def _api_key() -> str | None:
    _load_dotenv()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def gemini_available() -> bool:
    """True when the ``google-genai`` package is importable and a key is configured."""
    if not _api_key():
        return False
    try:
        import google.genai  # noqa: F401

        return True
    except Exception:
        return False


def _parse_json(raw: str) -> dict | None:
    """Parse the model's reply, tolerating stray prose or ```json fences."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)  # first balanced-ish object
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _gemini_understanding(text: str, doc_type: str | None) -> dict | None:
    """Call Gemini for summary + fields. Returns None on any failure (caller falls back)."""
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None

    key = _api_key()
    if not key:
        return None

    prompt = (
        f"{_SCHEMA_HINT}\n\n"
        f"Detected document type: {doc_type or 'unknown'}\n\n"
        f"OCR text:\n\"\"\"\n{text[:MAX_INPUT_CHARS]}\n\"\"\""
    )
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        data = _parse_json(getattr(resp, "text", "") or "")
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    summary = (data.get("summary") or "").strip()
    if not summary:
        return None
    key_points = [str(p).strip() for p in (data.get("key_points") or []) if str(p).strip()]
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    # Drop nulls the model emitted for missing fields.
    fields = {k: v for k, v in fields.items() if v not in (None, "", [])}
    return {
        "engine": f"gemini:{DEFAULT_MODEL}",
        "summary": summary,
        "key_points": key_points,
        "fields": fields,
    }


def _offline_understanding(text: str, doc_type: str | None) -> dict:
    """Fully offline: extractive summary + deterministic field extraction."""
    fields = _extract.extract_fields(text, doc_type)
    summary = _extract.extractive_summary(text, max_sentences=3)
    if not summary and fields:
        summary = "No prose detected; extracted the structured fields below."
    return {
        "engine": "offline",
        "summary": summary,
        "key_points": [],
        "fields": fields,
    }


def understand(text: str, doc_type: str | None = None) -> dict:
    """Summarise and extract from OCR text.

    Uses Gemini when configured, otherwise the offline heuristics. Always merges in
    the deterministic :func:`extract.extract_fields` results so that reliable,
    regex-verifiable values (e.g. the numeric total) are present even when Gemini
    is used — the model's fields win on conflict, ours fill the gaps.
    """
    text = (text or "").strip()
    if not text:
        return {"engine": "none", "summary": "", "key_points": [], "fields": {}}

    result = _gemini_understanding(text, doc_type)
    if result is None:
        return _offline_understanding(text, doc_type)

    deterministic = _extract.extract_fields(text, doc_type)
    merged = {**deterministic, **result["fields"]}  # model wins, ours backfill
    result["fields"] = merged
    return result
