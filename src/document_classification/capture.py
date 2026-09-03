"""Capture human review corrections so every fix compounds.

On a handwritten pipeline a person will always confirm the flagged rows — so
make that unavoidable review *productive*. When a reviewer corrects a line item,
we do two things:

1. **Teach the knowledge graph** — record the raw OCR text as a new alias of the
   corrected item (``मेरे शैन्य → Chair``), or create a new item node if it's not
   in the graph yet. Learned aliases live in ``data/item_graph_learned.json``
   (kept separate from the curated seed so provenance is clear) and take effect
   on the *next* extraction. The same garbled text auto-resolves next time.

2. **Log a labelled row** to ``data/review_labels.csv`` — growing ground truth
   for measuring accuracy and, later, fine-tuning.

Region-level *image* crops for training the recognizer come from
``training/export_line_crops.py``; this module captures the field-level text
corrections + graph learning.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from . import item_graph as ig

_DATA = Path(__file__).resolve().parents[2] / "data"
LEARNED_PATH = _DATA / "item_graph_learned.json"
LABELS_PATH = _DATA / "review_labels.csv"

_LABEL_COLUMNS = [
    "timestamp", "source", "invoice_no", "item_raw", "item_corrected",
    "qty", "amount", "erp_code", "category", "confidence",
]


def _as_int(value, default: int = 1) -> int:
    """Coerce an editor cell (float/str/NaN) to a positive integer quantity."""
    try:
        n = int(float(value))
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s[:32] or "item"


def _load_learned() -> dict:
    if LEARNED_PATH.is_file():
        try:
            data = json.loads(LEARNED_PATH.read_text(encoding="utf-8"))
            data.setdefault("aliases", [])
            data.setdefault("new_items", [])
            return data
        except Exception:
            pass
    return {"aliases": [], "new_items": []}


def _save_learned(data: dict) -> None:
    LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    ig.load_graph.cache_clear()  # so the new aliases resolve on the next extraction


def learn_from_corrections(original: list[dict], corrected: list[dict]) -> tuple[int, int]:
    """Teach the graph from a reviewer's corrections.

    Returns ``(n_aliases_added, n_new_items)``. For each corrected line with a raw
    OCR string and a real item name, add the raw as an alias of the matching entity
    (by canonical name), or register a new learned item when the name is unknown.
    Idempotent — re-saving the same corrections adds nothing new.
    """
    g = ig.load_graph()
    name_to_id = {ig._norm(node.get("name", "")): iid for iid, node in g.items.items()}
    learned = _load_learned()
    have_alias = {(a.get("raw_norm"), a.get("entity_id")) for a in learned["aliases"]}
    new_by_id = {ni["id"]: ni for ni in learned["new_items"]}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    n_alias = n_new = 0

    for corr in corrected:
        raw = str(corr.get("item_raw") or "").strip()
        name = str(corr.get("item") or "").strip()
        if not raw or not name or name == "(illegible)":
            continue
        rn, nn = ig._norm(raw), ig._norm(name)
        if not rn or not nn:
            continue
        eid = name_to_id.get(nn)
        if eid:
            # Known item — teach the raw text as an alias (unless already known).
            if rn in g.alias_index or (rn, eid) in have_alias:
                continue
            learned["aliases"].append({"raw": raw, "raw_norm": rn, "entity_id": eid,
                                       "canonical": g.items[eid].get("name", name), "ts": now})
            have_alias.add((rn, eid))
            n_alias += 1
        else:
            # Unknown item — register a new learned entity (or extend one).
            iid = "LEARN_" + _slug(name)
            if iid not in new_by_id:
                node = {"id": iid, "name": name, "erp": "", "category": "", "aliases": [raw]}
                new_by_id[iid] = node
                learned["new_items"].append(node)
                name_to_id[nn] = iid
                n_new += 1
            elif raw not in new_by_id[iid]["aliases"]:
                new_by_id[iid]["aliases"].append(raw)
                n_alias += 1

    _save_learned(learned)
    return n_alias, n_new


def log_labels(source: str, invoice: dict) -> int:
    """Append the (corrected) line items to the ground-truth label log. Returns the
    number of rows written."""
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS_PATH.is_file()
    inv_no = invoice.get("invoice_no", "")
    rows = invoice.get("particulars") or []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LABELS_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(_LABEL_COLUMNS)
        for p in rows:
            w.writerow([now, source, inv_no, p.get("item_raw", ""), p.get("item", ""),
                        p.get("qty", 1), p.get("amount", ""), p.get("erp_code", ""),
                        p.get("category", ""), p.get("confidence", "")])
    return len(rows)
