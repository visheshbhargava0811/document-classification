"""Cross-lingual item **knowledge graph**: resolve any surface form to a canonical entity.

Items live as ENTITY NODES in a small property graph (``data/item_graph.json``),
not as a flat alias list. Each item has a canonical English name, an ERP code, and
many aliases across scripts/languages — ``कुर्सी`` / ``kursi`` / ``chair`` all
resolve to the entity ``Chair``. Items connect to CATEGORY nodes and categories to
parent categories (``is_a`` edges), so a line item rolls up to a procurement
category: Chair → Seating → Furniture. That hierarchy — plus a place to hang
typed relations later (``synonym_of``, ``substitute_for``, ``part_of`` in
``edges``) — is what a flat lexicon can't give.

Resolution: an exact/normalized alias hit first, then a fuzzy fallback (stdlib
:mod:`difflib`, so no new dependency; swap in ``rapidfuzz`` or embeddings later).
Below the threshold we keep the model's English and flag the row for review — a
wrong ERP code is worse than a blank one. The graph is a plain JSON file:
human-editable and version-controlled; upgrade to networkx / Neo4j / RDF when you
outgrow it.

Override the graph path with ``$DOC_ITEM_GRAPH``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

_DEFAULT_GRAPH = Path(__file__).resolve().parents[2] / "data" / "item_graph.json"

MATCH_THRESHOLD = 0.72   # below this we keep the model's own English
HIGH_CONF = 0.88         # a match this strong is high-confidence (if OCR was legible)


@dataclass
class Graph:
    items: dict          # id -> {name, erp, category, aliases}
    categories: dict     # id -> {name, parent}
    alias_index: dict    # normalized surface -> item id
    edges: list          # [{from, rel, to}]

    def category_path(self, item_id: str) -> list[str]:
        """Category names from the item's immediate category up to the root."""
        node = self.items.get(item_id)
        cat = node and node.get("category")
        path: list[str] = []
        seen: set[str] = set()
        while cat and cat in self.categories and cat not in seen:
            seen.add(cat)
            path.append(self.categories[cat].get("name", cat))
            cat = self.categories[cat].get("parent")
        return path


def _norm(s: str) -> str:
    """Lowercase, drop digits/punctuation (keep letters incl. Devanagari), squeeze spaces."""
    s = (s or "").lower()
    s = re.sub(r"[0-9]+", " ", s)
    s = re.sub(r"[^\wऀ-ॿ]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=8)
def load_graph(path: str | None = None) -> Graph:
    """Load and index the item graph (cached). Missing/broken file → empty graph
    (normalization then becomes a graceful no-op)."""
    p = Path(path or os.environ.get("DOC_ITEM_GRAPH") or _DEFAULT_GRAPH)
    if not p.is_file():
        return Graph({}, {}, {}, [])
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return Graph({}, {}, {}, [])
    items = {k: v for k, v in (raw.get("items") or {}).items() if isinstance(v, dict)}
    categories = {k: v for k, v in (raw.get("categories") or {}).items() if isinstance(v, dict)}
    index: dict[str, str] = {}
    for iid, node in items.items():
        surfaces = [node.get("name", ""), *(node.get("aliases") or [])]
        for s in surfaces:
            key = _norm(s)
            if key:
                index.setdefault(key, iid)  # first wins on collision
    return Graph(items, categories, index, list(raw.get("edges") or []))


def _score(query: str, candidate: str) -> float:
    """Similarity of two normalized strings, boosted when one contains the other."""
    q, c = _norm(query), _norm(candidate)
    if not q or not c:
        return 0.0
    ratio = SequenceMatcher(None, q, c).ratio()
    if len(c) >= 3 and len(q) >= 3 and (c in q or q in c):
        ratio = max(ratio, 0.9)
    return ratio


def resolve(raw: str, english: str, graph: Graph | None = None,
            threshold: float = MATCH_THRESHOLD) -> dict:
    """Resolve one line item to a graph entity. Returns ``{entity_id, canonical,
    erp_code, category, score, matched}``. Tries an exact normalized-alias hit on
    either the English guess or the raw text, then a fuzzy fallback. ``matched`` is
    False (blank canonical/erp) below ``threshold`` — the caller keeps the model's
    English then."""
    g = load_graph() if graph is None else graph
    blank = {"entity_id": "", "canonical": "", "erp_code": "", "category": "",
             "score": 0.0, "matched": False}
    if not g.items:
        return blank

    # 1) exact / normalized alias hit
    for q in (english, raw):
        iid = g.alias_index.get(_norm(q))
        if iid:
            node = g.items[iid]
            path = g.category_path(iid)
            return {"entity_id": iid, "canonical": node.get("name", ""),
                    "erp_code": node.get("erp", ""), "category": path[0] if path else "",
                    "score": 1.0, "matched": True}

    # 2) fuzzy fallback over every item's surfaces
    best_id, best_score = "", 0.0
    for iid, node in g.items.items():
        surfaces = [node.get("name", ""), *(node.get("aliases") or [])]
        s = max((max(_score(q, surf) for surf in surfaces)
                 for q in (english, raw) if q), default=0.0)
        if s > best_score:
            best_score, best_id = s, iid
    if best_id and best_score >= threshold:
        node = g.items[best_id]
        path = g.category_path(best_id)
        return {"entity_id": best_id, "canonical": node.get("name", ""),
                "erp_code": node.get("erp", ""), "category": path[0] if path else "",
                "score": round(best_score, 3), "matched": True}
    return {**blank, "score": round(best_score, 3)}


def normalize_particulars(particulars: list[dict]) -> list[dict]:
    """Resolve each line item against the graph (returns new dicts).

    A confident match replaces ``item`` with the canonical name and fills
    ``erp_code`` + ``category`` (+ ``entity_id``, ``match_score``). Confidence
    reflects OCR legibility AND match strength: a strong match on *illegible*
    handwriting is still a guess, so a ``low`` source is never promoted past
    ``medium``. A weak match keeps the model's English and stays flagged.
    """
    g = load_graph()
    if not g.items:
        return particulars
    out: list[dict] = []
    for p in particulars:
        source_conf = p.get("confidence")
        m = resolve(p.get("item_raw", ""), p.get("item", ""), g)
        q = dict(p)
        q["match_score"] = m["score"]
        if m["matched"]:
            q["item"] = m["canonical"]
            q["erp_code"] = m["erp_code"]
            q["category"] = m["category"]
            q["entity_id"] = m["entity_id"]
            if source_conf == "low":
                q["confidence"] = "medium"
            elif m["score"] >= HIGH_CONF:
                q["confidence"] = "high"
        out.append(q)
    return out
