"""Taxonomy integrity: the derived lookups must stay consistent with TAXONOMY."""
from __future__ import annotations

from document_classification.classify import (
    CATEGORY_OF,
    DOCUMENT_TYPES,
    TAXONOMY,
)


def test_no_duplicate_leaf_labels():
    leaves = [label for leaves in TAXONOMY.values() for label in leaves]
    assert len(leaves) == len(set(leaves)), "leaf labels must be unique across categories"


def test_derived_maps_cover_every_leaf():
    leaves = {label for leaves in TAXONOMY.values() for label in leaves}
    assert set(DOCUMENT_TYPES) == leaves
    assert set(CATEGORY_OF) == leaves


def test_category_of_points_back_to_taxonomy():
    for label, category in CATEGORY_OF.items():
        assert label in TAXONOMY[category]


def test_every_leaf_has_at_least_one_prompt():
    for label, prompts in DOCUMENT_TYPES.items():
        assert prompts, f"{label} has no prompt phrasings"
        assert all(isinstance(p, str) and p for p in prompts)


def test_expected_shape():
    # 8 categories, and the 43-type taxonomy the app documents.
    assert len(TAXONOMY) == 8
    assert len(DOCUMENT_TYPES) == 43
