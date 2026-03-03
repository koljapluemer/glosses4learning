from __future__ import annotations

from typing import Any, Iterator

RELATIONSHIP_TYPES = {
    "MORPHOLOGICALLY_RELATED",
    "PART",
    "HAS_SIMILAR_MEANING",
    "SOUNDS_SIMILAR",
    "USAGE_EXAMPLE",
    "TO_BE_DIFFERENTIATED_FROM",
    "COLLOCATION",
    "TYPICAL_FOLLOW_UP",
    "CHILD",
    "TRANSLATION",
    "NOTE",
}

LEGACY_TO_RELATIONSHIP_TYPE = {
    "morphologically_related": "MORPHOLOGICALLY_RELATED",
    "parts": "PART",
    "has_similar_meaning": "HAS_SIMILAR_MEANING",
    "sounds_similar": "SOUNDS_SIMILAR",
    "usage_examples": "USAGE_EXAMPLE",
    "to_be_differentiated_from": "TO_BE_DIFFERENTIATED_FROM",
    "collocations": "COLLOCATION",
    "typical_follow_up": "TYPICAL_FOLLOW_UP",
    "children": "CHILD",
    "translations": "TRANSLATION",
    "notes": "NOTE",
}

LEGACY_RELATIONSHIP_FIELDS = tuple(LEGACY_TO_RELATIONSHIP_TYPE.keys())


def _richness(credit: str | None, note: str | None) -> int:
    if note is not None:
        return 4
    if credit is not None:
        return 3
    return 2


def normalize_relationship(entry: Any) -> tuple[str, str, str | None, str | None] | None:
    if not isinstance(entry, list):
        return None
    if len(entry) < 2 or len(entry) > 4:
        return None

    rel_type = entry[0]
    ref = entry[1]
    if not isinstance(rel_type, str) or rel_type not in RELATIONSHIP_TYPES:
        return None
    if not isinstance(ref, str) or ":" not in ref:
        return None

    credit = entry[2] if len(entry) >= 3 else None
    note = entry[3] if len(entry) >= 4 else None
    if credit is not None and not isinstance(credit, str):
        return None
    if note is not None and not isinstance(note, str):
        return None
    return (rel_type, ref, credit, note)


def relationship_to_list(
    rel_type: str, ref: str, credit: str | None = None, note: str | None = None
) -> list[str]:
    row: list[str] = [rel_type, ref]
    if credit is not None:
        row.append(credit)
    if note is not None:
        if credit is None:
            row.append("")
        row.append(note)
    return row


def ensure_relationships(gloss: dict[str, Any]) -> list[Any]:
    relationships = gloss.get("relationships")
    if not isinstance(relationships, list):
        relationships = []
        gloss["relationships"] = relationships
    return relationships


def dedupe_relationships(gloss: dict[str, Any]) -> list[list[str]]:
    relationships = ensure_relationships(gloss)

    # Keep first-seen ordering by (type, ref), but retain richest metadata variant.
    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], tuple[str | None, str | None]] = {}
    for row in relationships:
        parsed = normalize_relationship(row)
        if parsed is None:
            continue
        rel_type, ref, credit, note = parsed
        key = (rel_type, ref)
        if key not in by_key:
            order.append(key)
            by_key[key] = (credit, note)
            continue

        current_credit, current_note = by_key[key]
        current_score = _richness(current_credit, current_note)
        new_score = _richness(credit, note)
        if new_score > current_score:
            by_key[key] = (credit, note)

    deduped: list[list[str]] = []
    for rel_type, ref in order:
        credit, note = by_key[(rel_type, ref)]
        deduped.append(relationship_to_list(rel_type, ref, credit, note))

    gloss["relationships"] = deduped
    return deduped


def add_relationship(
    gloss: dict[str, Any],
    rel_type: str,
    ref: str,
    credit: str | None = None,
    note: str | None = None,
) -> None:
    relationships = ensure_relationships(gloss)
    relationships.append(relationship_to_list(rel_type, ref, credit, note))
    dedupe_relationships(gloss)


def iter_relationships(
    gloss: dict[str, Any],
    rel_type: str | None = None,
    lang: str | None = None,
) -> Iterator[tuple[str, str, str | None, str | None]]:
    relationships = gloss.get("relationships")
    if not isinstance(relationships, list):
        return

    for row in relationships:
        parsed = normalize_relationship(row)
        if parsed is None:
            continue
        row_type, ref, credit, note = parsed
        if rel_type is not None and row_type != rel_type:
            continue
        if lang is not None:
            if ":" not in ref:
                continue
            ref_lang = ref.split(":", 1)[0]
            if ref_lang != lang:
                continue
        yield (row_type, ref, credit, note)


def has_relationship(
    gloss: dict[str, Any],
    rel_type: str,
    ref_prefix: str | None = None,
) -> bool:
    for _, ref, _, _ in iter_relationships(gloss, rel_type=rel_type):
        if ref_prefix is None or ref.startswith(ref_prefix):
            return True
    return False


def get_relationship_refs(
    gloss: dict[str, Any],
    rel_type: str,
    lang: str | None = None,
) -> list[str]:
    refs: list[str] = []
    for _, ref, _, _ in iter_relationships(gloss, rel_type=rel_type, lang=lang):
        if ref not in refs:
            refs.append(ref)
    return refs


def replace_relationship_ref(gloss: dict[str, Any], old_ref: str, new_ref: str) -> int:
    relationships = gloss.get("relationships")
    if not isinstance(relationships, list):
        return 0

    replacements = 0
    new_rows: list[list[str]] = []
    for row in relationships:
        parsed = normalize_relationship(row)
        if parsed is None:
            continue
        rel_type, ref, credit, note = parsed
        if ref == old_ref:
            ref = new_ref
            replacements += 1
        new_rows.append(relationship_to_list(rel_type, ref, credit, note))

    gloss["relationships"] = new_rows
    if replacements:
        dedupe_relationships(gloss)
    return replacements


def merge_legacy_relationship_fields(
    gloss: dict[str, Any], remove_legacy: bool = False
) -> tuple[int, int]:
    """
    Merge legacy arrays into `relationships`.
    Returns (added_count_before_dedupe, invalid_ref_count).
    """
    ensure_relationships(gloss)
    relationships = gloss["relationships"]

    before_valid = len(list(iter_relationships(gloss)))
    invalid_refs = 0

    for legacy_field, rel_type in LEGACY_TO_RELATIONSHIP_TYPE.items():
        values = gloss.get(legacy_field)
        if not isinstance(values, list):
            if remove_legacy:
                gloss.pop(legacy_field, None)
            continue

        for value in values:
            if not isinstance(value, str) or ":" not in value:
                invalid_refs += 1
                continue
            relationships.append([rel_type, value])

        if remove_legacy:
            gloss.pop(legacy_field, None)

    dedupe_relationships(gloss)
    after_valid = len(list(iter_relationships(gloss)))
    return (max(0, after_valid - before_valid), invalid_refs)
