from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def canonicalize_text_goal(text: str) -> str | None:
    """Return the target-category phrase of a goal text, or None."""
    raw = str(text or "")
    match = re.search(r"target\s+category\s*:\s*([A-Za-z0-9 _-]+)", raw, flags=re.IGNORECASE)
    if match:
        return normalize_phrase(match.group(1).split(".")[0]) or None
    normalized = normalize_phrase(raw)
    if normalized and len(normalized.split()) <= 4:
        return normalized
    return None


def detector_prompt_for_goal(goal: str) -> str:
    normalized = normalize_phrase(goal)
    return normalized if normalized else ""


def compose_textnav_goal_text(goal_record: dict) -> str:
    if not isinstance(goal_record, dict):
        raise TypeError("TextNav goal record must be a dictionary")

    attributes: dict[str, str] = {}
    for field_name in ("intrinsic_attributes", "extrinsic_attributes"):
        value = goal_record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TextNav goal record requires non-empty {field_name}")
        attributes[field_name] = value.strip()

    return f"{attributes['intrinsic_attributes']} {attributes['extrinsic_attributes']}"


def dedupe_goal_phrases(phrases: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = normalize_phrase(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@lru_cache(maxsize=8192)
def _cached_label_in_query_set(label: str, queries: tuple[str, ...]) -> bool:
    return bool(label) and label in queries


def label_in_query_set(label: str, queries: Iterable[str]) -> bool:
    """Return whether label exactly matches one of the detector queries."""
    return _cached_label_in_query_set(label, tuple(queries))
