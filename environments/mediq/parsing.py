"""Strict structured-output parsers for MediQ calls."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence


def parse_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response is not a JSON object")
        try:
            value = json.loads(clean[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def parse_distribution(text: str, labels: Sequence[str]) -> tuple[float, ...]:
    raw = parse_json_object(text).get("probabilities")
    if not isinstance(raw, dict):
        raise ValueError("response requires a 'probabilities' object")
    expected = tuple(str(label) for label in labels)
    if set(raw) != set(expected):
        raise ValueError("probability keys must exactly match the requested labels")
    probabilities: list[float] = []
    for label in expected:
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probability for {label!r} must be numeric")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"probability for {label!r} must be in [0, 1]")
        probabilities.append(number)
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    return tuple(value / total for value in probabilities)


def parse_fact_selection(
    text: str,
    *,
    num_facts: int,
    max_facts: int,
) -> tuple[tuple[int, ...], bool]:
    raw = parse_json_object(text)
    indices = raw.get("fact_indices")
    cannot_answer = raw.get("cannot_answer")
    if not isinstance(indices, list) or not isinstance(cannot_answer, bool):
        raise ValueError("patient response requires fact_indices list and cannot_answer boolean")
    parsed: list[int] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("fact_indices must contain integers")
        if not 0 <= value < num_facts:
            raise ValueError("fact index is outside the supplied record")
        if value not in parsed:
            parsed.append(value)
    if len(parsed) > max_facts:
        raise ValueError(f"patient may select at most {max_facts} facts")
    if cannot_answer and parsed:
        raise ValueError("cannot_answer=true requires an empty fact_indices list")
    if not cannot_answer and not parsed:
        raise ValueError("patient must select a fact or set cannot_answer=true")
    return tuple(parsed), cannot_answer


def parse_candidate_validation(text: str) -> tuple[bool, str]:
    raw = parse_json_object(text)
    valid = raw.get("valid")
    reason = raw.get("reason")
    if not isinstance(valid, bool) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("candidate validation requires valid boolean and non-empty reason")
    return valid, reason.strip()


def parse_candidate_set_validation(
    text: str, num_candidates: int
) -> tuple[tuple[tuple[int, ...], ...], str]:
    raw = parse_json_object(text)
    duplicate_groups = raw.get("duplicate_groups")
    reason = raw.get("reason")
    if not isinstance(duplicate_groups, list):
        raise ValueError("candidate-set validation requires duplicate_groups list")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("candidate-set validation requires a non-empty reason")
    parsed: list[tuple[int, ...]] = []
    used: set[int] = set()
    for group in duplicate_groups:
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError("each duplicate group requires at least two indices")
        indices: list[int] = []
        for value in group:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("duplicate-group indices must be integers")
            if not 0 <= value < num_candidates:
                raise ValueError("duplicate-group index is outside the candidate set")
            if value in indices or value in used:
                raise ValueError("duplicate-group indices must be unique")
            indices.append(value)
            used.add(value)
        parsed.append(tuple(indices))
    return tuple(parsed), reason.strip()


def parse_relevance(text: str) -> tuple[bool, str]:
    raw = parse_json_object(text)
    relevant = raw.get("relevant")
    reason = raw.get("reason")
    if not isinstance(relevant, bool) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("relevance judgment requires relevant boolean and non-empty reason")
    return relevant, reason.strip()


def parse_mapping(
    text: str,
    outcomes: Sequence[str],
) -> tuple[str | None, bool]:
    raw = parse_json_object(text)
    clean = raw.get("clean")
    selected = raw.get("outcome")
    if not isinstance(clean, bool):
        raise ValueError("mapping requires a clean boolean")
    if selected is not None and not isinstance(selected, str):
        raise ValueError("mapping outcome must be a string or null")
    if not clean and selected is not None:
        raise ValueError("unclean mapping requires outcome=null")
    canonical = None
    if clean and isinstance(selected, str):
        canonical = next(
            (
                outcome
                for outcome in outcomes
                if outcome.casefold() == selected.strip().casefold()
            ),
            None,
        )
        if canonical is None:
            raise ValueError("clean mapping outcome is not one of the supplied outcomes")
    if clean and canonical is None:
        raise ValueError("clean mapping requires a canonical outcome")
    return canonical, clean
