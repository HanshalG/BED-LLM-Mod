"""Strict-but-tolerant JSON parsing for Paprika LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a JSON object")


def parse_string_list(text: str, key: str, *, minimum: int, maximum: int) -> list[str]:
    value = parse_json_object(text).get(key)
    if not isinstance(value, list):
        raise ValueError(f"JSON field {key!r} must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = item.strip()
        if normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    if not minimum <= len(result) <= maximum:
        raise ValueError(f"JSON field {key!r} must contain {minimum}-{maximum} unique strings")
    return result


def parse_distribution(text: str, outcomes: tuple[str, ...]) -> tuple[float, ...]:
    raw = parse_json_object(text).get("probabilities")
    if not isinstance(raw, dict):
        raise ValueError("JSON field 'probabilities' must be an object")
    lookup = {str(key).strip().casefold(): value for key, value in raw.items()}
    values: list[float] = []
    for outcome in outcomes:
        key = outcome.casefold()
        if key not in lookup:
            values.append(0.0)
            continue
        value = lookup[key]
        if isinstance(value, bool):
            raise ValueError("Outcome probabilities must be numeric")
        if value is None:
            values.append(0.0)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing numeric probability for {outcome!r}") from exc
        if number < 0.0:
            raise ValueError("Outcome probabilities must be non-negative")
        values.append(number)
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Outcome probabilities must have positive total mass")
    return tuple(value / total for value in values)
