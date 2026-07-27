"""Strict, order-insensitive codecs for LLM line protocols."""

from __future__ import annotations

from collections.abc import Sequence


def parse_keyed_pipe_rows(
    text: str,
    *,
    expected_keys: Sequence[str],
    value_fields: int,
) -> dict[str, tuple[str, ...]]:
    """Parse one pipe-delimited row per key without requiring row order."""
    if text != text.strip():
        raise ValueError("row response must be canonical text")
    if value_fields < 1:
        raise ValueError("value_fields must be positive")
    keys = list(expected_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("expected keys must be nonempty and unique")
    expected_set = set(keys)
    lines = text.splitlines()
    if len(lines) != len(keys):
        raise ValueError("row response has wrong line count")
    rows: dict[str, tuple[str, ...]] = {}
    for line in lines:
        fields = line.split("|")
        if len(fields) != value_fields + 1:
            raise ValueError("row has wrong field count")
        key = fields[0]
        if key not in expected_set:
            raise ValueError("row has unknown key")
        if key in rows:
            raise ValueError("row repeats a key")
        if any(value == "" for value in fields[1:]):
            raise ValueError("row has an empty value")
        rows[key] = tuple(fields[1:])
    if set(rows) != expected_set:
        raise ValueError("row response is missing a key")
    return rows
