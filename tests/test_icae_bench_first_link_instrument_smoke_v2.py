from __future__ import annotations

import json

import pytest

from scripts.icae_bench_first_link_instrument_smoke_v2 import (
    EXCLUDED_OPENED_ALIASES,
    matcher_messages_set,
    parse_matcher_set,
    select_instrument_task_v2,
)


def test_matcher_set_canonicalizes_unique_known_ids() -> None:
    payload = {
        "matched_ids": ["C003", "C001"],
        "fallback": False,
    }
    parsed = parse_matcher_set(
        json.dumps(payload),
        valid_ids=["C001", "C002", "C003"],
        label="match",
    )
    assert parsed == {
        "matched_ids": ["C001", "C003"],
        "fallback": False,
    }


@pytest.mark.parametrize(
    "matched_ids,fallback,error",
    [
        (["C001", "C001"], False, "duplicates"),
        (["C999"], False, "unknown"),
        ([], False, "inconsistent"),
    ],
)
def test_matcher_set_rejects_invalid_sets(
    matched_ids: list[str],
    fallback: bool,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        parse_matcher_set(
            json.dumps(
                {"matched_ids": matched_ids, "fallback": fallback}
            ),
            valid_ids=["C001", "C002"],
            label="match",
        )


def test_matcher_prompt_declares_unordered_set() -> None:
    record = {
        "fuzzy_prd": "Build a service.",
        "oracle_data": {
            "hidden_constraints": [
                {
                    "constraint_id": "C001",
                    "trigger_keywords": ["timeout"],
                }
            ]
        },
    }
    messages = matcher_messages_set(record, "What timeout is required?")
    payload = json.loads(messages[1]["content"])
    assert any("unordered set" in rule for rule in payload["rules"])
    assert not any("catalog order" in rule for rule in payload["rules"])


def test_v2_selection_excludes_every_previously_opened_task() -> None:
    manifest = {
        "partitions": {
            "mechanics": [
                {"alias": alias, "language": "Python"}
                for alias in EXCLUDED_OPENED_ALIASES
            ]
            + [
                {"alias": f"realcode@{index:03d}", "language": "Python"}
                for index in range(1, 9)
            ]
        }
    }
    assert (
        select_instrument_task_v2(manifest)["alias"]
        not in EXCLUDED_OPENED_ALIASES
    )
