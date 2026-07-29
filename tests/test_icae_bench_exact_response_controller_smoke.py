from __future__ import annotations

import json

import pytest

from scripts.icae_bench_exact_response_controller_smoke import (
    exact_controller_reply,
    parse_matcher,
    select_unopened_mechanics,
    trigger_catalog,
)


def _record() -> dict:
    return {
        "fuzzy_prd": "Build an event adapter.",
        "oracle_data": {
            "hidden_constraints": [
                {
                    "constraint_id": "C001",
                    "trigger_keywords": ["what event formats are supported"],
                    "oracle_response": "Support formats A and B.",
                },
                {
                    "constraint_id": "C002",
                    "trigger_keywords": ["what output order is required"],
                    "oracle_response": "Preserve insertion order.",
                },
            ],
            "fallback_response": "Please ask a more specific question.",
        },
    }


def test_trigger_catalog_excludes_oracle_responses() -> None:
    catalog = trigger_catalog(_record())
    assert catalog == [
        {
            "id": "C001",
            "trigger_phrases": ["what event formats are supported"],
        },
        {
            "id": "C002",
            "trigger_phrases": ["what output order is required"],
        },
    ]
    assert "Support formats A and B." not in json.dumps(catalog)


def test_matcher_parser_requires_known_ordered_ids() -> None:
    parsed = parse_matcher(
        '{"matched_ids":["C001","C002"],"fallback":false}',
        valid_ids=["C001", "C002"],
        label="matcher",
    )
    assert parsed["matched_ids"] == ["C001", "C002"]
    with pytest.raises(ValueError, match="catalog order"):
        parse_matcher(
            '{"matched_ids":["C002","C001"],"fallback":false}',
            valid_ids=["C001", "C002"],
            label="matcher",
        )
    with pytest.raises(ValueError, match="fallback is inconsistent"):
        parse_matcher(
            '{"matched_ids":[],"fallback":false}',
            valid_ids=["C001", "C002"],
            label="matcher",
        )


def test_exact_controller_returns_stored_text_or_fallback() -> None:
    record = _record()
    assert exact_controller_reply(record, ["C002"]) == "Preserve insertion order."
    assert exact_controller_reply(record, []) == (
        "Please ask a more specific question."
    )


def test_unopened_selection_excludes_previous_smoke_tasks() -> None:
    manifest = {
        "partitions": {
            "mechanics": [
                {"alias": "realcode@044", "language": "JavaScript"},
                {"alias": "realcode@276", "language": "PHP"},
                *[
                    {"alias": f"realcode@{index:03d}", "language": "Python"}
                    for index in range(1, 11)
                ],
            ]
        }
    }
    selected = select_unopened_mechanics(manifest)
    assert len(selected) == 2
    assert not {
        row["alias"] for row in selected
    } & {"realcode@044", "realcode@276"}
