from __future__ import annotations

import json

import pytest

from scripts.icae_bench_full_world_instrument_smoke import (
    CLAUSES_PER_WORLD,
    QUESTION_COUNT,
    WORLD_COUNT,
    effective_world_count,
    endpoint_statistics,
    parse_endpoint,
    parse_likelihoods,
    parse_retention,
    parse_world_support,
    select_development_task,
)


def _support_payload() -> dict:
    return {
        "worlds": [
            {
                "world_index": world,
                "probability_percent": 13 if world < 4 else 12,
                "clauses": [
                    f"World {world} clause {clause} has concrete behavior."
                    for clause in range(CLAUSES_PER_WORLD)
                ],
            }
            for world in range(WORLD_COUNT)
        ],
        "questions": [
            {
                "question_index": index,
                "question": f"Which concrete behavior applies to topic {index}?",
            }
            for index in range(QUESTION_COUNT)
        ],
    }


def test_world_support_parser_requires_probability_distribution() -> None:
    support = parse_world_support(
        json.dumps(_support_payload()),
        label="support",
    )
    assert len(support["worlds"]) == WORLD_COUNT
    assert sum(world["probability"] for world in support["worlds"]) == 1.0
    payload = _support_payload()
    payload["worlds"][0]["probability_percent"] = 12
    with pytest.raises(ValueError, match="sum to 100"):
        parse_world_support(json.dumps(payload), label="support")


def test_world_support_parser_canonicalizes_indexed_collections() -> None:
    payload = _support_payload()
    payload["worlds"] = payload["worlds"][1:] + payload["worlds"][:1]
    payload["questions"] = (
        payload["questions"][1:] + payload["questions"][:1]
    )
    support = parse_world_support(json.dumps(payload), label="support")
    assert [world["world_index"] for world in support["worlds"]] == list(
        range(WORLD_COUNT)
    )
    assert support["questions"][0].endswith("topic 0?")


def test_world_support_parser_rejects_duplicate_worlds() -> None:
    payload = _support_payload()
    payload["worlds"][1]["clauses"] = payload["worlds"][0]["clauses"]
    with pytest.raises(ValueError, match="duplicate worlds"):
        parse_world_support(json.dumps(payload), label="support")


def test_likelihood_parser_canonicalizes_indexed_cells() -> None:
    cells = [
        {
            "world_index": world,
            "question_index": question,
            "positive_probability": 10 * question,
        }
        for world in range(WORLD_COUNT)
        for question in range(QUESTION_COUNT)
    ]
    cells.reverse()
    parsed = parse_likelihoods(json.dumps({"cells": cells}))
    assert parsed[2][3] == 0.3
    cells[0] = dict(cells[1])
    with pytest.raises(ValueError, match="duplicated"):
        parse_likelihoods(json.dumps({"cells": cells}))


def test_retention_parser_canonicalizes_indexed_rows() -> None:
    rows = [
        {
            "branch": branch,
            "initial_world_index": world,
            "represented": world % 2 == 0,
        }
        for branch in ("positive", "negative")
        for world in range(WORLD_COUNT)
    ]
    rows.reverse()
    parsed = parse_retention(json.dumps({"rows": rows}))
    assert sum(parsed["positive"]) == 4
    rows[0] = dict(rows[1])
    with pytest.raises(ValueError, match="duplicated"):
        parse_retention(json.dumps({"rows": rows}))


def test_endpoint_is_per_world_and_probability_weighted() -> None:
    ids = ["C1", "C2"]
    rows = [
        {
            "world_index": world,
            "constraint_id": constraint,
            "covered": world == 0,
        }
        for world in range(WORLD_COUNT)
        for constraint in ids
    ]
    coverage = parse_endpoint(
        json.dumps({"rows": rows}),
        constraint_ids=ids,
    )
    support = parse_world_support(
        json.dumps(_support_payload()),
        label="support",
    )
    expected, rates = endpoint_statistics(support, coverage)
    assert expected == pytest.approx(0.13)
    assert rates == [1.0] + [0.0] * (WORLD_COUNT - 1)


def test_effective_world_count_detects_noncollapsed_distribution() -> None:
    support = parse_world_support(
        json.dumps(_support_payload()),
        label="support",
    )
    assert effective_world_count(support) > 7.9


def test_development_selection_applies_structural_eligibility() -> None:
    manifest = {
        "partitions": {
            "mechanics": [
                {"alias": "weak", "language": "A"},
                {"alias": "strong-a", "language": "B"},
                {"alias": "strong-b", "language": "C"},
            ]
        }
    }
    audit = {
        "tasks": [
            {
                "alias": "weak",
                "substantive_constraint_count": 9,
                "lexical_unlock_target_count": 9,
            },
            {
                "alias": "strong-a",
                "substantive_constraint_count": 10,
                "lexical_unlock_target_count": 5,
            },
            {
                "alias": "strong-b",
                "substantive_constraint_count": 12,
                "lexical_unlock_target_count": 7,
            },
        ]
    }
    assert select_development_task(manifest, audit)["alias"] in {
        "strong-a",
        "strong-b",
    }
