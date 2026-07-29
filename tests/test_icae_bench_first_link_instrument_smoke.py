from __future__ import annotations

import json

import pytest

from scripts.icae_bench_first_link_instrument_smoke import (
    NUM_HYPOTHESES,
    NUM_QUESTIONS,
    parse_branch_answers,
    parse_coverage,
    parse_likelihoods,
    parse_proxy_coverage,
    select_instrument_task,
)


def test_branch_answers_require_complete_index_cover() -> None:
    payload = {
        "branches": [
            {
                "question_index": index,
                "positive_answer": f"Concrete answer for question {index}.",
            }
            for index in range(NUM_QUESTIONS)
        ]
    }
    assert len(parse_branch_answers(json.dumps(payload))) == NUM_QUESTIONS
    payload["branches"][1]["question_index"] = 0
    with pytest.raises(ValueError, match="duplicated"):
        parse_branch_answers(json.dumps(payload))


def test_likelihood_parser_builds_complete_matrix() -> None:
    payload = {
        "cells": [
            {
                "hypothesis_index": h,
                "question_index": q,
                "positive_probability": (h + q) % 101,
            }
            for h in range(NUM_HYPOTHESES)
            for q in range(NUM_QUESTIONS)
        ]
    }
    matrix = parse_likelihoods(json.dumps(payload))
    assert len(matrix) == NUM_HYPOTHESES
    assert len(matrix[0]) == NUM_QUESTIONS


def test_proxy_coverage_requires_frozen_order() -> None:
    rows = [
        {
            "branch": branch,
            "hypothesis_id": f"H{index:02d}",
            "covered": index % 2 == 0,
        }
        for branch in ("positive", "negative")
        for index in range(NUM_HYPOTHESES)
    ]
    parsed = parse_proxy_coverage(json.dumps({"rows": rows}))
    assert len(parsed["positive"]) == NUM_HYPOTHESES
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="frozen order"):
        parse_proxy_coverage(json.dumps({"rows": rows}))


def test_endpoint_coverage_requires_exact_ids_and_order() -> None:
    payload = {
        "rows": [
            {"constraint_id": "C001", "covered": True},
            {"constraint_id": "C002", "covered": False},
        ]
    }
    assert parse_coverage(
        json.dumps(payload),
        expected_ids=["C001", "C002"],
        id_key="constraint_id",
        label="endpoint",
    ) == {"C001": True, "C002": False}
    payload["rows"].reverse()
    with pytest.raises(ValueError, match="frozen order"):
        parse_coverage(
            json.dumps(payload),
            expected_ids=["C001", "C002"],
            id_key="constraint_id",
            label="endpoint",
        )


def test_instrument_task_selection_excludes_all_opened_tasks() -> None:
    opened = {
        "realcode@044",
        "realcode@276",
        "realcode@235",
        "realcode@185",
    }
    manifest = {
        "partitions": {
            "mechanics": [
                {"alias": alias, "language": "Python"}
                for alias in opened
            ]
            + [
                {"alias": f"realcode@{index:03d}", "language": "Python"}
                for index in range(1, 9)
            ]
        }
    }
    selected = select_instrument_task(manifest)
    assert selected["alias"] not in opened
