from __future__ import annotations

import copy

import pytest

from scripts import agentclinic_dynamic_support_source_audit as audit


def row(index: int) -> dict:
    return {
        "answers": [
            {"text": f"wrong-{index}", "correct": False},
            {"text": f"truth-{index}", "correct": True},
        ],
        "image_url": f"https://example.test/{index}.png",
        "patient_info": f"patient-{index}",
        "physical_exams": f"exam-{index}",
        "question": f"question-{index}",
        "type": ["case"],
    }


def test_value_blind_split_is_complete_disjoint_and_stable() -> None:
    rows = [row(index) for index in range(10)]
    counts = (("mechanics", 2), ("opportunity", 3), ("reserve", 5))

    first, first_splits = audit.audit_rows(rows, counts)
    second, second_splits = audit.audit_rows(list(reversed(rows)), counts)

    assert all(first["gates"].values())
    assert first["split_counts"] == {"mechanics": 2, "opportunity": 3, "reserve": 5}
    assert first_splits == second_splits
    assert first["ordered_split_case_id_sha256"] == second["ordered_split_case_id_sha256"]
    assert len({case_id for values in first_splits.values() for case_id in values}) == 10


@pytest.mark.parametrize("mutation", ["duplicate", "two_correct", "http_image", "missing_value", "extra_field"])
def test_source_gate_rejects_invalid_population(mutation: str) -> None:
    rows = [row(index) for index in range(4)]
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "two_correct":
        rows[0]["answers"][0]["correct"] = True
    elif mutation == "http_image":
        rows[0]["image_url"] = "http://example.test/0.png"
    elif mutation == "missing_value":
        rows[0]["physical_exams"] = ""
    elif mutation == "extra_field":
        rows[0]["diagnosis"] = "must not be in the release schema"

    summary, _ = audit.audit_rows(rows, (("mechanics", 2), ("reserve", 2)))

    assert not all(summary["gates"].values())


def test_split_counts_must_cover_population() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        audit.audit_rows([row(0), row(1)], (("mechanics", 1),))
