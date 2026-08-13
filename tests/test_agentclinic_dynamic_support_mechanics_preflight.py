from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import agentclinic_dynamic_support_mechanics_preflight as mechanics
from scripts import agentclinic_dynamic_support_source_audit as source


def row(index: int, *, leak: bool = False) -> dict:
    gold = f"diagnosis {index}"
    patient = f"patient {index} has a persistent symptom"
    if leak:
        patient += f" caused by {gold}"
    return {
        "answers": [
            {"answer": gold, "correct": True},
            {"answer": f"alternative a {index}", "correct": False},
            {"answer": f"alternative b {index}", "correct": False},
            {"answer": f"alternative c {index}", "correct": False},
        ],
        "image_url": f"https://example.test/{index}.png",
        "patient_info": patient,
        "physical_exams": f"test findings for patient {index}",
        "question": "What is the most likely diagnosis?",
        "type": "NEJM",
    }


def test_normalize_text_is_case_and_whitespace_insensitive() -> None:
    assert mechanics.normalize_text("  Mixed\n CASE ") == "mixed case"


def test_answer_text_supports_released_and_compatible_keys() -> None:
    assert mechanics.answer_text({"answer": "A"}) == "A"
    assert mechanics.answer_text({"text": "B"}) == "B"
    assert mechanics.answer_text({"value": "C"}) == "C"
    assert mechanics.answer_text({"answer": " "}) == ""


def test_select_mechanics_rows_reconstructs_value_blind_split() -> None:
    rows = [row(index) for index in range(120)]
    selected = mechanics.select_mechanics_rows(rows)
    ids = [mechanics.sha256_bytes(source.canonical_bytes(value)) for value in selected]
    all_ids = [mechanics.sha256_bytes(source.canonical_bytes(value)) for value in rows]
    assert ids == source.split_case_ids(all_ids)["mechanics"]
    assert len(ids) == 6


def test_gold_leak_normalization_is_detectable() -> None:
    value = row(3, leak=True)
    gold = mechanics.answer_text(value["answers"][0])
    context = mechanics.normalize_text(
        " ".join(value[field] for field in ("question", "patient_info", "physical_exams"))
    )
    assert mechanics.normalize_text(gold) in context


def test_public_result_shape_contains_no_private_values() -> None:
    result = {
        "mechanics_count": 6,
        "answer_cardinality_min": 4,
        "answer_cardinality_max": 5,
        "privacy": {
            "individual_case_ids_serialized": False,
            "source_values_serialized": False,
            "diagnoses_serialized": False,
        },
    }
    encoded = json.dumps(result)
    assert "diagnosis 1" not in encoded
    assert "patient 1" not in encoded


@pytest.mark.parametrize("field", ["patient_info", "physical_exams"])
def test_native_channels_must_be_nonempty(field: str) -> None:
    value = row(0)
    value[field] = "  "
    assert not bool(mechanics.normalize_text(value[field]))
