from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import agentclinic_medqa_dynamic_support_source_audit as audit


def valid_row(index: int) -> dict:
    return {
        "OSCE_Examination": {
            "Correct_Diagnosis": f"diagnosis {index}",
            "Objective_for_Doctor": f"objective {index}",
            "Patient_Actor": {"history": f"patient {index}"},
            "Physical_Examination_Findings": {"exam": f"finding {index}"},
            "Test_Results": {"test": f"result {index}"},
        }
    }


def test_load_rows_counts_final_non_newline_record(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(valid_row(0)) + "\n" + json.dumps(valid_row(1)), encoding="utf-8")
    assert len(audit.load_rows(path)) == 2


def test_exact_schema_rejects_additional_osce_field() -> None:
    row = valid_row(0)
    assert set(row["OSCE_Examination"]) == audit.EXPECTED_OSCE_FIELDS
    row["OSCE_Examination"]["Management_and_Follow_Up"] = "private value"
    assert set(row["OSCE_Examination"]) != audit.EXPECTED_OSCE_FIELDS


def test_expected_source_hashes_are_full_digests() -> None:
    for value in (audit.CODE_SHA256, audit.DATA_SHA256):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


def test_public_failure_shape_contains_no_source_values() -> None:
    result = {
        "status": "source_failed_closed",
        "population_shape": {
            "observed_nonblank_rows": 214,
            "rows_with_exact_five_field_osce_schema": 213,
        },
        "privacy": {"source_values_serialized": False, "diagnoses_serialized": False},
    }
    encoded = json.dumps(result)
    assert "diagnosis 0" not in encoded
    assert "patient 0" not in encoded


def test_load_rows_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row is not an object"):
        audit.load_rows(path)
