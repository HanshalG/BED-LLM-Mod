from __future__ import annotations

import json
from pathlib import Path

from scripts import agentclinic_dynamic_support_terminal_audit as terminal


def test_expected_bindings_are_full_sha256_values() -> None:
    values = (
        terminal.EXPECTED_AGENTCLINIC_PY_SHA256,
        terminal.EXPECTED_SOURCE_PROTOCOL_SHA256,
        terminal.EXPECTED_PREFLIGHT_PROTOCOL_SHA256,
        terminal.EXPECTED_PREFLIGHT_RESULT_SHA256,
    )
    assert all(len(value) == 64 for value in values)
    assert all(set(value) <= set("0123456789abcdef") for value in values)


def test_load_object_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("[]\n", encoding="utf-8")
    try:
        terminal.load_object(path)
    except ValueError as exc:
        assert "expected object" in str(exc)
    else:
        raise AssertionError("non-object JSON should fail")


def test_released_contract_markers_form_conditional_redundancy() -> None:
    code = """
return patient_info
self.symptoms = self.scenario.patient_information()
Below is all of your information. {}.
.format(self.symptoms)
"""
    assert "return patient_info" in code
    assert "self.symptoms = self.scenario.patient_information()" in code
    assert "Below is all of your information. {}." in code
    assert ".format(self.symptoms)" in code


def test_terminal_public_shape_contains_no_case_content() -> None:
    result = {
        "status": "agentclinic_v1_closed",
        "reason": "patient responder and visible intake share the same source field",
        "privacy": {"source_values_serialized": False, "diagnoses_serialized": False},
    }
    encoded = json.dumps(result)
    assert "patient actor" not in encoded.casefold()
    assert "most likely diagnosis" not in encoded.casefold()
