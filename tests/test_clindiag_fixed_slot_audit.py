from __future__ import annotations

from scripts.clindiag_fixed_slot_audit import (
    ACTION_IDS,
    fixed_evidence_slots,
    missing_slot_ids,
)
from scripts.clindiag_staged_generator_gate import ClinDiagCase


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="fresh",
        subset="challenging",
        initial_information="A patient presented with fatigue.",
        medical_history={
            "medical_history": {
                "history_of_present_illness": "Fatigue for one month.",
                "past_medical_history": "No prior illness.",
                "family_history": "No relevant family history.",
                "social_history": "Lives independently.",
            }
        },
        physical_examination={
            "physical_examinations": [{"relevant_findings": "Pallor"}]
        },
        diagnostic_test={
            "laboratory_examinations": [
                {"procedure_name": "CBC", "findings": "Low hemoglobin"},
                {"procedure_name": "Bone marrow biopsy", "findings": "Diagnostic"},
                {"procedure_name": "Ferritin", "findings": "Low"},
            ],
            "radiographic_examinations": [
                {"procedure_name": "Chest X-ray", "findings": "Clear"}
            ],
            "other_examinations": [
                {"procedure_name": "ECG", "findings": "Normal"}
            ],
        },
        final_diagnosis="Iron deficiency anemia",
    )


def test_fixed_slots_have_common_ids_and_filter_confirmatory_entries() -> None:
    slots = fixed_evidence_slots(_case())
    assert tuple(slots) == ACTION_IDS
    assert slots["lab_1"]["procedure_name"] == "CBC"
    assert slots["lab_2"]["procedure_name"] == "Ferritin"
    assert missing_slot_ids(slots) == []


def test_missing_slot_ids_detects_incomplete_fixed_world() -> None:
    slots = fixed_evidence_slots(_case())
    slots["other_1"] = None
    assert missing_slot_ids(slots) == ["other_1"]
