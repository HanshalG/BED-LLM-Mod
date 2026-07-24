from __future__ import annotations

from scripts.clindiag_staged_generator_gate import ClinDiagCase
from scripts.clindiag_test_card_audit import (
    eligible_test_cards,
    normalize_action_name,
)


def _case() -> ClinDiagCase:
    return ClinDiagCase(
        source_id="fresh1",
        subset="challenging",
        initial_information="A patient presented with fatigue.",
        medical_history={"medical_history": {"history": "Several weeks."}},
        physical_examination={"physical_examinations": [{"finding": "Pallor"}]},
        diagnostic_test={
            "laboratory_examinations": [
                {"procedure_name": "Complete Blood Count", "findings": "Low Hb"},
                {"procedure_name": "complete-blood count", "findings": "Stable"},
                {"procedure_name": "Bone marrow biopsy", "findings": "Diagnostic"},
                {"procedure_name": "Packed RBC transfusion", "findings": "Given"},
            ],
            "radiographic_examinations": [
                {"procedure_name": "CT Chest", "findings": "Clear"}
            ],
            "other_examinations": [
                {"procedure_name": "Not specified", "findings": "None"}
            ],
        },
        final_diagnosis="Example syndrome",
    )


def test_normalize_action_name_collapses_punctuation_and_case() -> None:
    assert normalize_action_name(" Complete-Blood  COUNT ") == "complete blood count"


def test_eligible_test_cards_group_and_apply_fixed_filters() -> None:
    cards, excluded = eligible_test_cards(_case())

    assert [card["procedure_name"] for card in cards] == [
        "Complete Blood Count",
        "CT Chest",
    ]
    assert len(cards[0]["observations"]) == 2
    assert excluded == {
        "confirmatory": 1,
        "intervention": 1,
        "missing_name": 1,
    }
