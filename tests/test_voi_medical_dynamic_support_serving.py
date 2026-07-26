import pytest

from scripts.voi_medical_dynamic_support_serving import (
    HYPOTHESIS_COUNT,
    MECHANICS_IDS,
    parse_hypotheses,
    target_is_covered,
)


def test_hypothesis_parser_requires_exact_distinct_support() -> None:
    text = "\n".join(
        f"H{index}|Diagnosis {index}|Evidence rationale number {index}."
        for index in range(1, HYPOTHESIS_COUNT + 1)
    )
    parsed = parse_hypotheses(text)
    assert len(parsed) == HYPOTHESIS_COUNT
    assert parsed[0]["diagnosis"] == "Diagnosis 1"
    duplicate = text.replace("Diagnosis 6", "diagnosis 1")
    with pytest.raises(ValueError, match="distinct"):
        parse_hypotheses(duplicate)


def test_target_coverage_is_casefolded_and_phrase_based() -> None:
    support = [
        {"diagnosis": "Acute gastroenteritis", "rationale": "Relevant symptoms."}
    ]
    assert target_is_covered("Gastroenteritis", support)
    assert not target_is_covered("Gastritis", support)
    assert MECHANICS_IDS == (474, 67, 151, 50, 284)
