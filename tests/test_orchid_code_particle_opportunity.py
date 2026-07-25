import pytest

from scripts.orchid_code_particle_opportunity import (
    entropy_from_outcomes,
    extract_python_program,
    score_task,
)


def test_extract_python_program_accepts_plain_or_single_fenced_code():
    plain = "def target(value):\n    return value + 1\n"
    assert extract_python_program(plain, "target") == plain.strip()
    fenced = "```python\ndef target(value):\n    return value - 1\n```"
    assert "return value - 1" in extract_python_program(fenced, "target")


def test_extract_python_program_rejects_wrong_entry_point():
    with pytest.raises(ValueError, match="does not define target"):
        extract_python_program("def other(value):\n    return value\n", "target")


def test_entropy_from_outcomes_uses_particle_multiplicity():
    assert entropy_from_outcomes(["a", "a", "b", "b"]) == pytest.approx(
        0.6931471805599453
    )
    assert entropy_from_outcomes(["a", "a", "a"]) == 0.0


def test_score_task_separates_query_selection_from_holdout_endpoint():
    outputs = [
        ["yes", "left", "pass", "pass"],
        ["yes", "right", "pass", "pass"],
        ["no", "middle", "fail", "fail"],
        ["no", "other", "fail", "fail"],
    ]
    target = ["yes", "middle", "pass", "pass"]
    record = score_task(
        task_id=7,
        particle_outputs=outputs,
        query_indices=[0, 1],
        holdout_indices=[2, 3],
        target_outputs=target,
    )
    assert record["initial_holdout_pass_fraction"] == pytest.approx(0.5)
    assert record["myopic_test_index"] == 1
    assert record["oracle_test_index"] == 0
    assert record["myopic_endpoint"] == 0.0
    assert record["oracle_endpoint"] == 1.0
    assert record["oracle_gap_over_myopic"] == 1.0
