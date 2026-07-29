from __future__ import annotations

from scripts import askbench_llm_native_source_audit as audit


def _row(**overrides):
    row = {
        "id": "row-1",
        "ori_question": (
            "Complete clinical question\nA. alpha\nB. beta\nC. gamma\nD. delta"
        ),
        "degraded_question": (
            "Incomplete clinical question\nA. alpha\nB. beta\nC. gamma\nD. delta"
        ),
        "degraded_info": "Age and laboratory values were removed.",
        "expected_answer": "The answer is B.",
        "required_points": ["Exact age", "Laboratory value", "Exposure"],
        "source_task": "ask_mind_medqade",
        "_public_row_sha256": "abc123",
    }
    row.update(overrides)
    return row


def test_eligibility_accepts_exact_medqa_row() -> None:
    assert audit.eligibility_errors(_row()) == []


def test_eligibility_rejects_unmodified_or_malformed_rows() -> None:
    assert "explicit_no_modification" in audit.eligibility_errors(
        _row(degraded_info="No modifications were made to the question.")
    )
    assert "answer_options" in audit.eligibility_errors(
        _row(degraded_question="Question\nA. alpha\nB. beta\nC. gamma")
    )
    assert "expected_answer_format" in audit.eligibility_errors(
        _row(expected_answer="B")
    )


def test_split_is_deterministic_and_disjoint() -> None:
    rows = [
        _row(id=f"row-{index:03d}", _public_row_sha256=str(index))
        for index in range(70)
    ]

    first = audit.split_rows(rows)
    second = audit.split_rows(list(reversed(rows)))

    assert [[row["id"] for row in split] for split in first] == [
        [row["id"] for row in split] for split in second
    ]
    development, holdout, unused = first
    assert len(development) == audit.DEVELOPMENT_SIZE
    assert len(holdout) == audit.HOLDOUT_SIZE
    assert len(unused) == 20
    assert not ({row["id"] for row in development} & {row["id"] for row in holdout})


def test_candidate_payload_excludes_hidden_fields() -> None:
    row = _row()
    payload = audit.candidate_payload(row)

    assert set(payload) == {"degraded_question", "answer_option_labels"}
    assert audit.hidden_state_separated(row)
