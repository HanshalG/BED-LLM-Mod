from __future__ import annotations

import pytest

from scripts import cupid_active_preference_source_audit as audit


def _interaction(context: str, preference: str = "hidden preference"):
    return {
        "context_factor": context,
        "contextual_preference": preference,
        "dialogue": [
            {"role": "user", "content": f"user in {context}"},
            {"role": "assistant", "content": f"assistant in {context}"},
        ],
    }


def _row(index: int = 0, instance_type: str = "consistent", **overrides):
    row = {
        "persona_id": f"persona-{index:03d}",
        "instance_type": instance_type,
        "current_request": "Write a response.",
        "current_context_factor": "current context",
        "current_contextual_preference": "formal and concise",
        "current_checklist": ["Is it formal?", "Is it concise?"],
        "prior_interactions": [
            _interaction("background one"),
            _interaction("current context", "formal and concise"),
            _interaction("background two"),
            _interaction("background three"),
            _interaction("background four"),
            _interaction("background five"),
            _interaction("background six"),
            _interaction("background seven"),
        ],
    }
    row.update(overrides)
    return row


def test_eligibility_accepts_complete_row() -> None:
    assert audit.eligibility_errors(_row()) == []


def test_eligibility_rejects_bad_history_or_missing_background() -> None:
    short = _row(prior_interactions=[_interaction("current context")] * 8)
    assert "insufficient_nonmatching_contexts" in audit.eligibility_errors(short)

    malformed = _row()
    malformed["prior_interactions"][0]["dialogue"][0]["role"] = "system"
    assert "invalid_prior_interaction" in audit.eligibility_errors(malformed)


def test_candidate_payload_excludes_hidden_metadata_and_same_context() -> None:
    row = _row()
    payload = audit.candidate_payload(row)

    assert set(payload) == {
        "current_request",
        "current_context_factor",
        "background_interactions",
    }
    assert [item["context_factor"] for item in payload["background_interactions"]] == [
        "background one",
        "background two",
    ]
    assert all(
        set(item) == {"context_factor", "dialogue"}
        for item in payload["background_interactions"]
    )
    assert audit.hidden_state_boundary_holds(row)


def test_split_is_deterministic_stratified_and_disjoint() -> None:
    rows = [
        _row(index=index, instance_type=instance_type)
        for instance_type in audit.INSTANCE_TYPES
        for index in range(40)
    ]

    first = audit.split_rows(rows)
    second = audit.split_rows(list(reversed(rows)))
    assert [
        [audit.row_id(row) for row in split] for split in first
    ] == [
        [audit.row_id(row) for row in split] for split in second
    ]

    serving, development, holdout, unused = first
    assert len(serving) == 5
    assert len(development) == 15
    assert len(holdout) == 60
    assert len(unused) == 40
    all_selected = [audit.row_id(row) for row in serving + development + holdout]
    assert len(all_selected) == len(set(all_selected))
    assert {
        instance_type: sum(
            row["instance_type"] == instance_type for row in holdout
        )
        for instance_type in audit.INSTANCE_TYPES
    } == {instance_type: 20 for instance_type in audit.INSTANCE_TYPES}


def test_public_split_rows_do_not_emit_hidden_text() -> None:
    public = audit.selected_public_row(_row())

    assert set(public) == {
        "id",
        "row_sha256",
        "instance_type",
        "prior_interaction_count",
        "exposed_background_count",
        "checklist_item_count",
    }
    assert "formal and concise" not in audit.canonical_json(public)


def test_official_formatter_exposes_dialogue_not_metadata() -> None:
    assert all(audit.official_formatter_control_flow().values())


def test_bound_source_and_full_dataset() -> None:
    pytest.importorskip("pyarrow")
    assert audit.verify_source()["data_sha256"] == audit.DATA_SHA256
    rows = audit.load_rows()
    assert len(rows) == audit.EXPECTED_TOTAL_ROWS
    assert all(not audit.eligibility_errors(row) for row in rows)
