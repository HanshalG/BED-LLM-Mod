from __future__ import annotations

import json

import pytest

from scripts import guessinggame_path_bed_source_audit as audit


def _row(
    index: int = 0,
    *,
    target: str = "abacus",
    material: str = "It is made of wood and metal.",
    function: str = "It is used to perform calculations.",
) -> audit.GameRow:
    return audit.GameRow(
        source_index=index,
        target=target,
        reported_turns=4,
        material_question="What material is the object made of?",
        material_answer=material,
        function_question=(
            "What is the primary function or purpose of the object?"
        ),
        function_answer=function,
    )


def test_parse_game_line_extracts_first_two_pairs() -> None:
    line = (
        "abacus,3,Oracle said: What is your first question?\t"
        "Guesser said: What material is the object made of?\t"
        "Oracle said: Wood and metal.\t"
        "Guesser said: What is the primary function of the object?\t"
        "Oracle said: It performs calculations.\t"
        "Guesser said: Is it an abacus?\tOracle said: Correct."
    )
    row = audit.parse_game_line(line, source_index=7)
    assert row.source_index == 7
    assert row.target == "abacus"
    assert row.material_answer == "Wood and metal."
    assert row.function_answer == "It performs calculations."


def test_literal_target_detection_is_token_bounded() -> None:
    assert audit.contains_literal_target(
        "The object is made of cashmere.", "cashmere"
    )
    assert not audit.contains_literal_target(
        "The object stores cash.", "ash"
    )


def test_eligibility_rejects_leak_or_missing_function() -> None:
    assert audit.eligibility_errors(_row()) == []
    leak = _row(material="The object is an abacus made of wood.")
    assert "material_answer_names_target" in audit.eligibility_errors(leak)
    wrong = audit.GameRow(
        **{
            **audit.row_payload(_row()),
            "function_question": "What is the size of the object?",
        }
    )
    assert "missing_function_question" in audit.eligibility_errors(wrong)


def test_split_is_deterministic_and_disjoint(monkeypatch) -> None:
    monkeypatch.setattr(audit, "SERVING_COUNT", 2)
    monkeypatch.setattr(audit, "MECHANICS_COUNT", 2)
    monkeypatch.setattr(audit, "DEVELOPMENT_COUNT", 3)
    monkeypatch.setattr(audit, "CONFIRMATION_COUNT", 4)
    rows = [
        _row(
            index=index,
            target=f"object-{index}",
            function=f"Function description {index}.",
        )
        for index in range(20)
    ]
    first = audit.split_rows(rows)
    second = audit.split_rows(list(reversed(rows)))
    assert [
        [audit.case_id(row) for row in split] for split in first
    ] == [
        [audit.case_id(row) for row in split] for split in second
    ]
    assert [len(split) for split in first] == [2, 2, 3, 4, 9]
    selected = sum((list(split) for split in first[:4]), [])
    assert len({audit.case_id(row) for row in selected}) == len(selected)


def test_public_row_is_opaque() -> None:
    row = _row()
    public = audit.public_row(row)
    serialized = json.dumps(public)
    assert set(public) == {"case_id", "source_row_sha256"}
    assert row.target not in serialized
    assert row.material_answer not in serialized


@pytest.mark.skipif(
    not audit.GAMES_PATH.exists(),
    reason="ignored official GuessingGame source is not installed",
)
def test_bound_official_source() -> None:
    source = audit.verify_source()
    assert source["commit"] == audit.SOURCE_COMMIT
    objects = audit.load_objects()
    games = audit.load_games()
    assert len(objects) == audit.EXPECTED_OBJECTS
    assert len(games) == audit.EXPECTED_GAMES
    assert {row.target for row in games} == set(objects)
