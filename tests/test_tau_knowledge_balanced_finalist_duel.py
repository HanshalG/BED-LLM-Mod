from scripts.tau_knowledge_balanced_finalist_duel import (
    compact_duel_input,
    parse_duel,
    resolve_balanced_duel,
)
from tests.test_tau_knowledge_shared_comparative_pooling import _record
from scripts.tau_knowledge_shared_comparative_pooling import merge_record_pools


def _response(winner: str) -> dict[str, str]:
    return {
        "candidate_a_coverage": "Candidate A covers one policy.",
        "candidate_b_coverage": "Candidate B covers another policy.",
        "winner": winner,
    }


def test_duel_input_contains_only_two_complete_candidates() -> None:
    record = merge_record_pools(
        [_record("task_1", "a")],
        [_record("task_1", "b")],
    )[0]
    payload = compact_duel_input(
        record,
        candidate_a_root=1,
        candidate_b_root=8,
    )
    assert [item["candidate"] for item in payload["candidates"]] == ["A", "B"]
    assert len(payload["candidates"][0]["followups"]) == 4
    assert len(payload["candidates"][1]["followups"]) == 4
    assert "required_documents" not in payload


def test_parse_duel_requires_exact_schema_and_valid_winner() -> None:
    parsed = parse_duel(
        '{"candidate_a_coverage":"A facts","candidate_b_coverage":'
        '"B facts","winner":"B"}'
    )
    assert parsed["winner"] == "B"
    try:
        parse_duel(
            '{"candidate_a_coverage":"A","candidate_b_coverage":"B",'
            '"winner":"C"}'
        )
    except ValueError as exc:
        assert "A, B, or T" in str(exc)
    else:
        raise AssertionError("invalid winner should fail")


def test_balanced_duel_overrides_only_on_canonical_unanimity() -> None:
    selected, decision = resolve_balanced_duel(
        myopic_root=2,
        nonmyopic_root=7,
        forward=_response("B"),
        reversed_order=_response("A"),
    )
    assert (selected, decision) == (7, "unanimous_nonmyopic")

    selected, decision = resolve_balanced_duel(
        myopic_root=2,
        nonmyopic_root=7,
        forward=_response("A"),
        reversed_order=_response("A"),
    )
    assert (selected, decision) == (2, "fallback_myopic")

    selected, decision = resolve_balanced_duel(
        myopic_root=2,
        nonmyopic_root=7,
        forward=_response("T"),
        reversed_order=_response("B"),
    )
    assert (selected, decision) == (2, "fallback_myopic")


def test_same_finalist_needs_no_duel() -> None:
    selected, decision = resolve_balanced_duel(
        myopic_root=4,
        nonmyopic_root=4,
        forward=_response("A"),
        reversed_order=_response("B"),
    )
    assert (selected, decision) == (4, "same_finalist")
