from scripts.tau_knowledge_shared_comparative_pooling import merge_record_pools
from scripts.tau_knowledge_shared_support_finalist import (
    categorical_agreement,
    compact_support_input,
    parse_support_map,
    resolve_support_scores,
)
from tests.test_tau_knowledge_shared_comparative_pooling import _record


def _map(score: int) -> dict:
    return {
        "best_followup_index": 0,
        "labels": {"H1": "C"},
        "score": score,
    }


def test_support_input_contains_one_candidate_and_shared_need_ids() -> None:
    record = merge_record_pools(
        [_record("task_1", "a")],
        [_record("task_1", "b")],
    )[0]
    payload = compact_support_input(
        record,
        root_index=7,
        replicate_index=0,
    )
    assert "candidate" in payload
    assert "candidates" not in payload
    assert len(payload["candidate"]["followups"]) == 4
    assert {
        item["need_id"] for item in payload["shared_information_needs"]
    } == {f"H{index + 1}" for index in range(16)}
    assert "required_documents" not in payload


def test_parse_support_map_requires_complete_canonical_lines() -> None:
    parsed = parse_support_map(
        "BEST|F3\nH2|P\nH1|C\nH3|U",
        need_count=3,
    )
    assert parsed["best_followup_index"] == 2
    assert parsed["score"] == 3
    try:
        parse_support_map("BEST|F1\nH1|C\nH1|P", need_count=2)
    except ValueError as exc:
        assert "hypothesis line" in str(exc)
    else:
        raise AssertionError("duplicate support line should fail")


def test_support_override_requires_strict_replicate_separation() -> None:
    selected, decision = resolve_support_scores(
        myopic_root=2,
        nonmyopic_root=7,
        myopic_maps=[_map(10), _map(11)],
        nonmyopic_maps=[_map(12), _map(13)],
    )
    assert (selected, decision) == (7, "robust_nonmyopic_override")

    selected, decision = resolve_support_scores(
        myopic_root=2,
        nonmyopic_root=7,
        myopic_maps=[_map(10), _map(12)],
        nonmyopic_maps=[_map(12), _map(13)],
    )
    assert (selected, decision) == (2, "myopic_fallback")


def test_categorical_agreement_uses_need_labels() -> None:
    left = _map(2)
    left["labels"] = {"H1": "C", "H2": "P"}
    right = _map(2)
    right["labels"] = {"H1": "C", "H2": "U"}
    assert categorical_agreement(left, right) == 0.5
