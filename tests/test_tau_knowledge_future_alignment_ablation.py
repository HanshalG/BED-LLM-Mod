from __future__ import annotations

import copy
import json

import pytest

from scripts.tau_knowledge_future_alignment_ablation import (
    blinding_assignments,
    derange_future_subtrees,
    paired_scorer_messages,
    parse_paired_scores,
    unblind_scores,
    validate_intervention,
)


def _records(count: int = 2) -> list[dict]:
    records = []
    for task_index in range(count):
        branches = []
        for root_index in range(5):
            branches.append(
                {
                    "query": f"root {root_index}",
                    "first_results": [
                        {
                            "id": f"first-{root_index}",
                            "title": f"First {root_index}",
                            "content": f"First content {root_index}",
                        }
                    ],
                    "refreshed_information_need_hypotheses": [
                        f"need {root_index}"
                    ],
                    "followups": [
                        {
                            "query": f"follow {root_index} {follow_index}",
                            "results": [
                                {
                                    "id": (
                                        f"future-{root_index}-{follow_index}"
                                    ),
                                    "title": (
                                        f"Future {root_index} {follow_index}"
                                    ),
                                    "content": (
                                        f"Future content {root_index} "
                                        f"{follow_index}"
                                    ),
                                }
                            ],
                        }
                        for follow_index in range(4)
                    ],
                }
            )
        records.append(
            {
                "task_id": f"task-{task_index}",
                "opening": "opening",
                "initial_information_need_hypotheses": ["initial"],
                "first_branches": branches,
            }
        )
    return records


def _response() -> str:
    return json.dumps(
        {
            "condition_a_scores": ["10", "20", "30", "40", "50"],
            "condition_a_best_followups": ["1", "2", "3", "4", "1"],
            "condition_b_scores": ["50", "40", "30", "20", "10"],
            "condition_b_best_followups": ["4", "3", "2", "1", "4"],
        }
    )


def test_future_derangement_preserves_only_subtree_multiset() -> None:
    original = _records()
    transformed, diagnostics = derange_future_subtrees(original)
    intervention = validate_intervention(
        original, transformed, diagnostics
    )
    assert intervention["all_permutations_are_derangements"]
    assert intervention["all_future_subtree_multisets_preserved"]
    assert intervention["nonfuture_fields_unchanged"]
    for before, after in zip(original, transformed, strict=True):
        for root_index in range(5):
            before_branch = before["first_branches"][root_index]
            after_branch = after["first_branches"][root_index]
            assert before_branch["query"] == after_branch["query"]
            assert (
                before_branch["first_results"]
                == after_branch["first_results"]
            )
            assert (
                before_branch["followups"] != after_branch["followups"]
            )


def test_future_derangement_does_not_mutate_source() -> None:
    original = _records()
    snapshot = copy.deepcopy(original)
    derange_future_subtrees(original)
    assert original == snapshot


def test_blinding_is_balanced_and_deterministic() -> None:
    records = _records(20)
    first = blinding_assignments(records)
    second = blinding_assignments(records)
    assert first == second
    assert sum(row["condition_a"] == "aligned" for row in first) == 10
    assert all(
        {row["condition_a"], row["condition_b"]}
        == {"aligned", "shuffled"}
        for row in first
    )


def test_blinding_balances_two_task_smoke() -> None:
    assignments = blinding_assignments(_records(2))
    assert sum(
        row["condition_a"] == "aligned" for row in assignments
    ) == 1


def test_paired_prompt_does_not_reveal_condition_labels() -> None:
    original = _records(1)
    transformed, _diagnostics = derange_future_subtrees(original)
    messages = paired_scorer_messages(
        original[0],
        transformed[0],
        {
            "condition_a": "shuffled",
            "condition_b": "aligned",
        },
    )
    user_text = messages[1]["content"]
    assert '"condition_a"' in user_text
    assert '"condition_b"' in user_text
    assert "aligned" not in user_text
    assert "shuffled" not in user_text


def test_parse_paired_scores_accepts_canonical_arrays() -> None:
    parsed = parse_paired_scores(_response())
    assert parsed["condition_a"]["scores"] == [10, 20, 30, 40, 50]
    assert parsed["condition_a"]["best_followup_indices"] == [0, 1, 2, 3, 0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("condition_a_scores", ["01", "20", "30", "40", "50"]),
        ("condition_b_scores", ["101", "40", "30", "20", "10"]),
        ("condition_a_best_followups", ["0", "2", "3", "4", "1"]),
        ("condition_b_best_followups", ["4", "3", "2", "1"]),
    ],
)
def test_parse_paired_scores_rejects_invalid_arrays(field, value) -> None:
    payload = json.loads(_response())
    payload[field] = value
    with pytest.raises(ValueError):
        parse_paired_scores(json.dumps(payload))


def test_parse_paired_scores_rejects_extra_text() -> None:
    with pytest.raises(ValueError, match="extra data"):
        parse_paired_scores(_response() + " trailing")


def test_unblind_scores_uses_hidden_assignment() -> None:
    parsed = parse_paired_scores(_response())
    aligned, shuffled = unblind_scores(
        parsed,
        {
            "condition_a": "shuffled",
            "condition_b": "aligned",
        },
    )
    assert aligned["scores"] == [50, 40, 30, 20, 10]
    assert shuffled["scores"] == [10, 20, 30, 40, 50]
