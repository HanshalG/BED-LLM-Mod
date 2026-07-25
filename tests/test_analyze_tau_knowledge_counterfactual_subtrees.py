from __future__ import annotations

import copy

from scripts.analyze_tau_knowledge_counterfactual_subtrees import (
    _condition_metrics,
    _permutations_match,
)


def _record() -> dict:
    branches = []
    for root_index in range(5):
        branches.append(
            {
                "query": f"root-{root_index}",
                "first_results": [
                    {
                        "id": f"first-{root_index}",
                        "title": "first",
                        "content": "first",
                    }
                ],
                "refreshed_information_need_hypotheses": [
                    f"need-{root_index}"
                ],
                "followups": [
                    {
                        "query": f"follow-{root_index}-{followup_index}",
                        "results": [
                            {
                                "id": (
                                    f"required-{root_index}"
                                    if followup_index == root_index % 4
                                    else f"noise-{root_index}-{followup_index}"
                                ),
                                "title": "future",
                                "content": "future",
                            }
                        ],
                    }
                    for followup_index in range(4)
                ],
            }
        )
    return {
        "task_id": "fixture",
        "opening": "opening",
        "required_documents": [
            "required-1",
            "required-3",
            "first-4",
        ],
        "initial_information_need_hypotheses": ["initial"],
        "first_branches": branches,
    }


def test_condition_metrics_scores_roots_and_declared_followups():
    record = _record()
    scores = [
        {
            "scores": [0, 80, 10, 70, 60],
            "best_followup_indices": [0, 1, 0, 3, 0],
        }
    ]

    result = _condition_metrics([record], scores)

    assert result["root_pairwise_accuracy"] == 1.0
    assert result["selected_root_oracle_tail_total"] == 1
    assert result["selected_pair_total"] == 1
    assert result["best_followup_optimal_count"] == 5
    assert result["best_followup_total_regret"] == 0


def test_condition_metrics_uses_counterfactual_documents():
    record = _record()
    moved = copy.deepcopy(record)
    moved["first_branches"][0]["followups"] = copy.deepcopy(
        record["first_branches"][3]["followups"]
    )
    scores = [
        {
            "scores": [90, 20, 10, 30, 0],
            "best_followup_indices": [3, 1, 0, 3, 0],
        }
    ]

    original = _condition_metrics([record], scores)
    counterfactual = _condition_metrics([moved], scores)

    assert original["selected_pair_total"] == 0
    assert counterfactual["selected_pair_total"] == 1


def test_permutation_match_is_exact_and_ordered():
    generated = [
        {
            "task_id": "task",
            "source_branch_for_target": [1, 2, 3, 4, 0],
            "is_derangement": True,
            "future_subtree_multiset_preserved": True,
        }
    ]

    assert _permutations_match(generated, generated)
    changed = copy.deepcopy(generated)
    changed[0]["source_branch_for_target"] = [2, 3, 4, 0, 1]
    assert not _permutations_match(generated, changed)
