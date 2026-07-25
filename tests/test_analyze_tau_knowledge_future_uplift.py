from __future__ import annotations

from scripts.analyze_tau_knowledge_future_uplift import (
    _summarize_rows,
    _task_row,
)


def _record(task_id: str = "fixture") -> dict:
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
                "refreshed_information_need_hypotheses": ["need"],
                "followups": [
                    {
                        "query": f"follow-{root_index}-{followup_index}",
                        "results": [
                            {
                                "id": (
                                    f"future-{root_index}"
                                    if followup_index == 0
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
        "task_id": task_id,
        "opening": "opening",
        "required_documents": [
            "first-0",
            "first-1",
            "future-1",
            "future-2",
            "future-3",
            "future-4",
        ],
        "initial_information_need_hypotheses": ["initial"],
        "first_branches": branches,
    }


def test_task_row_uses_full_minus_myopic_for_incremental_gain():
    myopic = {"scores": [90, 80, 70, 60, 50]}
    full = {"scores": [90, 100, 100, 100, 100]}

    row = _task_row(_record(), myopic, full)

    assert row["immediate_values"] == [1, 1, 0, 0, 0]
    assert row["total_values"] == [1, 2, 1, 1, 1]
    assert row["future_gain_values"] == [0, 1, 1, 1, 1]
    assert row["uplift_scores"] == [0, 20, 30, 40, 50]
    assert row["uplift_future_gain_points"] == 4.0
    assert row["future_gain_comparable"] == 4
    assert row["uplift_root_index"] == 4
    assert row["uplift_selected_future_gain"] == 1


def test_summary_uses_task_clustered_sign_flip_and_totals():
    row = _task_row(
        _record(),
        {"scores": [90, 80, 70, 60, 50]},
        {"scores": [90, 100, 100, 100, 100]},
    )
    summary = _summarize_rows([row])

    assert summary["future_gain_comparable_pairs"] == 4
    assert summary["uplift_future_gain_pairwise_accuracy"] == 1.0
    assert summary["uplift_selected_future_gain_total"] == 1
    assert summary["tasks_with_comparable_future_gain"] == 1
