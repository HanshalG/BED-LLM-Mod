import math

from scripts.tau_knowledge_execution_replication_analysis import (
    _clustered_rows,
    exact_sign_flip_p,
    holm_two,
    retrieval_metrics,
)


def test_retrieval_metrics_deduplicate_and_discount_rank():
    metrics = retrieval_metrics(
        ["required_a", "required_b"],
        ["noise", "required_a", "required_a", "required_b"],
    )
    assert metrics["count"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 2 / 3
    assert metrics["reciprocal_rank"] == 0.5
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = 1 + 1 / math.log2(3)
    assert metrics["ndcg"] == expected_dcg / ideal_dcg


def test_exact_sign_flip_uses_task_as_unit():
    assert exact_sign_flip_p([1.0, 1.0, 1.0]) == 1 / 8
    assert exact_sign_flip_p([1.0, -1.0]) == 0.75


def test_clustered_rows_average_replicates_within_task():
    first = {
        "task": {
            "nonmyopic": {"count": 2.0},
            "myopic": {"count": 0.0},
        }
    }
    second = {
        "task": {
            "nonmyopic": {"count": 0.0},
            "myopic": {"count": 2.0},
        }
    }
    clustered = _clustered_rows(first, second)
    assert clustered["task"]["nonmyopic"]["count"] == 1.0
    assert clustered["task"]["myopic"]["count"] == 1.0


def test_holm_two_controls_declared_secondary_family():
    result = holm_two({"ndcg": 0.02, "reciprocal_rank": 0.04})
    assert result["holm_rejected_at_0_05"] == {
        "ndcg": True,
        "reciprocal_rank": True,
    }
    result = holm_two({"ndcg": 0.03, "reciprocal_rank": 0.001})
    assert result["holm_rejected_at_0_05"] == {
        "ndcg": True,
        "reciprocal_rank": True,
    }
