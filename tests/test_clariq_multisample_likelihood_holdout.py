from pathlib import Path

from scripts.clariq_multisample_likelihood_holdout import (
    EXPECTED_REQUESTS,
    SELECTED_TOPIC_IDS,
    VALID_ROOT_COUNTS,
    apply_holdout_gates,
    exact_sign_flip_p,
    load_manifest,
)


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results/nonmyopic/clariq_multisample_likelihood_holdout_manifest/MANIFEST.json"
)


def test_frozen_holdout_manifest_reproduces() -> None:
    tasks = load_manifest(MANIFEST)
    assert tuple(tasks) == SELECTED_TOPIC_IDS
    assert tuple(len(task["questions"]) for task in tasks.values()) == (
        VALID_ROOT_COUNTS
    )
    assert sum(len(task["questions"]) for task in tasks.values()) * 5 == (
        EXPECTED_REQUESTS
    )


def test_exact_sign_flip_p() -> None:
    assert exact_sign_flip_p([1.0, 1.0, 1.0, 1.0]) == 0.0625
    assert exact_sign_flip_p([0.0, 0.0]) == 1.0


def test_missing_endpoint_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.clariq_multisample_likelihood_holdout._selected",
        lambda maps, indices: "depth",
    )
    rows = [
        {
            "topic_id": topic_id,
            "myopic_oracle_tail": 0.1,
            "depth_two_oracle_tail": (
                None if topic_id == SELECTED_TOPIC_IDS[0] else 0.2
            ),
            "random_oracle_tail": 0.1,
            "distinct_modal_partition_count": 3,
            "myopic_eig_range": 0.1,
        }
        for topic_id in SELECTED_TOPIC_IDS
    ]
    payload = {
        "rows": rows,
        "maps": {
            topic_id: {"root": ["Y", "N", "U", "Y", "N"]}
            for topic_id in SELECTED_TOPIC_IDS
        },
        "policies": {
            topic_id: {
                "depth_two_question_id": "depth",
                "myopic_question_id": "myopic",
            }
            for topic_id in SELECTED_TOPIC_IDS
        },
        "usage": {
            "physical_requests": EXPECTED_REQUESTS,
            "http_attempts": EXPECTED_REQUESTS,
            "retry_count": 0,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        },
        "metrics": {
            "depth_two_root_change_count": len(SELECTED_TOPIC_IDS),
            "depth_two_wins_over_myopic": len(SELECTED_TOPIC_IDS) - 1,
            "depth_two_losses_to_myopic": 0,
            "mean_depth_two_gain_over_myopic": 0.1,
            "mean_depth_two_gain_over_random": 0.1,
            "pooled_depth_two_score_oracle_tail_spearman": 0.5,
        },
    }

    result = apply_holdout_gates(payload)

    assert result["status"] == "gate_failed"
    assert not result["gates"]["all_selected_roots_have_endpoints"]
    assert not result["gates"]["exact_one_sided_p_at_most_0_10"]
    assert result["holdout_statistics"]["exact_one_sided_sign_flip_p"] == 1.0
    assert (
        result["holdout_statistics"]["paired_differences"][
            SELECTED_TOPIC_IDS[0]
        ]
        is None
    )
