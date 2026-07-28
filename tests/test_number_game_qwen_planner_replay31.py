import json

from scripts.number_game_qwen_planner_replay31 import (
    RAW_PATH,
    RAW_SHA256,
    RUN_LOG_PATH,
    RUN_LOG_SHA256,
    ReplayAdapter,
    development_gates,
    sha256_file,
)


def test_replay_sources_are_hash_bound() -> None:
    assert sha256_file(RAW_PATH) == RAW_SHA256
    assert sha256_file(RUN_LOG_PATH) == RUN_LOG_SHA256
    assert len(json.loads(RAW_PATH.read_text())["trees"]) == 31


def test_replay_adapter_preserves_batches() -> None:
    adapter = ReplayAdapter((("one",), ("two", "three")))
    assert adapter.chat_complete_messages_batched_structured([{}]) == [
        "one"
    ]
    assert adapter.chat_complete_messages_batched_structured(
        [{}, {}]
    ) == ["two", "three"]
    assert not adapter.batches


def test_development_gates_accept_positive_shape() -> None:
    comparison = {
        "relative_brier_reduction": 0.08,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.03, -0.01],
        "brier_tree_wins": 20,
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }
    aggregate = {
        "comparisons": {
            key: dict(comparison)
            for key in (
                "crossfit_depth_two",
                "myopic_eig",
                "fixed_support_depth_three",
                "positive_test_strategy",
            )
        },
        "root_differences": {"crossfit_depth_two": 20},
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": -0.01,
            "candidate_minus_baseline_hamming": -0.01,
            "coverage_difference": 0.01,
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.9},
            "crossfit_depth_two_spearman_brier": {"mean": 0.5},
        },
    }
    assert all(development_gates(aggregate).values())
