from types import SimpleNamespace

from scripts.number_game_qwen_crossjudge_depth_three import (
    EXPECTED_FORMAL_REQUESTS,
    FORMAL_BUDGET_USD,
    SOURCE_STUDIES,
    _load_sources,
    endpoint_seeds_for_tree,
    formal_gates,
    sha256_file,
    smoke_gates,
)


def test_sources_are_hash_bound() -> None:
    for source in SOURCE_STUDIES:
        assert sha256_file(source["result"]) == source["result_sha256"]
        assert sha256_file(source["trees"]) == source["trees_sha256"]
    assert [source["name"] for source in _load_sources()] == [
        source["name"] for source in SOURCE_STUDIES
    ]


def test_formal_endpoint_seeds_are_unique() -> None:
    groups = [endpoint_seeds_for_tree(index) for index in range(64)]
    seeds = [seed for group in groups for seed in group]
    assert len(seeds) == EXPECTED_FORMAL_REQUESTS
    assert len(set(seeds)) == EXPECTED_FORMAL_REQUESTS


def test_smoke_gates_accept_clean_supports() -> None:
    supports = [
        [
            SimpleNamespace(extension=(seed, index))
            for index in range(16)
        ]
        for seed in range(10)
    ]
    usage = {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.05,
    }
    assert all(smoke_gates(supports=supports, usage=usage).values())


def test_formal_gates_accept_clean_positive_result() -> None:
    comparison = {
        "relative_brier_reduction": 0.04,
        "mean_candidate_minus_baseline_brier": -0.01,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, -0.005],
        "brier_tree_wins": 30,
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
        "root_differences": {"crossfit_depth_two": 35},
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
    trees = [
        {
            "mechanics": {
                "endpoint_draw_count": 16,
                "minimum_endpoint_support_valid": 16,
                "total_novel_endpoint_hypotheses": 128,
            }
        }
        for _ in range(64)
    ]
    usage = {
        "adapter_requests": EXPECTED_FORMAL_REQUESTS,
        "http_attempts": EXPECTED_FORMAL_REQUESTS,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": FORMAL_BUDGET_USD - 0.01,
    }
    studies = {
        name: {
            "comparisons": {
                "crossfit_depth_two": {
                    "mean_candidate_minus_baseline_brier": -0.01
                }
            }
        }
        for name in ("one", "two")
    }
    assert all(
        formal_gates(
            scored_trees=trees,
            usage=usage,
            aggregate=aggregate,
            study_aggregates=studies,
        ).values()
    )
