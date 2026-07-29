from __future__ import annotations

import copy

from scripts import number_game_qwen_external_canonical_confirmation as confirm


def _comparison(
    *,
    relative: float,
    upper: float,
    wins: int,
) -> dict[str, float | int | list[float]]:
    return {
        "relative_brier_reduction": relative,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, upper],
        "brier_tree_wins": wins,
        "mean_candidate_minus_baseline_brier": -0.01,
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }


def _aggregate() -> dict:
    return {
        "root_differences": {"crossfit_depth_two": 16},
        "comparisons": {
            "crossfit_depth_two": _comparison(
                relative=0.02,
                upper=-0.001,
                wins=14,
            ),
            "myopic_eig": _comparison(
                relative=0.08,
                upper=-0.003,
                wins=20,
            ),
            "fixed_support_depth_three": _comparison(
                relative=0.03,
                upper=0.001,
                wins=15,
            ),
            "positive_test_strategy": _comparison(
                relative=0.04,
                upper=0.001,
                wins=17,
            ),
            "uniform_random_candidate_root": _comparison(
                relative=0.09,
                upper=-0.002,
                wins=22,
            ),
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.4},
            "crossfit_depth_two_spearman_brier": {"mean": 0.2},
        },
    }


def test_primary_gates_match_frozen_thresholds() -> None:
    aggregate = _aggregate()
    assert all(confirm.primary_gates(aggregate).values())

    failed = copy.deepcopy(aggregate)
    failed["comparisons"]["crossfit_depth_two"][
        "tree_cluster_brier_difference_95pct_bootstrap"
    ][1] = 0.0
    gates = confirm.primary_gates(failed)
    assert not gates["depth_three_vs_depth_two_ci_below_zero"]


def test_diagnostic_gates_do_not_enter_primary_gate_set() -> None:
    aggregate = _aggregate()
    aggregate["comparisons"]["fixed_support_depth_three"][
        "mean_candidate_minus_baseline_brier"
    ] = 0.01

    assert all(confirm.primary_gates(aggregate).values())
    assert not confirm.diagnostic_gates(aggregate)[
        "depth_three_directionally_beats_fixed_support"
    ]


def test_seed_schedule_is_fresh_and_nonoverlapping() -> None:
    validation = {
        seed
        for index in range(confirm.TREE_COUNT)
        for seed in confirm.validation_seeds_for_tree(index)
    }
    assert len(validation) == confirm.TREE_COUNT * 8
    assert not validation.intersection(confirm.TREE_SEEDS)
    assert not validation.intersection(confirm.TARGET_SEEDS)
    assert confirm.EXPECTED_REQUESTS == 1856


def test_hash_bound_smoke_has_more_than_ten_clean_calls() -> None:
    smoke = confirm.validate_smoke_result(confirm.SMOKE_RESULT)
    assert smoke["usage"]["adapter_requests"] == 73
    assert smoke["usage"]["retry_count"] == 0
    assert smoke["usage"]["provider_error_retries"] == 0
