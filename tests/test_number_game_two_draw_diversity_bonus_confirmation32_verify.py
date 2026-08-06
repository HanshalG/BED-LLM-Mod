from __future__ import annotations

from copy import deepcopy

from scripts import number_game_two_draw_diversity_bonus_confirmation32 as confirmation
from scripts import number_game_two_draw_diversity_bonus_confirmation32_verify as verify


def _comparison(*, difference: float, reduction: float, wins: int, losses: int, changed: int):
    return {
        "mean_candidate_minus_baseline_brier": difference,
        "relative_brier_reduction": reduction,
        "wins": wins,
        "ties": 32 - wins - losses,
        "losses": losses,
        "changed_roots": changed,
        "tree_bootstrap_95pct": [difference - 0.002, difference + 0.002],
    }


def _fixtures():
    comparisons = {
        "crossfit_depth_two": _comparison(
            difference=-0.006,
            reduction=0.05,
            wins=16,
            losses=8,
            changed=24,
        ),
        "original_depth_three": _comparison(
            difference=-0.001,
            reduction=0.01,
            wins=8,
            losses=4,
            changed=12,
        ),
    }
    replay = {
        "source_artifacts": {
            "result_sha256": "a",
            "trees_sha256": "b",
            "raw_sha256": "c",
        },
        "comparisons": comparisons,
        "rank_metrics": {"rho": 0.4},
        "rows": [
            {"tree_seed": tree_seed}
            for tree_seed in confirmation.TREE_SEEDS
        ],
    }
    validation_seeds = [
        list(
            range(
                confirmation.VALIDATION_SEED_START + tree_index * 16,
                confirmation.VALIDATION_SEED_START + (tree_index + 1) * 16,
            )
        )
        for tree_index in range(confirmation.TREE_COUNT)
    ]
    source = {
        "status": "gated_null",
        "protocol": {
            "interface_version": confirmation.SOURCE_INTERFACE_VERSION,
            "tree_count": confirmation.TREE_COUNT,
            "tree_seeds": list(confirmation.TREE_SEEDS),
            "target_seeds": list(confirmation.TARGET_SEEDS),
            "validation_draws_per_tree": 16,
            "validation_seeds": validation_seeds,
            "bootstrap_seed": confirmation.BOOTSTRAP_SEED,
        },
        "usage": {
            "adapter_requests": confirmation.EXPECTED_REQUESTS,
            "run_cost_usd": 4.2,
        },
        "mechanics_gates": {"mechanics": True},
    }
    result = {
        "interface_version": confirmation.INTERFACE_VERSION,
        "status": "passed",
        "protocol": {
            "preregistration_sha256": confirmation.PREREGISTRATION_SHA256,
            "tree_seeds": list(confirmation.TREE_SEEDS),
            "target_seeds": list(confirmation.TARGET_SEEDS),
            "validation_seed_start": confirmation.VALIDATION_SEED_START,
            "bootstrap_seed": confirmation.BOOTSTRAP_SEED,
            "bootstrap_samples": confirmation.audit.BOOTSTRAP_SAMPLES,
            "diversity_coefficient": -0.5,
            "no_coefficient_sweep_on_fresh_data": True,
            "accepted_requests_expected": confirmation.EXPECTED_REQUESTS,
        },
        "usage": deepcopy(source["usage"]),
        "mechanics_gates": {
            "mechanics": True,
            "accepted_request_count_exact": True,
        },
        "source_artifacts": deepcopy(replay["source_artifacts"]),
        "comparisons": deepcopy(comparisons),
        "rank_metrics": deepcopy(replay["rank_metrics"]),
        "rows": deepcopy(replay["rows"]),
        "scientific_gates": confirmation.scientific_gates(
            {"comparisons": comparisons}
        ),
    }
    return result, source, replay


def test_complete_result_replays_all_checks() -> None:
    result, source, replay = _fixtures()
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert all(checks.values())


def test_tampered_fixed_selector_row_is_detected() -> None:
    result, source, replay = _fixtures()
    result["rows"][0]["tree_seed"] += 1
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert not checks["all_fixed_selector_rows_replay_exactly"]


def test_reused_actual_source_tree_is_detected() -> None:
    result, source, replay = _fixtures()
    replay["rows"][0]["tree_seed"] -= 1
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert not checks["actual_source_tree_seeds_exact"]


def test_reused_source_protocol_seeds_are_detected() -> None:
    result, source, replay = _fixtures()
    source["protocol"]["tree_seeds"][0] -= 1
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert not checks["source_protocol_tree_seeds_exact"]


def test_published_coefficient_grid_is_detected() -> None:
    result, source, replay = _fixtures()
    result["rows"][0]["coefficient_grid_roots"] = {"1.0": 7}
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert not checks["no_coefficient_grid_published"]


def test_tampered_bootstrap_or_gate_is_detected() -> None:
    result, source, replay = _fixtures()
    result["comparisons"]["crossfit_depth_two"][
        "tree_bootstrap_95pct"
    ][1] = 0.1
    result["scientific_gates"][
        "depth_three_vs_depth_two_interval_below_zero"
    ] = False
    checks = verify.verification_checks(
        result=result,
        source_result=source,
        replay=replay,
    )
    assert not checks["all_comparisons_and_bootstraps_replay_exactly"]
    assert not checks["scientific_gates_replay_exactly"]
