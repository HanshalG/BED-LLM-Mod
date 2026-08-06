from __future__ import annotations

from copy import deepcopy

from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged
from scripts import number_game_two_draw_diversity_bonus_confirmation64_verify as verify


def _fixtures():
    stages = {
        "a": {
            "status": "block_b_authorized",
            "calendar_date": "2026-08-08",
            "authorization_inputs": "source mechanics gates and request count only",
            "source_science_was_not_an_authorization_input": True,
            "mechanics_gates": {
                "mechanics": True,
                "accepted_request_count_exact": True,
            },
            "usage": {
                "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
                "run_cost_usd": 4.2,
            },
        },
        "b": {
            "status": "block_complete",
            "calendar_date": "2026-08-09",
            "authorization_inputs": "source mechanics gates and request count only",
            "source_science_was_not_an_authorization_input": True,
            "mechanics_gates": {
                "mechanics": True,
                "accepted_request_count_exact": True,
            },
            "usage": {
                "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
                "run_cost_usd": 4.2,
            },
        },
    }
    comparisons = {
        "crossfit_depth_two": {
            "relative_brier_reduction": 0.05,
            "tree_bootstrap_95pct": [-0.01, -0.001],
            "wins": 30,
            "losses": 18,
        },
        "original_depth_three": {
            "changed_roots": 20,
            "mean_candidate_minus_baseline_brier": -0.001,
        },
    }
    gates = staged.scientific_gates({"comparisons": comparisons})
    blocks = {}
    rows = []
    for block, spec in staged.BLOCKS.items():
        source = {
            "protocol": {
                "interface_version": f"{staged.SOURCE_INTERFACE_PREFIX}-{block}-1",
                "tree_seeds": list(spec["tree_seeds"]),
                "target_seeds": list(spec["target_seeds"]),
                "validation_seeds": verify._expected_validation_seeds(
                    spec["validation_seed_start"]
                ),
                "bootstrap_seed": spec["source_bootstrap_seed"],
            },
            "usage": {
                "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
                "run_cost_usd": 4.2,
            },
            "mechanics_gates": {"mechanics": True},
        }
        block_rows = [
            {"tree_seed": seed, "block": block}
            for seed in spec["tree_seeds"]
        ]
        artifacts = {
            "result_sha256": f"{block}r",
            "trees_sha256": f"{block}t",
            "raw_sha256": f"{block}w",
        }
        blocks[block] = {
            "source_result": source,
            "source_artifacts": artifacts,
            "stage_source_artifacts": {"result_sha256": block},
            "rows": block_rows,
        }
        stages[block]["source_artifacts"] = {"result_sha256": block}
        rows.extend(block_rows)
    replay = {
        "blocks": blocks,
        "rows": rows,
        "comparisons": comparisons,
        "rank_metrics": {"rho": 0.4},
        "scientific_gates": gates,
    }
    result = {
        "interface_version": staged.INTERFACE_VERSION,
        "status": "passed",
        "protocol": {
            "preregistration_sha256": staged.PREREGISTRATION_SHA256,
            "blocks_were_mandatory_after_block_a_mechanics": True,
            "block_a_science_was_not_an_authorization_input": True,
            "tree_count": staged.TREE_COUNT,
            "tree_seeds": {
                block: list(staged.BLOCKS[block]["tree_seeds"])
                for block in staged.BLOCKS
            },
            "target_seeds": {
                block: list(staged.BLOCKS[block]["target_seeds"])
                for block in staged.BLOCKS
            },
            "validation_seed_starts": {
                block: staged.BLOCKS[block]["validation_seed_start"]
                for block in staged.BLOCKS
            },
            "source_bootstrap_seeds": {
                block: staged.BLOCKS[block]["source_bootstrap_seed"]
                for block in staged.BLOCKS
            },
            "combined_bootstrap_seed": staged.COMBINED_BOOTSTRAP_SEED,
            "bootstrap_samples": staged.audit.BOOTSTRAP_SAMPLES,
            "diversity_coefficient": -0.5,
            "no_coefficient_sweep_on_fresh_data": True,
            "expected_requests_total": staged.EXPECTED_REQUESTS_TOTAL,
        },
        "usage": {
            "adapter_requests": staged.EXPECTED_REQUESTS_TOTAL,
            "run_cost_usd": 8.4,
            "block_cost_usd": {"a": 4.2, "b": 4.2},
        },
        "block_stages": deepcopy(stages),
        "source_artifacts": {
            block: deepcopy(blocks[block]["source_artifacts"])
            for block in staged.BLOCKS
        },
        "rows": deepcopy(rows),
        "comparisons": deepcopy(comparisons),
        "rank_metrics": deepcopy(replay["rank_metrics"]),
        "scientific_gates": deepcopy(gates),
    }
    return result, stages, replay


def test_complete_staged_result_passes_all_checks() -> None:
    result, stages, replay = _fixtures()
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert all(checks.values())


def test_same_day_block_b_is_detected() -> None:
    result, stages, replay = _fixtures()
    stages["b"]["calendar_date"] = stages["a"]["calendar_date"]
    result["block_stages"] = deepcopy(stages)
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert not checks["block_b_date_is_later"]


def test_reused_block_tree_seed_is_detected() -> None:
    result, stages, replay = _fixtures()
    replay["blocks"]["b"]["rows"][0]["tree_seed"] = 110_000
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert not checks["block_b_actual_tree_seeds_exact"]


def test_tampered_combined_bootstrap_is_detected() -> None:
    result, stages, replay = _fixtures()
    result["comparisons"]["crossfit_depth_two"][
        "tree_bootstrap_95pct"
    ][1] = 0.1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert not checks["all_comparisons_and_bootstraps_replay_exactly"]


def test_tampered_seed_manifest_is_detected() -> None:
    result, stages, replay = _fixtures()
    result["protocol"]["target_seeds"]["b"][0] -= 1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert not checks["outer_seed_manifest_exact"]


def test_tampered_stage_mechanics_and_cost_are_detected() -> None:
    result, stages, replay = _fixtures()
    stages["a"]["mechanics_gates"]["accepted_request_count_exact"] = False
    result["block_stages"] = deepcopy(stages)
    result["usage"]["block_cost_usd"]["a"] = 4.1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
    )
    assert not checks["block_a_stage_mechanics_match_source"]
    assert not checks["combined_usage_costs_exact"]
