from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged
from scripts import number_game_two_draw_diversity_bonus_confirmation64_verify as verify


def _write_real_block_a_fixture(run_dir: Path) -> None:
    source_dir = staged.block_directory(run_dir, "a") / "source"
    private_dir = source_dir / "private"
    private_dir.mkdir(parents=True)
    (source_dir / "TREES.json").write_text("{}", encoding="utf-8")
    (source_dir / "TARGETS.json").write_text("{}", encoding="utf-8")
    (private_dir / "RAW_RESPONSES.json").write_text("[]", encoding="utf-8")
    spec = staged.BLOCKS["a"]
    result = {
        "protocol": {
            "interface_version": f"{staged.SOURCE_INTERFACE_PREFIX}-a-1",
            "tree_seeds": list(spec["tree_seeds"]),
            "target_seeds": list(spec["target_seeds"]),
            "validation_seeds": verify._expected_validation_seeds(
                spec["validation_seed_start"]
            ),
            "bootstrap_seed": spec["source_bootstrap_seed"],
            "staged_64_confirmation": True,
            "staged_block": "a",
            "block_calendar_date": "2026-08-08",
        },
        "usage": {
            "adapter_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "run_cost_usd": 4.2,
        },
        "mechanics_gates": {"mechanics": True},
    }
    (source_dir / "RESULT.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    stage = {
        "interface_version": staged.INTERFACE_VERSION,
        "status": "block_b_authorized",
        "block": "a",
        "calendar_date": "2026-08-08",
        "authorization_inputs": (
            "source mechanics gates and request count only"
        ),
        "source_science_was_not_an_authorization_input": True,
        "mechanics_gates": {
            "mechanics": True,
            "accepted_request_count_exact": True,
        },
        "usage": deepcopy(result["usage"]),
        "source_artifacts": verify._source_hashes(source_dir),
    }
    staged.block_stage_path(run_dir, "a").write_text(
        json.dumps(stage), encoding="utf-8"
    )


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
        "positive_test_strategy": {
            "baseline_is_mean_of_two_roots_per_tree": True,
        },
        "uniform_random_candidate_root": {
            "baseline_is_mean_of_two_roots_per_tree": True,
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
    block_a_verification_sha256 = "block-a-verification-sha256"
    stages["b"]["block_a_authorization_verification_sha256"] = (
        block_a_verification_sha256
    )
    block_a_verification = {
        "interface_version": verify.BLOCK_A_INTERFACE_VERSION,
        "status": "verified",
        "authorization_reads_scientific_endpoints": False,
        "checks": {"all": True},
        "block_a_source_artifacts": deepcopy(stages["a"]["source_artifacts"]),
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
            "descriptive_controls_reported": [
                "positive_test_strategy",
                "uniform_random_candidate_root",
            ],
            "descriptive_controls_are_not_scientific_gates": True,
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
    return (
        result,
        stages,
        replay,
        block_a_verification,
        block_a_verification_sha256,
    )


def test_complete_staged_result_passes_all_checks() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert all(checks.values())


def test_same_day_block_b_is_detected() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    stages["b"]["calendar_date"] = stages["a"]["calendar_date"]
    result["block_stages"] = deepcopy(stages)
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert not checks["block_b_date_is_later"]
    assert not checks["formal_block_dates_exact"]


def test_reused_block_tree_seed_is_detected() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    replay["blocks"]["b"]["rows"][0]["tree_seed"] = 110_000
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert not checks["block_b_actual_tree_seeds_exact"]


def test_tampered_combined_bootstrap_is_detected() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    result["comparisons"]["crossfit_depth_two"][
        "tree_bootstrap_95pct"
    ][1] = 0.1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert not checks["all_comparisons_and_bootstraps_replay_exactly"]


def test_tampered_seed_manifest_is_detected() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    result["protocol"]["target_seeds"]["b"][0] -= 1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert not checks["outer_seed_manifest_exact"]


def test_tampered_stage_mechanics_and_cost_are_detected() -> None:
    result, stages, replay, authorization, authorization_sha = _fixtures()
    stages["a"]["mechanics_gates"]["accepted_request_count_exact"] = False
    result["block_stages"] = deepcopy(stages)
    result["usage"]["block_cost_usd"]["a"] = 4.1
    checks = verify.verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=authorization,
        block_a_verification_sha256=authorization_sha,
    )
    assert not checks["block_a_stage_mechanics_match_source"]
    assert not checks["combined_usage_costs_exact"]


def test_real_block_a_authorization_verifier_binds_files(tmp_path: Path) -> None:
    _write_real_block_a_fixture(tmp_path)

    verification = verify.verify_block_a_authorization(run_dir=tmp_path)

    assert verification["status"] == "verified"
    assert all(verification["checks"].values())
    assert staged.block_a_verification_path(tmp_path).exists()


def test_real_block_a_authorization_verifier_rejects_tamper(
    tmp_path: Path,
) -> None:
    _write_real_block_a_fixture(tmp_path)
    stage_path = staged.block_stage_path(tmp_path, "a")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["mechanics_gates"]["accepted_request_count_exact"] = False
    stage_path.write_text(json.dumps(stage), encoding="utf-8")

    with pytest.raises(ValueError, match="authorization verification failed"):
        verify.verify_block_a_authorization(run_dir=tmp_path)

    failure = json.loads(
        staged.block_a_verification_path(tmp_path).read_text(encoding="utf-8")
    )
    assert failure["status"] == "verification_failed"
    assert not failure["checks"]["stage_mechanics_match_source"]
