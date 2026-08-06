#!/usr/bin/env python3
"""Independently verify the staged 64-tree diversity confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-two-draw-diversity-bonus-confirmation64-verify-1"
BLOCK_A_INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation64-block-a-verify-1"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(source_dir: Path) -> dict[str, str]:
    return {
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "targets_sha256": audit.sha256_file(source_dir / "TARGETS.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
    }


def block_a_authorization_checks(
    *,
    stage: dict[str, Any],
    source_result: dict[str, Any],
    source_artifacts: dict[str, str],
) -> dict[str, bool]:
    spec = staged.BLOCKS["a"]
    protocol = source_result.get("protocol") or {}
    expected_mechanics = dict(source_result.get("mechanics_gates") or {})
    expected_mechanics["accepted_request_count_exact"] = (
        source_result.get("usage", {}).get("adapter_requests")
        == staged.EXPECTED_REQUESTS_PER_BLOCK
    )
    forbidden = ("brier", "comparison", "scientific", "selected_root")
    return {
        "preregistration_hash_exact": (
            audit.sha256_file(staged.PREREGISTRATION)
            == staged.PREREGISTRATION_SHA256
        ),
        "stage_interface_and_status_exact": (
            stage.get("interface_version") == staged.INTERFACE_VERSION
            and stage.get("block") == "a"
            and stage.get("status") == "block_b_authorized"
            and stage.get("calendar_date") == staged.FORMAL_BLOCK_DATES["a"]
        ),
        "authorization_is_mechanics_only": (
            stage.get("authorization_inputs")
            == "source mechanics gates and request count only"
            and stage.get("source_science_was_not_an_authorization_input")
            is True
            and not any(
                term in json.dumps(stage, sort_keys=True).lower()
                for term in forbidden
            )
        ),
        "source_protocol_seed_manifest_exact": (
            protocol.get("interface_version")
            == f"{staged.SOURCE_INTERFACE_PREFIX}-a-1"
            and protocol.get("tree_seeds") == list(spec["tree_seeds"])
            and protocol.get("target_seeds") == list(spec["target_seeds"])
            and protocol.get("validation_seeds")
            == _expected_validation_seeds(spec["validation_seed_start"])
            and protocol.get("bootstrap_seed") == spec["source_bootstrap_seed"]
            and protocol.get("staged_64_confirmation") is True
            and protocol.get("staged_block") == "a"
            and protocol.get("block_calendar_date") == stage.get("calendar_date")
        ),
        "source_requests_exact": (
            source_result.get("usage", {}).get("adapter_requests")
            == staged.EXPECTED_REQUESTS_PER_BLOCK
            == stage.get("usage", {}).get("adapter_requests")
        ),
        "stage_mechanics_match_source": (
            bool(expected_mechanics)
            and all(expected_mechanics.values())
            and stage.get("mechanics_gates") == expected_mechanics
        ),
        "stage_usage_matches_source": (
            stage.get("usage") == source_result.get("usage")
        ),
        "stage_source_hashes_match_files": (
            stage.get("source_artifacts") == source_artifacts
        ),
    }


def verify_block_a_authorization(*, run_dir: Path) -> dict[str, Any]:
    stage_path = staged.block_stage_path(run_dir, "a")
    source_dir = staged.block_directory(run_dir, "a") / "source"
    stage = _load(stage_path)
    source_result = _load(source_dir / "RESULT.json")
    source_artifacts = _source_hashes(source_dir)
    checks = block_a_authorization_checks(
        stage=stage,
        source_result=source_result,
        source_artifacts=source_artifacts,
    )
    failed = [name for name, passed in checks.items() if not passed]
    verification = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": BLOCK_A_INTERFACE_VERSION,
        "status": "verified" if not failed else "verification_failed",
        "authorization_reads_scientific_endpoints": False,
        "model_calls": 0,
        "cost_usd": 0.0,
        "checks": checks,
        "failed_checks": failed,
        "block_a_stage_sha256": audit.sha256_file(stage_path),
        "block_a_source_artifacts": source_artifacts,
    }
    checkpoint(staged.block_a_verification_path(run_dir), verification)
    if failed:
        raise ValueError(
            "Block A authorization verification failed: "
            + ", ".join(failed)
        )
    return verification


def _rank_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    original_rhos = []
    bonus_rhos = []
    original_regrets = []
    bonus_regrets = []
    for row in rows:
        roots = [int(item["root"]) for item in row["root_rows"]]
        predicted = {
            int(item["root"]): float(item["predicted_brier"])
            for item in row["root_rows"]
        }
        realized = {
            int(item["root"]): float(item["realized_brier"])
            for item in row["root_rows"]
        }
        adjusted = {
            int(root): float(value)
            for root, value in row["adjusted_scores"].items()
        }
        original_rhos.append(
            spearman_correlation(
                [predicted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        bonus_rhos.append(
            spearman_correlation(
                [adjusted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        oracle = min(realized.values())
        original_regrets.append(
            realized[int(row["original_root"])] - oracle
        )
        bonus_regrets.append(realized[int(row["bonus_root"])] - oracle)
    return {
        "original_mean_candidate_root_spearman": statistics.fmean(original_rhos),
        "bonus_mean_candidate_root_spearman": statistics.fmean(bonus_rhos),
        "original_mean_candidate_set_oracle_regret": statistics.fmean(
            original_regrets
        ),
        "bonus_mean_candidate_set_oracle_regret": statistics.fmean(
            bonus_regrets
        ),
    }


def replay_block(run_dir: Path, block: str) -> dict[str, Any]:
    source_dir = staged.block_directory(run_dir, block) / "source"
    spec = {
        "name": f"prospective_block_{block}",
        "role": "prospective_confirmation",
        "directory": source_dir,
        "tree_count": staged.BLOCK_TREE_COUNT,
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
    }
    rows = audit.load_source(
        spec,
        coefficients=(0.0, audit.DIVERSITY_COEFFICIENT),
    )
    public_rows = []
    for row in rows:
        item = dict(row)
        item.pop("coefficient_grid_roots")
        item["adjusted_scores"] = {
            str(root): float(value)
            for root, value in item["adjusted_scores"].items()
        }
        item["block"] = block
        item["source"] = f"prospective_block_{block}"
        public_rows.append(item)
    return {
        "source_result": _load(source_dir / "RESULT.json"),
        "stage_source_artifacts": staged._source_hashes(source_dir),
        "source_artifacts": {
            key: spec[key]
            for key in ("result_sha256", "trees_sha256", "raw_sha256")
        },
        "rows": public_rows,
    }


def replay_combined(run_dir: Path) -> dict[str, Any]:
    blocks = {block: replay_block(run_dir, block) for block in staged.BLOCKS}
    rows = [row for block in staged.BLOCKS for row in blocks[block]["rows"]]
    summary = audit.source_summary(
        rows,
        seed=staged.COMBINED_BOOTSTRAP_SEED,
        include_coefficient_grid=False,
    )
    dynamic_fixed_rows = audit._comparison_rows(
        rows,
        candidate_key="original_root",
        baseline_key="fixed_depth_three_root",
    )
    summary["comparisons"][
        "unadjusted_dynamic_vs_fixed_depth_three"
    ] = audit.comparison_summary(dynamic_fixed_rows) | {
        "tree_bootstrap_95pct": audit.bootstrap_comparison(
            dynamic_fixed_rows,
            seed=staged.DYNAMIC_FIXED_BOOTSTRAP_SEED,
            stratified=False,
        ),
        "candidate_policy": "unadjusted_dynamic_depth_three",
        "baseline_policy": "fixed_support_depth_three",
        "selector_independent_of_diversity_bonus": True,
        "registered_scientific_gate": False,
    }
    return {
        "blocks": blocks,
        "rows": rows,
        "comparisons": summary["comparisons"],
        "rank_metrics": _rank_metrics(rows),
        "scientific_gates": staged.scientific_gates(summary),
    }


def _expected_validation_seeds(start: int) -> list[list[int]]:
    return [
        list(range(start + index * 16, start + (index + 1) * 16))
        for index in range(staged.BLOCK_TREE_COUNT)
    ]


def verification_checks(
    *,
    result: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    replay: dict[str, Any],
    block_a_verification: dict[str, Any],
    block_a_verification_sha256: str,
) -> dict[str, bool]:
    protocol = result.get("protocol") or {}
    mechanics_pass = all(
        all((stages[block].get("mechanics_gates") or {}).values())
        for block in staged.BLOCKS
    )
    expected_status = (
        "passed"
        if mechanics_pass and all(replay["scientific_gates"].values())
        else "mechanics_failed"
        if not mechanics_pass
        else "gated_null"
    )
    checks = {
        "interface_version_exact": (
            result.get("interface_version") == staged.INTERFACE_VERSION
        ),
        "preregistration_hash_exact": (
            protocol.get("preregistration_sha256")
            == staged.PREREGISTRATION_SHA256
            == audit.sha256_file(staged.PREREGISTRATION)
        ),
        "combined_tree_count_exact": (
            protocol.get("tree_count") == staged.TREE_COUNT
            and len(replay["rows"]) == staged.TREE_COUNT
        ),
        "mandatory_block_protocol_exact": (
            protocol.get("blocks_were_mandatory_after_block_a_mechanics")
            is True
            and protocol.get("block_a_science_was_not_an_authorization_input")
            is True
        ),
        "combined_bootstrap_exact": (
            protocol.get("combined_bootstrap_seed")
            == staged.COMBINED_BOOTSTRAP_SEED
            and protocol.get("bootstrap_samples") == audit.BOOTSTRAP_SAMPLES
        ),
        "outer_seed_manifest_exact": (
            protocol.get("tree_seeds")
            == {
                block: list(staged.BLOCKS[block]["tree_seeds"])
                for block in staged.BLOCKS
            }
            and protocol.get("target_seeds")
            == {
                block: list(staged.BLOCKS[block]["target_seeds"])
                for block in staged.BLOCKS
            }
            and protocol.get("validation_seed_starts")
            == {
                block: staged.BLOCKS[block]["validation_seed_start"]
                for block in staged.BLOCKS
            }
            and protocol.get("source_bootstrap_seeds")
            == {
                block: staged.BLOCKS[block]["source_bootstrap_seed"]
                for block in staged.BLOCKS
            }
        ),
        "coefficient_exact_and_no_grid_declared": (
            protocol.get("diversity_coefficient")
            == audit.DIVERSITY_COEFFICIENT
            and protocol.get("no_coefficient_sweep_on_fresh_data") is True
        ),
        "descriptive_control_contract_exact": (
            protocol.get("descriptive_controls_reported")
            == [
                "positive_test_strategy",
                "uniform_random_candidate_root",
            ]
            and protocol.get(
                "descriptive_controls_are_not_scientific_gates"
            )
            is True
            and all(
                name in replay["comparisons"]
                for name in protocol.get("descriptive_controls_reported", [])
            )
        ),
        "selector_independent_dynamic_fixed_contract_exact": (
            protocol.get("selector_independent_dynamic_fixed_reported")
            is True
            and protocol.get(
                "selector_independent_dynamic_fixed_bootstrap_seed"
            )
            == staged.DYNAMIC_FIXED_BOOTSTRAP_SEED
            and "unadjusted_dynamic_vs_fixed_depth_three"
            in replay["comparisons"]
            and replay["comparisons"][
                "unadjusted_dynamic_vs_fixed_depth_three"
            ].get("selector_independent_of_diversity_bonus")
            is True
            and replay["comparisons"][
                "unadjusted_dynamic_vs_fixed_depth_three"
            ].get("registered_scientific_gate")
            is False
        ),
        "no_coefficient_grid_published": all(
            "coefficient_grid_roots" not in row
            for row in result.get("rows") or []
        ),
        "request_count_exact": (
            protocol.get("expected_requests_total")
            == staged.EXPECTED_REQUESTS_TOTAL
            == result.get("usage", {}).get("adapter_requests")
        ),
        "combined_usage_costs_exact": (
            result.get("usage", {}).get("block_cost_usd")
            == {
                block: float(stages[block]["usage"]["run_cost_usd"])
                for block in staged.BLOCKS
            }
            and result.get("usage", {}).get("run_cost_usd")
            == sum(
                float(stages[block]["usage"]["run_cost_usd"])
                for block in staged.BLOCKS
            )
        ),
        "block_b_date_is_later": (
            str(stages["b"]["calendar_date"])
            > str(stages["a"]["calendar_date"])
        ),
        "formal_block_dates_exact": (
            all(
                stages[block].get("calendar_date")
                == staged.FORMAL_BLOCK_DATES[block]
                for block in staged.BLOCKS
            )
        ),
        "block_a_authorization_is_mechanics_only": (
            stages["a"].get("authorization_inputs")
            == "source mechanics gates and request count only"
            and stages["a"].get("source_science_was_not_an_authorization_input")
            is True
            and not any(
                term in json.dumps(stages["a"], sort_keys=True).lower()
                for term in ("brier", "comparison", "selected_root")
            )
        ),
        "block_a_authorization_was_independently_verified_before_b": (
            block_a_verification.get("interface_version")
            == BLOCK_A_INTERFACE_VERSION
            and block_a_verification.get("status") == "verified"
            and block_a_verification.get(
                "authorization_reads_scientific_endpoints"
            )
            is False
            and bool(block_a_verification.get("checks"))
            and all(block_a_verification["checks"].values())
            and block_a_verification.get("block_a_source_artifacts")
            == stages["a"].get("source_artifacts")
            and stages["b"].get(
                "block_a_authorization_verification_sha256"
            )
            == block_a_verification_sha256
        ),
        "all_fixed_selector_rows_replay_exactly": (
            result.get("rows") == replay["rows"]
        ),
        "all_comparisons_and_bootstraps_replay_exactly": (
            result.get("comparisons") == replay["comparisons"]
        ),
        "rank_metrics_replay_exactly": (
            result.get("rank_metrics") == replay["rank_metrics"]
        ),
        "scientific_gates_replay_exactly": (
            result.get("scientific_gates") == replay["scientific_gates"]
        ),
        "final_status_replays_exactly": result.get("status") == expected_status,
    }
    for block, spec in staged.BLOCKS.items():
        source = replay["blocks"][block]["source_result"]
        source_protocol = source.get("protocol") or {}
        expected_mechanics = dict(source.get("mechanics_gates") or {})
        expected_mechanics["accepted_request_count_exact"] = (
            source.get("usage", {}).get("adapter_requests")
            == staged.EXPECTED_REQUESTS_PER_BLOCK
        )
        actual_seeds = [
            int(row["tree_seed"]) for row in replay["blocks"][block]["rows"]
        ]
        prefix = f"block_{block}"
        checks.update(
            {
                f"{prefix}_actual_tree_seeds_exact": (
                    actual_seeds == list(spec["tree_seeds"])
                ),
                f"{prefix}_source_protocol_seeds_exact": (
                    source_protocol.get("tree_seeds")
                    == list(spec["tree_seeds"])
                    and source_protocol.get("target_seeds")
                    == list(spec["target_seeds"])
                    and source_protocol.get("validation_seeds")
                    == _expected_validation_seeds(
                        spec["validation_seed_start"]
                    )
                    and source_protocol.get("bootstrap_seed")
                    == spec["source_bootstrap_seed"]
                ),
                f"{prefix}_source_interface_exact": (
                    source_protocol.get("interface_version")
                    == f"{staged.SOURCE_INTERFACE_PREFIX}-{block}-1"
                ),
                f"{prefix}_stage_embedded_exactly": (
                    result.get("block_stages", {}).get(block) == stages[block]
                ),
                f"{prefix}_source_artifacts_exact": (
                    result.get("source_artifacts", {}).get(block)
                    == replay["blocks"][block]["source_artifacts"]
                ),
                f"{prefix}_stage_source_artifacts_exact": (
                    stages[block].get("source_artifacts")
                    == replay["blocks"][block]["stage_source_artifacts"]
                ),
                f"{prefix}_request_count_exact": (
                    source.get("usage", {}).get("adapter_requests")
                    == staged.EXPECTED_REQUESTS_PER_BLOCK
                    == stages[block].get("usage", {}).get("adapter_requests")
                ),
                f"{prefix}_stage_mechanics_match_source": (
                    stages[block].get("mechanics_gates")
                    == expected_mechanics
                ),
                f"{prefix}_stage_usage_matches_source": (
                    stages[block].get("usage") == source.get("usage")
                ),
            }
        )
    return checks


def verify_completed_confirmation(*, run_dir: Path) -> dict[str, Any]:
    block_a_verification = verify_block_a_authorization(run_dir=run_dir)
    result = _load(run_dir / "RESULT.json")
    stages = {
        block: _load(staged.block_stage_path(run_dir, block))
        for block in staged.BLOCKS
    }
    replay = replay_combined(run_dir)
    checks = verification_checks(
        result=result,
        stages=stages,
        replay=replay,
        block_a_verification=block_a_verification,
        block_a_verification_sha256=audit.sha256_file(
            staged.block_a_verification_path(run_dir)
        ),
    )
    failed = [name for name, passed in checks.items() if not passed]
    verification = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not failed else "verification_failed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "checks": checks,
        "failed_checks": failed,
        "result_sha256": audit.sha256_file(run_dir / "RESULT.json"),
    }
    checkpoint(run_dir / "VERIFICATION.json", verification)
    if failed:
        raise ValueError(
            "staged-64 diversity confirmation verification failed: "
            + ", ".join(failed)
        )
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_completed_confirmation(run_dir=args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
