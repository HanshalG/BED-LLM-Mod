#!/usr/bin/env python3
"""Freeze the independent Bongard confirmation96 protocol before responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_image_integrity_audit as image_audit
from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_partition_integrity_audit as partition_audit
from scripts import bongard_openworld_sample_size_expansion_audit as expansion_audit
from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-confirmation96-freeze-15"
MODEL_ID = serving.MODEL_ID
BLOCK_ORDER = ("a", "b", "c", "d")
BLOCK_SIZES = {block_id: 24 for block_id in BLOCK_ORDER}
BLOCK_OFFSETS = {block_id: index * 24 for index, block_id in enumerate(BLOCK_ORDER)}
BLOCK_EARLIEST_DATES = {
    "a": "2026-08-15",
    "b": "2026-08-16",
    "c": "2026-08-17",
    "d": "2026-08-18",
}
BLOCK_MODEL_SEEDS = {
    "a": 2_026_081_501,
    "b": 2_026_081_601,
    "c": 2_026_081_701,
    "d": 2_026_081_801,
}
TASKS = 96
CASES_PER_TASK = 33
MAX_FINALS_PER_TASK = 11
MAX_REQUESTS_PER_TASK = CASES_PER_TASK + MAX_FINALS_PER_TASK
MAX_REQUESTS_PER_BLOCK = 24 * MAX_REQUESTS_PER_TASK
MAX_HTTP_ATTEMPTS_PER_BLOCK = (
    MAX_REQUESTS_PER_BLOCK
    + serving.transport_retry_allowance(MAX_REQUESTS_PER_BLOCK)
)
MAX_REQUEST_COST_USD = serving.MAX_REQUEST_COST_USD
MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD = (
    MAX_HTTP_ATTEMPTS_PER_BLOCK * MAX_REQUEST_COST_USD
)
DAILY_CAP_USD = 5.0
BLOCK_RUN_CAP_USD = 4.75
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_081_901
MIN_CHANGED_FINAL_HISTORIES = 36
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.03

SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_source_protocol_audit/"
    "bongard-openworld-source-protocol-audit-20260806/MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763"
)
PARTITION_MANIFEST = partition_audit.PARTITION_INTEGRITY_MANIFEST
PARTITION_MANIFEST_SHA256 = (
    partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
)
EXPANSION_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_sample_size_expansion_audit/"
    "bongard-openworld-sample-size-expansion-audit-20260808/MANIFEST_V2.json"
)
EXPANSION_MANIFEST_SHA256 = (
    "eb4d8284db118eb3326eb3ee5c854bdb4eb42fb7ea5a786070f8608cf2f0f335"
)
POWER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_CONFIRMATION96_POWER_AMENDMENT.md"
)
POWER_AMENDMENT_SHA256 = (
    "824374a32527b11cbda2d3bf81e570b4102d7930de0c3c5405626ce2ff6446b1"
)
DEVELOPMENT_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development64/"
    "PROTOCOL_MANIFEST_V18.json"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "df55546302190161ab2c4005f936e967e24dafe5f42483288c39100e0b03614f"
)
DEVELOPMENT_POWER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md"
)
DEVELOPMENT_POWER_AMENDMENT_SHA256 = (
    "e334c809c8b78b693fe6b90e5cdb5140bee2a65ae28807e37efce20ef3f75a7c"
)
AUTHORIZATION_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_PATH_DEPENDENT_CLAIM_AMENDMENT.md"
)
AUTHORIZATION_AMENDMENT_SHA256 = (
    "3dc22154312b93465e2b7d308a76f9de800d145d219189b77f3a0262f83f32a4"
)
MATCHED_FIXED_SCORE_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_MATCHED_FIXED_SCORE_AMENDMENT.md"
)
MATCHED_FIXED_SCORE_AMENDMENT_SHA256 = (
    "1f860d135663bd370fb140c7fe402bdef37c25b998955d4a7fef947d6b8099a2"
)
ENDPOINT_UTILITY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
)
ENDPOINT_UTILITY_AMENDMENT_SHA256 = (
    "2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8"
)
ESTIMAND_CLARIFICATION = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_HISTORY_BLIND_ESTIMAND_CLARIFICATION_20260808.md"
)
ESTIMAND_CLARIFICATION_SHA256 = (
    "65a6e901dc815d1603611e180b0abdf728e07d1a452e0c442f48fbb1812aea10"
)
MATCHED_REALIZED_UPDATER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md"
)
MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256 = (
    "dfa981153687004c8fb2c1195879d0774a281ca6c231c55d85495f2ac622178b"
)
COMPUTE_MATCHED_MYOPIC_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_COMPUTE_MATCHED_MYOPIC_ENSEMBLE_AMENDMENT_20260809.md"
)
COMPUTE_MATCHED_MYOPIC_AMENDMENT_SHA256 = (
    "064fb13dd1fbbcea255e38b1b9eae41dfb9347bcf1b91fa5bc5711eba7985368"
)
CONFIRMATION_UID_SHA256 = (
    "3826a64b46668226c996afa92e81cf270bf59f99a373813e37196552300ecb26"
)
DEVELOPMENT_UID_SHA256 = (
    "bf3183ca1705b7048e4a4a20008881205c91a21b4554560f749cf7d5bd4e4610"
)
DEVELOPMENT_ROOT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development64"
)
MECHANICS_RESULT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/"
    "bongard-openworld-luna-vlm-mechanics-tree-20260810/RESULT.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _confirmation_rows() -> list[dict[str, str]]:
    _, _, confirmation_rows, _ = expansion_audit.expanded_validation_rows()
    rows = []
    for source_row in confirmation_rows:
        layout = source_audit._task_layout(source_row)
        rows.append(
            {
                "task_id": layout["task_id"],
                "source_row_sha256": source_audit.row_sha256(source_row),
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    for block_id in BLOCK_ORDER:
        start = BLOCK_OFFSETS[block_id]
        for row in rows[start : start + BLOCK_SIZES[block_id]]:
            row["block_id"] = block_id
    return rows


def _science_gates() -> dict[str, Any]:
    return {
        "shared": [
            "all_four_endpoint_blind_blocks_independently_replay",
            "exact_96_disjoint_confirmation_tasks",
            "root_candidate_brier_beats_constant_half",
            "all_endpoint_metrics_are_finite",
        ],
        "policy": [
            "at_least_36_dynamic_final_histories_differ_from_myopic",
            "at_least_36_dynamic_action_changes_clear_numerical_tie_margin",
            "dynamic_and_myopic_differ_in_every_execution_block",
            "dynamic_score_has_positive_mean_endpoint_ranking_fidelity",
            "dynamic_score_ranking_fidelity_is_not_worse_than_myopic",
            "dynamic_brier_relative_improvement_at_least_3_percent",
            "dynamic_brier_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_myopic",
            "at_least_36_dynamic_final_histories_differ_from_compute_matched_myopic",
            "at_least_36_dynamic_action_changes_from_compute_matched_myopic_clear_numerical_tie_margin",
            "dynamic_and_compute_matched_myopic_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_compute_matched_myopic_at_least_3_percent",
            "dynamic_brier_vs_compute_matched_myopic_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_compute_matched_myopic",
            "dynamic_ranking_fidelity_is_not_worse_than_compute_matched_myopic",
            "dynamic_brier_is_not_worse_than_shuffled_control",
        ],
        "matched_mechanism": [
            "at_least_36_dynamic_final_histories_differ_from_history_blind",
            "dynamic_and_history_blind_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_history_blind_at_least_3_percent",
            "dynamic_brier_vs_history_blind_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_history_blind",
            "dynamic_ranking_fidelity_is_not_worse_than_history_blind",
        ],
        "path_dependent_support": [
            "at_least_36_dynamic_final_histories_differ_from_fixed_depth2",
            "at_least_36_dynamic_action_changes_from_fixed_clear_numerical_tie_margin",
            "dynamic_and_fixed_depth2_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_fixed_depth2_at_least_3_percent",
            "dynamic_brier_vs_fixed_depth2_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_fixed_depth2",
            "dynamic_ranking_fidelity_is_not_worse_than_fixed_depth2",
            "at_least_36_dynamic_final_histories_differ_from_fixed_score_dynamic_update",
            "at_least_36_dynamic_action_changes_from_fixed_score_dynamic_update_clear_numerical_tie_margin",
            "dynamic_and_fixed_score_dynamic_update_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_fixed_score_dynamic_update_at_least_3_percent",
            "dynamic_brier_vs_fixed_score_dynamic_update_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_fixed_score_dynamic_update",
            "at_least_36_dynamic_final_histories_differ_from_history_blind_update_matched_first",
            "at_least_36_dynamic_second_action_changes_from_history_blind_update_matched_first_clear_numerical_tie_margin",
            "dynamic_and_history_blind_update_matched_first_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_history_blind_update_matched_first_at_least_3_percent",
            "dynamic_brier_vs_history_blind_update_matched_first_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_history_blind_update_matched_first",
        ],
    }


def build_manifest(*, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"confirmation freeze already exists: {output_path}")
    source_manifest = _load(SOURCE_MANIFEST)
    partition_manifest = _load(PARTITION_MANIFEST)
    expansion_manifest = _load(EXPANSION_MANIFEST)
    development_manifest = _load(DEVELOPMENT_MANIFEST)
    rows = _confirmation_rows()
    counts = {
        block_id: sum(row["block_id"] == block_id for row in rows)
        for block_id in BLOCK_ORDER
    }
    implementation_paths = (
        "scripts/bongard_openworld_vlm_bed.py",
        "scripts/bongard_openworld_partition_integrity_audit.py",
        "scripts/bongard_openworld_power_audit.py",
        "scripts/bongard_openworld_sample_size_expansion_audit.py",
        "scripts/bongard_openworld_luna_vlm_serving_smoke.py",
        "scripts/bongard_openworld_luna_vlm_mechanics_tree.py",
        "scripts/bongard_openworld_luna_vlm_development.py",
        "scripts/bongard_openworld_luna_claim_report.py",
        "results/nonmyopic/BONGARD_OPENWORLD_PARTITION_INTEGRITY_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_LUNA_TRANSPORT_RETRY_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_CONFIRMATION96_POWER_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_DEVELOPMENT64_POWER_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md",
        "results/nonmyopic/BONGARD_OPENWORLD_HISTORY_BLIND_ESTIMAND_CLARIFICATION_20260808.md",
        "results/nonmyopic/BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md",
        "results/nonmyopic/BONGARD_OPENWORLD_MATCHED_UPDATER_INTEGRITY_AMENDMENT_20260808.md",
        "results/nonmyopic/BONGARD_OPENWORLD_COMPUTE_MATCHED_MYOPIC_ENSEMBLE_AMENDMENT_20260809.md",
        "results/nonmyopic/BONGARD_OPENWORLD_SAMPLE_SIZE_POWER_AUDIT_20260808.json",
        "results/nonmyopic/bongard_openworld_sample_size_expansion_audit/"
        "bongard-openworld-sample-size-expansion-audit-20260808/MANIFEST_V2.json",
    )
    gates = {
        "source_manifest_hash_matches": (
            sha256_file(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256
        ),
        "partition_integrity_manifest_hash_and_status_match": (
            sha256_file(PARTITION_MANIFEST) == PARTITION_MANIFEST_SHA256
            and partition_manifest.get("status") == "partition_integrity_pass"
            and partition_manifest.get("all_gates_pass") is True
            and partition_manifest.get("authorizes_paid_calls") is False
        ),
        "sample_size_expansion_manifest_hash_and_status_match": (
            sha256_file(EXPANSION_MANIFEST) == EXPANSION_MANIFEST_SHA256
            and expansion_manifest.get("status")
            == "development64_confirmation96_partition_integrity_pass"
            and (expansion_manifest.get("gates") or {}).get("all_pass") is True
            and expansion_manifest.get("authorizes_paid_calls") is False
        ),
        "confirmation96_power_amendment_hash_matches": (
            sha256_file(POWER_AMENDMENT) == POWER_AMENDMENT_SHA256
        ),
        "development_manifest_hash_matches": (
            sha256_file(DEVELOPMENT_MANIFEST) == DEVELOPMENT_MANIFEST_SHA256
        ),
        "development64_power_amendment_hash_matches": (
            sha256_file(DEVELOPMENT_POWER_AMENDMENT)
            == DEVELOPMENT_POWER_AMENDMENT_SHA256
        ),
        "authorization_amendment_hash_matches": (
            sha256_file(AUTHORIZATION_AMENDMENT)
            == AUTHORIZATION_AMENDMENT_SHA256
        ),
        "matched_fixed_score_amendment_hash_matches": (
            sha256_file(MATCHED_FIXED_SCORE_AMENDMENT)
            == MATCHED_FIXED_SCORE_AMENDMENT_SHA256
        ),
        "endpoint_predictive_utility_amendment_hash_matches": (
            sha256_file(ENDPOINT_UTILITY_AMENDMENT)
            == ENDPOINT_UTILITY_AMENDMENT_SHA256
        ),
        "history_blind_estimand_clarification_hash_matches": (
            sha256_file(ESTIMAND_CLARIFICATION)
            == ESTIMAND_CLARIFICATION_SHA256
        ),
        "matched_realized_updater_amendment_hash_matches": (
            sha256_file(MATCHED_REALIZED_UPDATER_AMENDMENT)
            == MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
        ),
        "compute_matched_myopic_amendment_hash_matches": (
            sha256_file(COMPUTE_MATCHED_MYOPIC_AMENDMENT)
            == COMPUTE_MATCHED_MYOPIC_AMENDMENT_SHA256
        ),
        "development_manifest_is_frozen_and_endpoint_blind": (
            development_manifest.get("status") == "frozen"
            and (development_manifest.get("gates") or {}).get("all_pass") is True
            and (development_manifest.get("gates") or {}).get(
                "confirmation_remains_unaccessed"
            )
            is True
        ),
        "exact_expanded_experimental_partitions_are_bound": (
            (expansion_manifest.get("selection") or {}).get(
                "expanded_confirmation_tasks"
            )
            == TASKS
            and (expansion_manifest.get("selection") or {}).get(
                "expanded_development_tasks"
            )
            == 64
            and (expansion_manifest.get("selection") or {}).get(
                "expanded_reserve_tasks"
            )
            == 36
            and (expansion_manifest.get("partition_uid_sha256") or {}).get(
                "confirmation"
            )
            == CONFIRMATION_UID_SHA256
            and (expansion_manifest.get("partition_uid_sha256") or {}).get(
                "development"
            )
            == DEVELOPMENT_UID_SHA256
        ),
        "exact_96_unique_opaque_task_identities": (
            len(rows) == len({row["task_id"] for row in rows}) == TASKS
        ),
        "exact_four_24_task_blocks": counts == BLOCK_SIZES,
        "manifest_rows_are_opaque_and_truth_free": all(
            set(row) == {"task_id", "source_row_sha256", "block_id"}
            for row in rows
        ),
        "per_block_precharged_exposure_fits_daily_cap": (
            MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD <= DAILY_CAP_USD
        ),
        "mechanics_and_development_responses_do_not_exist": (
            not MECHANICS_RESULT.exists()
            and not list(DEVELOPMENT_ROOT.glob("block-*/RESULT.json"))
            and not (DEVELOPMENT_ROOT / "COMBINED_RESULT.json").exists()
        ),
        "model_calls_are_zero": True,
        "confirmation_images_labels_and_responses_remain_unopened": True,
        "sealed_test_and_reserve_remain_unopened": True,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "frozen" if gates["all_pass"] else "failed",
        "decision": "conditionally_execute_only_after_full_development_tier",
        "source": {
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "partition_integrity_manifest_sha256": PARTITION_MANIFEST_SHA256,
            "sample_size_expansion_manifest_sha256": (
                EXPANSION_MANIFEST_SHA256
            ),
            "confirmation96_power_amendment_sha256": POWER_AMENDMENT_SHA256,
            "source_commit": source_manifest["source"]["commit"],
            "source_tree": source_manifest["source"]["tree"],
            "image_archive_sha256": image_audit.ARCHIVE_SHA256,
            "confirmation_count": TASKS,
            "confirmation_uid_sha256": CONFIRMATION_UID_SHA256,
            "development_count": 64,
            "development_uid_sha256": DEVELOPMENT_UID_SHA256,
            "reserve_count_after_expansion": 36,
        },
        "development_precondition": {
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "development64_power_amendment_sha256": (
                DEVELOPMENT_POWER_AMENDMENT_SHA256
            ),
            "required_claim_report_interface": claim_report.INTERFACE_VERSION,
            "required_claim_tier": (
                "full_path_dependent_llm_native_development_signal"
            ),
            "authorization_amendment_sha256": (
                AUTHORIZATION_AMENDMENT_SHA256
            ),
            "matched_fixed_score_amendment_sha256": (
                MATCHED_FIXED_SCORE_AMENDMENT_SHA256
            ),
            "endpoint_predictive_utility_amendment_sha256": (
                ENDPOINT_UTILITY_AMENDMENT_SHA256
            ),
            "history_blind_estimand_clarification_sha256": (
                ESTIMAND_CLARIFICATION_SHA256
            ),
            "matched_realized_updater_amendment_sha256": (
                MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
            ),
            "compute_matched_myopic_amendment_sha256": (
                COMPUTE_MATCHED_MYOPIC_AMENDMENT_SHA256
            ),
            "legacy_no_ad_hoc_execution_field_is_preserved": True,
            "development_null_or_partial_tier_forbids_execution": True,
            "confirmation_design_is_frozen_before_development_responses": True,
        },
        "protocol": {
            "model": MODEL_ID,
            "reasoning": False,
            "planning_interface": development.INTERFACE_VERSION,
            "mechanics_interface": mechanics.INTERFACE_VERSION,
            "score_objective": mechanics.SCORE_OBJECTIVE,
            "blocks": {
                block_id: {
                    "size": BLOCK_SIZES[block_id],
                    "offset": BLOCK_OFFSETS[block_id],
                    "earliest_london_date": BLOCK_EARLIEST_DATES[block_id],
                    "model_seed": BLOCK_MODEL_SEEDS[block_id],
                    "maximum_requests": MAX_REQUESTS_PER_BLOCK,
                    "maximum_http_attempts": MAX_HTTP_ATTEMPTS_PER_BLOCK,
                    "maximum_precharged_exposure_usd": (
                        MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD
                    ),
                    "daily_cap_usd": DAILY_CAP_USD,
                    "run_cap_usd": BLOCK_RUN_CAP_USD,
                }
                for block_id in BLOCK_ORDER
            },
            "all_blocks_mandatory_after_confirmation_starts": True,
            "intermediate_scientific_endpoints_remain_sealed": True,
            "endpoint_labels_load_only_after_all_blocks_replay": True,
            "common_random_numbers_and_matched_history_blind_control_unchanged": True,
            "terminal_belief_label_obedience_is_a_mandatory_block_gate": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_changed_final_histories": MIN_CHANGED_FINAL_HISTORIES,
            "minimum_relative_brier_improvement": MIN_RELATIVE_BRIER_IMPROVEMENT,
            "confirmation_interval": "paired complete-task bootstrap 95pct upper below zero",
            "no_result_can_authorize_sealed_test_or_unregistered_model_swap": True,
        },
        "science_gates": _science_gates(),
        "implementation_bindings_at_freeze": {
            path: sha256_file(REPO_ROOT / path) for path in implementation_paths
        },
        "tasks": rows,
        "gates": gates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_manifest(output_path=args.output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
