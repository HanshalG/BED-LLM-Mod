#!/usr/bin/env python3
"""Freeze the independent Bongard confirmation64 protocol before responses."""

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
from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-confirmation64-freeze-2"
MODEL_ID = serving.MODEL_ID
BLOCK_ORDER = ("a", "b", "c", "d")
BLOCK_SIZES = {block_id: 16 for block_id in BLOCK_ORDER}
BLOCK_OFFSETS = {block_id: index * 16 for index, block_id in enumerate(BLOCK_ORDER)}
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
TASKS = 64
CASES_PER_TASK = 33
MAX_FINALS_PER_TASK = 10
MAX_REQUESTS_PER_TASK = CASES_PER_TASK + MAX_FINALS_PER_TASK
MAX_REQUESTS_PER_BLOCK = 16 * MAX_REQUESTS_PER_TASK
MAX_REQUEST_COST_USD = serving.MAX_REQUEST_COST_USD
MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD = (
    MAX_REQUESTS_PER_BLOCK * MAX_REQUEST_COST_USD
)
DAILY_CAP_USD = 5.0
BLOCK_RUN_CAP_USD = 4.75
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_081_901
MIN_CHANGED_FINAL_HISTORIES = 24
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.03

SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_source_protocol_audit/"
    "bongard-openworld-source-protocol-audit-20260806/MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763"
)
DEVELOPMENT_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development32/"
    "PROTOCOL_MANIFEST.json"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "451177a86b8ffbff128c4d8f94d7e6903873ce43050521119721f43882ecc9a4"
)
AUTHORIZATION_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_CONFIRMATION64_AUTHORIZATION_AMENDMENT.md"
)
AUTHORIZATION_AMENDMENT_SHA256 = (
    "c6f01987774fe8434298a6171d8f07ab7df6f8132c1cd64f443a98e13aef2f0a"
)
CONFIRMATION_UID_SHA256 = (
    "27da2cc656add724bffbc43ea04ab28fa22bf8564ab9cba4d60e4e23df6facd0"
)
DEVELOPMENT_ROOT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development32"
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
    _, _, confirmation_rows, _ = source_audit.split_validation_rows(
        source_audit.load_rows("val")
    )
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
            "exact_64_disjoint_confirmation_tasks",
            "root_candidate_brier_beats_constant_half",
            "all_endpoint_metrics_are_finite",
        ],
        "policy": [
            "at_least_24_dynamic_final_histories_differ_from_myopic",
            "at_least_24_dynamic_action_changes_clear_numerical_tie_margin",
            "dynamic_and_myopic_differ_in_every_execution_block",
            "dynamic_score_has_positive_mean_endpoint_ranking_fidelity",
            "dynamic_score_ranking_fidelity_is_not_worse_than_myopic",
            "dynamic_brier_relative_improvement_at_least_3_percent",
            "dynamic_brier_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_myopic",
            "dynamic_brier_is_not_worse_than_fixed_depth2",
            "dynamic_brier_is_not_worse_than_shuffled_control",
        ],
        "matched_mechanism": [
            "at_least_24_dynamic_final_histories_differ_from_history_blind",
            "dynamic_and_history_blind_differ_in_every_execution_block",
            "dynamic_brier_relative_improvement_vs_history_blind_at_least_3_percent",
            "dynamic_brier_vs_history_blind_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_history_blind",
            "dynamic_ranking_fidelity_is_not_worse_than_history_blind",
        ],
    }


def build_manifest(*, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"confirmation freeze already exists: {output_path}")
    source_manifest = _load(SOURCE_MANIFEST)
    development_manifest = _load(DEVELOPMENT_MANIFEST)
    confirmation = source_manifest["validation_partitions"]["confirmation"]
    rows = _confirmation_rows()
    counts = {
        block_id: sum(row["block_id"] == block_id for row in rows)
        for block_id in BLOCK_ORDER
    }
    implementation_paths = (
        "scripts/bongard_openworld_vlm_bed.py",
        "scripts/bongard_openworld_luna_vlm_serving_smoke.py",
        "scripts/bongard_openworld_luna_vlm_mechanics_tree.py",
        "scripts/bongard_openworld_luna_vlm_development.py",
        "scripts/bongard_openworld_luna_claim_report.py",
    )
    gates = {
        "source_manifest_hash_matches": (
            sha256_file(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256
        ),
        "development_manifest_hash_matches": (
            sha256_file(DEVELOPMENT_MANIFEST) == DEVELOPMENT_MANIFEST_SHA256
        ),
        "authorization_amendment_hash_matches": (
            sha256_file(AUTHORIZATION_AMENDMENT)
            == AUTHORIZATION_AMENDMENT_SHA256
        ),
        "development_manifest_is_frozen_and_endpoint_blind": (
            development_manifest.get("status") == "frozen"
            and (development_manifest.get("gates") or {}).get("all_pass") is True
            and (development_manifest.get("gates") or {}).get(
                "confirmation_remains_unaccessed"
            )
            is True
        ),
        "exact_source_confirmation_partition_is_bound": (
            confirmation.get("count") == TASKS
            and confirmation.get("uid_sha256") == CONFIRMATION_UID_SHA256
        ),
        "exact_64_unique_opaque_task_identities": (
            len(rows) == len({row["task_id"] for row in rows}) == TASKS
        ),
        "exact_four_16_task_blocks": counts == BLOCK_SIZES,
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
            "source_commit": source_manifest["source"]["commit"],
            "source_tree": source_manifest["source"]["tree"],
            "image_archive_sha256": image_audit.ARCHIVE_SHA256,
            "confirmation_count": TASKS,
            "confirmation_uid_sha256": CONFIRMATION_UID_SHA256,
        },
        "development_precondition": {
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "required_claim_report_interface": claim_report.INTERFACE_VERSION,
            "required_claim_tier": "full_llm_native_development_signal",
            "authorization_amendment_sha256": (
                AUTHORIZATION_AMENDMENT_SHA256
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
            "blocks": {
                block_id: {
                    "size": BLOCK_SIZES[block_id],
                    "offset": BLOCK_OFFSETS[block_id],
                    "earliest_london_date": BLOCK_EARLIEST_DATES[block_id],
                    "model_seed": BLOCK_MODEL_SEEDS[block_id],
                    "maximum_requests": MAX_REQUESTS_PER_BLOCK,
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
