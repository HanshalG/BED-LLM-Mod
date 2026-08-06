#!/usr/bin/env python3
"""Independently verify the unopened Bongard confirmation64 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_image_integrity_audit as image_audit
from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_source_protocol_audit as source_audit


INTERFACE_VERSION = "bongard-openworld-luna-confirmation64-freeze-1"
MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_confirmation64/"
    "PROTOCOL_MANIFEST.json"
)
MANIFEST_SHA256 = (
    "0c7b77e7f6c1eabffd1e287320b465871b566cc72b6f0b5c90cb2c718c055fc4"
)
SOURCE_MANIFEST_SHA256 = (
    "7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "8659fb5fc6a02ddc59eb7147b6663d1fef3f96880e29f5e6de9bc0386f8e24aa"
)
CONFIRMATION_UID_SHA256 = (
    "27da2cc656add724bffbc43ea04ab28fa22bf8564ab9cba4d60e4e23df6facd0"
)
BLOCK_ORDER = ("a", "b", "c", "d")
BLOCK_DATES = {
    "a": "2026-08-15",
    "b": "2026-08-16",
    "c": "2026-08-17",
    "d": "2026-08-18",
}
BLOCK_SEEDS = {
    "a": 2_026_081_501,
    "b": 2_026_081_601,
    "c": 2_026_081_701,
    "d": 2_026_081_801,
}
IMPLEMENTATION_PATHS = (
    "scripts/bongard_openworld_vlm_bed.py",
    "scripts/bongard_openworld_luna_vlm_serving_smoke.py",
    "scripts/bongard_openworld_luna_vlm_mechanics_tree.py",
    "scripts/bongard_openworld_luna_vlm_development.py",
    "scripts/bongard_openworld_luna_claim_report.py",
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


def _expected_tasks() -> list[dict[str, str]]:
    _, _, confirmation_rows, _ = source_audit.split_validation_rows(
        source_audit.load_rows("val")
    )
    rows = sorted(
        (
            {
                "task_id": source_audit._task_layout(row)["task_id"],
                "source_row_sha256": source_audit.row_sha256(row),
            }
            for row in confirmation_rows
        ),
        key=lambda row: row["task_id"],
    )
    for index, row in enumerate(rows):
        row["block_id"] = BLOCK_ORDER[index // 16]
    return rows


def verify_manifest(
    path: Path = MANIFEST,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
    require_unopened_predecessors: bool = True,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("confirmation protocol manifest hash changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source = manifest.get("source") or {}
    precondition = manifest.get("development_precondition") or {}
    protocol = manifest.get("protocol") or {}
    blocks = protocol.get("blocks") or {}
    expected_tasks = _expected_tasks()
    expected_implementation = {
        item: sha256_file(REPO_ROOT / item) for item in IMPLEMENTATION_PATHS
    }
    expected_blocks = {
        block_id: {
            "size": 16,
            "offset": index * 16,
            "earliest_london_date": BLOCK_DATES[block_id],
            "model_seed": BLOCK_SEEDS[block_id],
            "maximum_requests": 688,
            "maximum_precharged_exposure_usd": 2.7520000000000002,
            "daily_cap_usd": 5.0,
            "run_cap_usd": 4.75,
        }
        for index, block_id in enumerate(BLOCK_ORDER)
    }
    checks = {
        "frozen_interface_and_decision": (
            manifest.get("status") == "frozen"
            and manifest.get("interface_version") == INTERFACE_VERSION
            and manifest.get("decision")
            == "conditionally_execute_only_after_full_development_tier"
        ),
        "source_and_archive_are_exactly_bound": (
            source.get("manifest_sha256") == SOURCE_MANIFEST_SHA256
            and source.get("source_commit") == source_audit.SOURCE_COMMIT
            and source.get("source_tree") == source_audit.SOURCE_TREE
            and source.get("image_archive_sha256") == image_audit.ARCHIVE_SHA256
            and source.get("confirmation_count") == 64
            and source.get("confirmation_uid_sha256")
            == CONFIRMATION_UID_SHA256
        ),
        "development_precondition_is_exact": (
            precondition.get("manifest_sha256")
            == DEVELOPMENT_MANIFEST_SHA256
            and precondition.get("required_claim_report_interface")
            == claim_report.INTERFACE_VERSION
            and precondition.get("required_claim_tier")
            == "full_policy_and_matched_mechanism"
            and precondition.get(
                "development_null_or_partial_tier_forbids_execution"
            )
            is True
            and precondition.get(
                "confirmation_design_is_frozen_before_development_responses"
            )
            is True
        ),
        "model_and_interfaces_are_exact": (
            protocol.get("model") == serving.MODEL_ID
            and protocol.get("reasoning") is False
            and protocol.get("planning_interface") == development.INTERFACE_VERSION
            and protocol.get("mechanics_interface") == mechanics.INTERFACE_VERSION
        ),
        "blocks_seeds_and_budget_are_exact": blocks == expected_blocks,
        "analysis_thresholds_are_exact": (
            protocol.get("bootstrap_replicates") == 20_000
            and protocol.get("bootstrap_seed") == 2_026_081_901
            and protocol.get("minimum_changed_final_histories") == 24
            and protocol.get("minimum_relative_brier_improvement") == 0.03
            and protocol.get("confirmation_interval")
            == "paired complete-task bootstrap 95pct upper below zero"
        ),
        "endpoint_and_continuation_rules_are_exact": (
            protocol.get("all_blocks_mandatory_after_confirmation_starts") is True
            and protocol.get("intermediate_scientific_endpoints_remain_sealed")
            is True
            and protocol.get("endpoint_labels_load_only_after_all_blocks_replay")
            is True
            and protocol.get(
                "common_random_numbers_and_matched_history_blind_control_unchanged"
            )
            is True
            and protocol.get(
                "no_result_can_authorize_sealed_test_or_unregistered_model_swap"
            )
            is True
        ),
        "task_rows_recompute_exactly": manifest.get("tasks") == expected_tasks,
        "implementation_bindings_recompute_exactly": (
            manifest.get("implementation_bindings_at_freeze")
            == expected_implementation
        ),
        "freeze_gates_are_exact_true_booleans": (
            bool(manifest.get("gates"))
            and set(manifest["gates"])
            == {
                "source_manifest_hash_matches",
                "development_manifest_hash_matches",
                "development_manifest_is_frozen_and_endpoint_blind",
                "exact_source_confirmation_partition_is_bound",
                "exact_64_unique_opaque_task_identities",
                "exact_four_16_task_blocks",
                "manifest_rows_are_opaque_and_truth_free",
                "per_block_precharged_exposure_fits_daily_cap",
                "mechanics_and_development_responses_do_not_exist",
                "model_calls_are_zero",
                "confirmation_images_labels_and_responses_remain_unopened",
                "sealed_test_and_reserve_remain_unopened",
                "all_pass",
            }
            and all(value is True for value in manifest["gates"].values())
        ),
        "science_gate_families_are_complete": (
            set(manifest.get("science_gates") or {})
            == {"shared", "policy", "matched_mechanism"}
            and len(manifest["science_gates"]["shared"]) == 4
            and len(manifest["science_gates"]["policy"]) == 10
            and len(manifest["science_gates"]["matched_mechanism"]) == 6
        ),
    }
    if require_unopened_predecessors:
        checks["mechanics_and_development_are_still_unopened"] = (
            not MECHANICS_RESULT.exists()
            and not list(DEVELOPMENT_ROOT.glob("block-*/RESULT.json"))
            and not (DEVELOPMENT_ROOT / "COMBINED_RESULT.json").exists()
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"confirmation freeze verification failed: {failed}")
    return {
        "verified": True,
        "manifest_sha256": actual_sha256,
        "task_count": len(expected_tasks),
        "block_sizes": {
            block_id: sum(row["block_id"] == block_id for row in expected_tasks)
            for block_id in BLOCK_ORDER
        },
        "maximum_requests_per_block": 688,
        "maximum_precharged_exposure_per_block_usd": 2.752,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
