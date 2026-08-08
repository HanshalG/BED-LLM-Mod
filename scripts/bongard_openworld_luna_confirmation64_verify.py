#!/usr/bin/env python3
"""Independently verify the unopened Bongard confirmation96 freeze."""

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
from scripts import bongard_openworld_partition_integrity_audit as partition_audit
from scripts import bongard_openworld_sample_size_expansion_audit as expansion_audit
from scripts import bongard_openworld_source_protocol_audit as source_audit


INTERFACE_VERSION = "bongard-openworld-luna-confirmation96-freeze-9"
MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_confirmation64/"
    "PROTOCOL_MANIFEST_V9.json"
)
MANIFEST_SHA256 = (
    "ad1ddefffb8340dd2ba2b86c86f4d48a1fed05595e5b395ed328a86ba41d05d4"
)
SOURCE_MANIFEST_SHA256 = (
    "7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763"
)
PARTITION_MANIFEST_SHA256 = (
    partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
)
EXPANSION_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_sample_size_expansion_audit/"
    "bongard-openworld-sample-size-expansion-audit-20260808/MANIFEST.json"
)
EXPANSION_MANIFEST_SHA256 = (
    "809621dd848741848d25c2ddc8603751d43b24486ef1e666260248f9195cea37"
)
POWER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_CONFIRMATION96_POWER_AMENDMENT.md"
)
POWER_AMENDMENT_SHA256 = (
    "824374a32527b11cbda2d3bf81e570b4102d7930de0c3c5405626ce2ff6446b1"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "3e52e97c1ff28968273bedeea37ca2695cb41df5aff6478a3c3848b1bbee2ae0"
)
CONFIRMATION_UID_SHA256 = (
    "3826a64b46668226c996afa92e81cf270bf59f99a373813e37196552300ecb26"
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
    "results/nonmyopic/BONGARD_OPENWORLD_SAMPLE_SIZE_POWER_AUDIT_20260808.json",
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
    _, _, confirmation_rows, _ = expansion_audit.expanded_validation_rows()
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
        row["block_id"] = BLOCK_ORDER[index // 24]
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
            "size": 24,
            "offset": index * 24,
            "earliest_london_date": BLOCK_DATES[block_id],
            "model_seed": BLOCK_SEEDS[block_id],
            "maximum_requests": 1_032,
            "maximum_http_attempts": 1_053,
            "maximum_precharged_exposure_usd": 4.212,
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
            and source.get("partition_integrity_manifest_sha256")
            == PARTITION_MANIFEST_SHA256
            and sha256_file(partition_audit.PARTITION_INTEGRITY_MANIFEST)
            == PARTITION_MANIFEST_SHA256
            and source.get("confirmation96_expansion_manifest_sha256")
            == EXPANSION_MANIFEST_SHA256
            and sha256_file(EXPANSION_MANIFEST) == EXPANSION_MANIFEST_SHA256
            and source.get("confirmation96_power_amendment_sha256")
            == POWER_AMENDMENT_SHA256
            and sha256_file(POWER_AMENDMENT) == POWER_AMENDMENT_SHA256
            and source.get("source_commit") == source_audit.SOURCE_COMMIT
            and source.get("source_tree") == source_audit.SOURCE_TREE
            and source.get("image_archive_sha256") == image_audit.ARCHIVE_SHA256
            and source.get("confirmation_count") == 96
            and source.get("confirmation_uid_sha256")
            == CONFIRMATION_UID_SHA256
            and source.get("reserve_count_after_expansion") == 68
        ),
        "development_precondition_is_exact": (
            precondition.get("manifest_sha256")
            == DEVELOPMENT_MANIFEST_SHA256
            and precondition.get("required_claim_report_interface")
            == claim_report.INTERFACE_VERSION
            and precondition.get("required_claim_tier")
            == "full_path_dependent_llm_native_development_signal"
            and precondition.get("authorization_amendment_sha256")
            == AUTHORIZATION_AMENDMENT_SHA256
                and sha256_file(AUTHORIZATION_AMENDMENT)
                == AUTHORIZATION_AMENDMENT_SHA256
                and precondition.get("matched_fixed_score_amendment_sha256")
                == MATCHED_FIXED_SCORE_AMENDMENT_SHA256
                and sha256_file(MATCHED_FIXED_SCORE_AMENDMENT)
                == MATCHED_FIXED_SCORE_AMENDMENT_SHA256
            and precondition.get(
                "legacy_no_ad_hoc_execution_field_is_preserved"
            )
            is True
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
            and protocol.get("minimum_changed_final_histories") == 36
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
                "terminal_belief_label_obedience_is_a_mandatory_block_gate"
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
                "partition_integrity_manifest_hash_and_status_match",
                "confirmation96_expansion_manifest_hash_and_status_match",
                "confirmation96_power_amendment_hash_matches",
                "development_manifest_hash_matches",
                "authorization_amendment_hash_matches",
                "matched_fixed_score_amendment_hash_matches",
                "development_manifest_is_frozen_and_endpoint_blind",
                "exact_repaired_confirmation_partition_is_bound",
                "exact_96_unique_opaque_task_identities",
                "exact_four_24_task_blocks",
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
            == {
                "shared",
                "policy",
                "matched_mechanism",
                "path_dependent_support",
            }
            and len(manifest["science_gates"]["shared"]) == 4
            and len(manifest["science_gates"]["policy"]) == 9
            and len(manifest["science_gates"]["matched_mechanism"]) == 6
                and len(manifest["science_gates"]["path_dependent_support"]) == 13
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
        "maximum_requests_per_block": 1_032,
        "maximum_http_attempts_per_block": 1_053,
        "maximum_precharged_exposure_per_block_usd": 4.212,
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
