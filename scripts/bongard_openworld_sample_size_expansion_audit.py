#!/usr/bin/env python3
"""Freeze byte-clean Bongard development-64 and confirmation-96 partitions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_partition_integrity_audit as partition_audit
from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-sample-size-expansion-audit-2"
BOUND_PARTITION_MANIFEST = partition_audit.PARTITION_INTEGRITY_MANIFEST
BOUND_PARTITION_MANIFEST_SHA256 = (
    partition_audit.PARTITION_INTEGRITY_MANIFEST_SHA256
)
MECHANICS_TASKS = 4
ORIGINAL_DEVELOPMENT_TASKS = 32
DEVELOPMENT_TASKS = 64
ORIGINAL_CONFIRMATION_TASKS = 64
EXPANDED_CONFIRMATION_TASKS = 96
CONFIRMATION_ADDITIONS = 32
DEVELOPMENT_ADDITIONS = 32
EXPANDED_RESERVE_TASKS = 36
EXPECTED_UID_SHA256 = {
    "mechanics": partition_audit.EXPECTED_UID_SHA256["mechanics"],
    "development": (
        "bf3183ca1705b7048e4a4a20008881205c91a21b4554560f749cf7d5bd4e4610"
    ),
    "confirmation": (
        "3826a64b46668226c996afa92e81cf270bf59f99a373813e37196552300ecb26"
    ),
}
EXPECTED_ADDITION_UIDS = (
    "0378",
    "0729",
    "0498",
    "0822",
    "0997",
    "0246",
    "0052",
    "0719",
    "0019",
    "0455",
    "0224",
    "0604",
    "0484",
    "0346",
    "0809",
    "0625",
    "0657",
    "0405",
    "0268",
    "0238",
    "0086",
    "0260",
    "0250",
    "0859",
    "0843",
    "0589",
    "0896",
    "0555",
    "0923",
    "0597",
    "0651",
    "0573",
)
EXPECTED_EXTENSION_REJECTIONS = 11
EXPECTED_DEVELOPMENT_ADDITION_UIDS = (
    "0526",
    "0704",
    "0804",
    "0119",
    "0432",
    "0065",
    "0016",
    "1008",
    "0456",
    "0609",
    "0968",
    "0910",
    "0767",
    "0307",
    "0082",
    "0403",
    "0924",
    "0295",
    "0506",
    "0660",
    "0776",
    "0450",
    "0614",
    "0179",
    "0232",
    "0357",
    "0042",
    "0103",
    "0948",
    "0396",
    "0161",
    "0114",
)
EXPECTED_DEVELOPMENT_EXTENSION_REJECTIONS = 13


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within_semantic_duplicate_count(
    rows: Sequence[Mapping[str, Any]], field: str
) -> int:
    values = [
        partition_audit._normalized_semantic(str(row[field])) for row in rows
    ]
    return len(values) - len(set(values))


def _within_strict_perceptual_overlap_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    archive_path: Path = partition_audit.image_audit.ARCHIVE_PATH,
) -> int:
    with ZipFile(archive_path) as archive:
        task_hashes = [
            [
                partition_audit._perceptual_hashes(archive.read(str(path)))
                for path in row["imageFiles"]
            ]
            for row in rows
        ]
    return sum(
        (left_average ^ right_average).bit_count() <= 2
        and (left_difference ^ right_difference).bit_count() <= 2
        for index, left_task in enumerate(task_hashes)
        for right_task in task_hashes[index + 1 :]
        for left_average, left_difference in left_task
        for right_average, right_difference in right_task
    )


def _select_confirmation_additions(
    *,
    mechanics: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    fingerprints: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]]]:
    used = {
        fingerprints[str(path)]
        for row in (*mechanics, *development, *confirmation)
        for path in row["imageFiles"]
    }
    additions: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cursor = 0
    while len(additions) < CONFIRMATION_ADDITIONS:
        if cursor >= len(reserve):
            raise ValueError("not enough byte-clean reserve tasks for expansion")
        row = reserve[cursor]
        cursor += 1
        values = [fingerprints[str(path)] for path in row["imageFiles"]]
        internal_duplicates = len(values) - len(set(values))
        prior_duplicates = len(set(values) & used)
        if internal_duplicates or prior_duplicates:
            rejected.append(
                {
                    "row": row,
                    "internal_duplicate_images": internal_duplicates,
                    "prior_selected_duplicate_images": prior_duplicates,
                }
            )
            continue
        additions.append(row)
        used.update(values)
    expanded_reserve = [item["row"] for item in rejected] + list(reserve[cursor:])
    return additions, expanded_reserve, rejected


def _select_development_additions(
    *,
    mechanics: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    fingerprints: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]]]:
    used = {
        fingerprints[str(path)]
        for row in (*mechanics, *development, *confirmation)
        for path in row["imageFiles"]
    }
    additions: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cursor = 0
    while len(additions) < DEVELOPMENT_ADDITIONS:
        if cursor >= len(reserve):
            raise ValueError("not enough byte-clean reserve tasks for development")
        row = reserve[cursor]
        cursor += 1
        values = [fingerprints[str(path)] for path in row["imageFiles"]]
        internal_duplicates = len(values) - len(set(values))
        prior_duplicates = len(set(values) & used)
        if internal_duplicates or prior_duplicates:
            rejected.append(
                {
                    "row": row,
                    "internal_duplicate_images": internal_duplicates,
                    "prior_selected_duplicate_images": prior_duplicates,
                }
            )
            continue
        additions.append(row)
        used.update(values)
    expanded_reserve = [item["row"] for item in rejected] + list(reserve[cursor:])
    return additions, expanded_reserve, rejected


def expanded_validation_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    archive_path: Path = partition_audit.image_audit.ARCHIVE_PATH,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    bound_rows = list(rows) if rows is not None else source_audit.load_rows("val")
    fingerprints = partition_audit.image_fingerprints(
        bound_rows, archive_path=archive_path
    )
    mechanics, development, confirmation, reserve = (
        partition_audit.clean_validation_rows(
            bound_rows, archive_path=archive_path
        )
    )
    confirmation_additions, confirmation_reserve, _ = _select_confirmation_additions(
        mechanics=mechanics,
        development=development,
        confirmation=confirmation,
        reserve=reserve,
        fingerprints=fingerprints,
    )
    expanded_confirmation = [*confirmation, *confirmation_additions]
    development_additions, expanded_reserve, _ = _select_development_additions(
        mechanics=mechanics,
        development=development,
        confirmation=expanded_confirmation,
        reserve=confirmation_reserve,
        fingerprints=fingerprints,
    )
    return (
        mechanics,
        [*development, *development_additions],
        expanded_confirmation,
        expanded_reserve,
    )


def run_audit(*, output_path: Path) -> dict[str, Any]:
    if sha256_file(BOUND_PARTITION_MANIFEST) != BOUND_PARTITION_MANIFEST_SHA256:
        raise ValueError("bound 4/32/64 partition manifest changed")
    bound_manifest = json.loads(
        BOUND_PARTITION_MANIFEST.read_text(encoding="utf-8")
    )
    if bound_manifest.get("status") != "partition_integrity_pass":
        raise ValueError("bound partition manifest is not a clean pass")

    rows = source_audit.load_rows("val")
    fingerprints = partition_audit.image_fingerprints(rows)
    original = partition_audit.clean_validation_rows(rows)
    mechanics, development, confirmation, reserve = expanded_validation_rows(rows)
    confirmation_additions = confirmation[ORIGINAL_CONFIRMATION_TASKS:]
    development_additions = development[ORIGINAL_DEVELOPMENT_TASKS:]
    _, confirmation_reserve, confirmation_rejected = _select_confirmation_additions(
        mechanics=original[0],
        development=original[1],
        confirmation=original[2],
        reserve=original[3],
        fingerprints=fingerprints,
    )
    _, _, development_rejected = _select_development_additions(
        mechanics=original[0],
        development=original[1],
        confirmation=confirmation,
        reserve=confirmation_reserve,
        fingerprints=fingerprints,
    )
    selected = {
        "mechanics": mechanics,
        "development": development,
        "confirmation": confirmation,
    }
    duplicate_stats = partition_audit._duplicate_stats(selected, fingerprints)
    semantic_overlap = {
        field: partition_audit._semantic_overlap_count(
            development, confirmation, field
        )
        for field in ("concept", "caption")
    }
    strict_perceptual_overlap = partition_audit.strict_perceptual_overlap_count(
        development, confirmation
    )
    development_semantic_duplicates = {
        field: _within_semantic_duplicate_count(development, field)
        for field in ("concept", "caption")
    }
    development_strict_perceptual_overlap = (
        _within_strict_perceptual_overlap_count(development)
    )
    uid_hashes = {
        name: partition_audit._uid_sha256(part)
        for name, part in selected.items()
    }
    gates = {
        "bound_4_32_64_partition_manifest_matches": True,
        "mechanics_partition_is_byte_identical": list(mechanics)
        == list(original[0]),
        "all_original_development_tasks_are_retained": (
            list(development[:ORIGINAL_DEVELOPMENT_TASKS]) == list(original[1])
        ),
        "all_original_confirmation_tasks_are_retained": (
            list(confirmation[:ORIGINAL_CONFIRMATION_TASKS]) == list(original[2])
        ),
        "exact_4_64_96_36_partition_sizes": (
            len(mechanics),
            len(development),
            len(confirmation),
            len(reserve),
        )
        == (
            MECHANICS_TASKS,
            DEVELOPMENT_TASKS,
            EXPANDED_CONFIRMATION_TASKS,
            EXPANDED_RESERVE_TASKS,
        ),
        "exact_32_frozen_confirmation_additions": (
            tuple(str(row["uid"]) for row in confirmation_additions)
            == EXPECTED_ADDITION_UIDS
        ),
        "exact_32_frozen_development_additions": (
            tuple(str(row["uid"]) for row in development_additions)
            == EXPECTED_DEVELOPMENT_ADDITION_UIDS
        ),
        "expanded_partition_uid_hashes_match": uid_hashes
        == EXPECTED_UID_SHA256,
        "extension_rejection_count_matches": (
            len(confirmation_rejected) == EXPECTED_EXTENSION_REJECTIONS
        ),
        "development_extension_rejection_count_matches": (
            len(development_rejected)
            == EXPECTED_DEVELOPMENT_EXTENSION_REJECTIONS
        ),
        "selected_tasks_have_no_internal_duplicate_images": not any(
            duplicate_stats["within_task_duplicate_images"].values()
        ),
        "selected_experimental_tasks_share_no_exact_image_bytes": (
            not duplicate_stats["within_partition_duplicate_groups"]
            and not duplicate_stats["cross_partition_duplicate_groups"]
        ),
        "development_and_confirmation_have_no_exact_semantic_duplicate": (
            semantic_overlap == {"concept": 0, "caption": 0}
        ),
        "development_and_confirmation_have_no_strict_perceptual_near_duplicate": (
            strict_perceptual_overlap == 0
        ),
        "development_has_no_exact_semantic_duplicate": (
            development_semantic_duplicates == {"concept": 0, "caption": 0}
        ),
        "development_has_no_strict_perceptual_near_duplicate": (
            development_strict_perceptual_overlap == 0
        ),
        "selection_uses_no_semantic_truth_labels_or_model_outputs": True,
        "model_calls_are_zero": True,
        "endpoint_data_remain_unaccessed": True,
    }
    gates["all_pass"] = all(gates.values())
    if not gates["all_pass"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"sample-size expansion audit failed: {failed}")
    confirmation_rejected_counts = Counter(
        "internal" if item["internal_duplicate_images"] else "prior_selected"
        for item in confirmation_rejected
    )
    development_rejected_counts = Counter(
        "internal" if item["internal_duplicate_images"] else "prior_selected"
        for item in development_rejected
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "development64_confirmation96_partition_integrity_pass",
        "bound_partition_manifest_sha256": BOUND_PARTITION_MANIFEST_SHA256,
        "selection": {
            "rule": (
                "retain exact current mechanics/development/confirmation; scan "
                "the current reserve in frozen order to append 32 byte-clean "
                "confirmation rows, then scan the remainder in frozen order "
                "to append 32 byte-clean development rows"
            ),
            "uses_semantic_truth_labels_or_model_outputs": False,
            "original_development_tasks": ORIGINAL_DEVELOPMENT_TASKS,
            "development_additions": DEVELOPMENT_ADDITIONS,
            "expanded_development_tasks": DEVELOPMENT_TASKS,
            "original_confirmation_tasks": ORIGINAL_CONFIRMATION_TASKS,
            "confirmation_additions": CONFIRMATION_ADDITIONS,
            "expanded_confirmation_tasks": EXPANDED_CONFIRMATION_TASKS,
            "expanded_reserve_tasks": EXPANDED_RESERVE_TASKS,
            "confirmation_extension_rejections": len(confirmation_rejected),
            "confirmation_extension_rejection_types": dict(
                sorted(confirmation_rejected_counts.items())
            ),
            "development_extension_rejections": len(development_rejected),
            "development_extension_rejection_types": dict(
                sorted(development_rejected_counts.items())
            ),
        },
        "partition_uid_sha256": uid_hashes,
        "confirmation_additions": [
            {
                "task_id": source_audit._task_layout(row)["task_id"],
                "source_row_sha256": source_audit.row_sha256(dict(row)),
            }
            for row in confirmation_additions
        ],
        "development_additions": [
            {
                "task_id": source_audit._task_layout(row)["task_id"],
                "source_row_sha256": source_audit.row_sha256(dict(row)),
            }
            for row in development_additions
        ],
        "duplicate_stats": duplicate_stats,
        "post_selection_semantic_overlap": semantic_overlap,
        "post_selection_strict_perceptual_overlap": strict_perceptual_overlap,
        "development_semantic_duplicate_count": development_semantic_duplicates,
        "development_strict_perceptual_overlap": (
            development_strict_perceptual_overlap
        ),
        "gates": gates,
        "model_calls_made": 0,
        "endpoint_data_accessed": False,
        "authorizes_paid_calls": False,
    }
    checkpoint(output_path, result)
    return result


def verify_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("sample-size expansion manifest hash changed")
    observed = json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(
        prefix="bongard-sample-expansion-verify-"
    ) as tmp:
        replay = run_audit(output_path=Path(tmp) / "MANIFEST.json")
    if (
        observed != replay
        or observed.get("status")
        != "development64_confirmation96_partition_integrity_pass"
        or observed.get("gates", {}).get("all_pass") is not True
        or observed.get("model_calls_made") != 0
        or observed.get("endpoint_data_accessed") is not False
    ):
        raise ValueError("sample-size expansion manifest is not a clean pass")
    return {"verified": True, "manifest_sha256": expected_sha256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
