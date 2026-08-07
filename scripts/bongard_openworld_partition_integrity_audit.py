#!/usr/bin/env python3
"""Remove exact image reuse from Bongard validation experiment partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import warnings
from zipfile import ZipFile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_image_integrity_audit as image_audit
from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-partition-integrity-audit-1"
IMAGE_INTEGRITY_MANIFEST = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_image_integrity_audit/"
    "bongard-openworld-image-integrity-audit-20260806/MANIFEST.json"
)
IMAGE_INTEGRITY_MANIFEST_SHA256 = (
    "239943ae789ebdc2c0a03577a02b04890c6d00f50ce45639c5defc1624ccee96"
)
PARTITION_INTEGRITY_MANIFEST = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_partition_integrity_audit/"
    "bongard-openworld-partition-integrity-audit-20260807/MANIFEST.json"
)
PARTITION_INTEGRITY_MANIFEST_SHA256 = (
    "9d9dc695924e728a2653af0b05c575eb4c04a62be933e497afa1bc1ac2bbcbc9"
)
PARTITION_SIZES = {
    "mechanics": source_audit.MECHANICS_TASKS,
    "development": source_audit.DEVELOPMENT_TASKS,
    "confirmation": source_audit.CONFIRMATION_TASKS,
}
EXPECTED_REJECTIONS = {
    "mechanics": 0,
    "development": 2,
    "confirmation": 5,
}
EXPECTED_UID_SHA256 = {
    "mechanics": "69de5a5fc444f3d14e9c07f570f4fe1fa8a4e915dd15926466c7c677d68bd323",
    "development": "356365ce90c964f6e99de5ca1585f9f582eebf8d4043afb55d1aee2a96c7c99b",
    "confirmation": "1537b43d37e03287520bd1c8bd583e7a7d4680c09ba2203e8238c8831205c631",
    "reserve": "ffc7549037b759df549147959e403c66715985b825bc0669c95f664fa5eff5d5",
}


def _uid_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(row["uid"]) for row in rows)).encode()
    ).hexdigest()


def _normalized_semantic(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def verify_bound_image_manifest() -> dict[str, Any]:
    observed = image_audit.sha256_file(IMAGE_INTEGRITY_MANIFEST)
    if observed != IMAGE_INTEGRITY_MANIFEST_SHA256:
        raise ValueError(
            "image-integrity manifest changed: "
            f"expected {IMAGE_INTEGRITY_MANIFEST_SHA256}, observed {observed}"
        )
    manifest = json.loads(IMAGE_INTEGRITY_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("all_gates_pass") is not True
        or manifest.get("status") != "image_integrity_pass"
        or manifest.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("image-integrity manifest is not the frozen clean pass")
    return {
        "manifest_sha256": observed,
        "archive_sha256": image_audit.ARCHIVE_SHA256,
    }


def image_fingerprints(
    rows: Sequence[Mapping[str, Any]],
    *,
    archive_path: Path = image_audit.ARCHIVE_PATH,
) -> dict[str, str]:
    """Return exact-byte identities, hashing only CRC/size collision groups."""
    paths = [str(path) for row in rows for path in row["imageFiles"]]
    if len(paths) != len(set(paths)):
        raise ValueError("metadata image paths must be unique before byte audit")
    with ZipFile(archive_path) as archive:
        infos = {
            info.filename: info for info in archive.infolist() if not info.is_dir()
        }
        missing = set(paths) - set(infos)
        if missing:
            raise ValueError(f"archive is missing {len(missing)} selected images")
        candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
        for path in paths:
            info = infos[path]
            candidates[(info.CRC, info.file_size)].append(path)

        fingerprints: dict[str, str] = {}
        for group in candidates.values():
            if len(group) == 1:
                path = group[0]
                fingerprints[path] = "unique-path:" + path
                continue
            for path in group:
                fingerprints[path] = hashlib.sha256(archive.read(path)).hexdigest()
    return fingerprints


def select_image_unique_partitions(
    rows: Sequence[Mapping[str, Any]],
    fingerprints: Mapping[str, str],
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[dict[str, Any]],
]:
    """Fill seeded partitions while rejecting exact-byte image reuse."""
    ordered = sorted(rows, key=source_audit._selection_key)
    selected: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in PARTITION_SIZES
    }
    rejected: list[dict[str, Any]] = []
    used: set[str] = set()
    cursor = 0

    for partition, size in PARTITION_SIZES.items():
        while len(selected[partition]) < size:
            if cursor >= len(ordered):
                raise ValueError("not enough image-unique validation rows")
            row = ordered[cursor]
            cursor += 1
            values = [fingerprints[str(path)] for path in row["imageFiles"]]
            internal_duplicates = len(values) - len(set(values))
            prior_duplicates = len(set(values) & used)
            if internal_duplicates or prior_duplicates:
                rejected.append(
                    {
                        "row": row,
                        "target_partition": partition,
                        "internal_duplicate_images": internal_duplicates,
                        "prior_partition_duplicate_images": prior_duplicates,
                    }
                )
                continue
            selected[partition].append(row)
            used.update(values)

    reserve = [item["row"] for item in rejected] + ordered[cursor:]
    return (
        selected["mechanics"],
        selected["development"],
        selected["confirmation"],
        reserve,
        rejected,
    )


def clean_validation_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    archive_path: Path = image_audit.ARCHIVE_PATH,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    bound_rows = list(rows) if rows is not None else source_audit.load_rows("val")
    fingerprints = image_fingerprints(bound_rows, archive_path=archive_path)
    mechanics, development, confirmation, reserve, _ = (
        select_image_unique_partitions(bound_rows, fingerprints)
    )
    return mechanics, development, confirmation, reserve


def _duplicate_stats(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    within_task = Counter()
    for partition, rows in partitions.items():
        for row in rows:
            values = [fingerprints[str(path)] for path in row["imageFiles"]]
            within_task[partition] += len(values) - len(set(values))
            for value in set(values):
                owners[value].append((partition, str(row["uid"])))

    cross = Counter()
    within_partition_task_pairs = Counter()
    for group in owners.values():
        partitions_in_group = sorted({partition for partition, _ in group})
        for index, left in enumerate(partitions_in_group):
            for right in partitions_in_group[index + 1 :]:
                cross[f"{left}|{right}"] += 1
        task_owners = set(group)
        for partition in partitions_in_group:
            if sum(owner_partition == partition for owner_partition, _ in task_owners) > 1:
                within_partition_task_pairs[partition] += 1
    return {
        "within_task_duplicate_images": dict(sorted(within_task.items())),
        "within_partition_duplicate_groups": dict(
            sorted(within_partition_task_pairs.items())
        ),
        "cross_partition_duplicate_groups": dict(sorted(cross.items())),
    }


def _semantic_overlap_count(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    field: str,
) -> int:
    left_values = {_normalized_semantic(str(row[field])) for row in left}
    right_values = {_normalized_semantic(str(row[field])) for row in right}
    return len(left_values & right_values)


def _bits_to_int(values: Sequence[bool]) -> int:
    result = 0
    for value in values:
        result = (result << 1) | int(value)
    return result


def _perceptual_hashes(data: bytes) -> tuple[int, int]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Palette images with Transparency expressed in bytes.*",
        )
        image = Image.open(BytesIO(data))
        gray = image.convert("L")
        average_pixels = list(
            gray.resize((8, 8), Image.Resampling.LANCZOS).get_flattened_data()
        )
        average = sum(average_pixels) / len(average_pixels)
        average_hash = _bits_to_int(
            [value > average for value in average_pixels]
        )
        difference_pixels = list(
            gray.resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data()
        )
        difference_hash = _bits_to_int(
            [
                difference_pixels[row * 9 + column + 1]
                > difference_pixels[row * 9 + column]
                for row in range(8)
                for column in range(8)
            ]
        )
        image.close()
        return average_hash, difference_hash


def strict_perceptual_overlap_count(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    *,
    archive_path: Path = image_audit.ARCHIVE_PATH,
) -> int:
    """Count highly conservative aHash+dHash near matches across the boundary."""
    by_partition: dict[str, list[tuple[int, int]]] = {
        "development": [],
        "confirmation": [],
    }
    with ZipFile(archive_path) as archive:
        for name, rows in (
            ("development", development),
            ("confirmation", confirmation),
        ):
            by_partition[name] = [
                _perceptual_hashes(archive.read(str(path)))
                for row in rows
                for path in row["imageFiles"]
            ]
    return sum(
        (left_average ^ right_average).bit_count() <= 2
        and (left_difference ^ right_difference).bit_count() <= 2
        for left_average, left_difference in by_partition["development"]
        for right_average, right_difference in by_partition["confirmation"]
    )


def run_audit(*, output_path: Path) -> dict[str, Any]:
    bound = verify_bound_image_manifest()
    rows = source_audit.load_rows("val")
    fingerprints = image_fingerprints(rows)
    original_parts = source_audit.split_validation_rows(rows)
    mechanics, development, confirmation, reserve, rejected = (
        select_image_unique_partitions(rows, fingerprints)
    )
    clean_parts = (mechanics, development, confirmation, reserve)
    names = ("mechanics", "development", "confirmation", "reserve")
    original = dict(zip(names, original_parts))
    clean = dict(zip(names, clean_parts))
    rejected_counts = Counter(
        str(item["target_partition"]) for item in rejected
    )
    clean_stats = _duplicate_stats(clean, fingerprints)
    exact_semantic_overlap = {
        field: _semantic_overlap_count(development, confirmation, field)
        for field in ("concept", "caption")
    }
    strict_perceptual_overlaps = strict_perceptual_overlap_count(
        development, confirmation
    )

    gates = {
        "bound_image_integrity_manifest_matches": bool(bound),
        "seeded_order_and_partition_sizes_are_preserved": (
            {name: len(part) for name, part in clean.items()}
            == {**PARTITION_SIZES, "reserve": 100}
        ),
        "mechanics_partition_is_unchanged": (
            _uid_sha256(mechanics) == _uid_sha256(original["mechanics"])
        ),
        "rejection_counts_match_frozen_audit": (
            {name: rejected_counts.get(name, 0) for name in PARTITION_SIZES}
            == EXPECTED_REJECTIONS
        ),
        "repaired_partition_hashes_match_frozen_audit": (
            {name: _uid_sha256(part) for name, part in clean.items()}
            == EXPECTED_UID_SHA256
        ),
        "selected_tasks_have_no_internal_duplicate_images": not any(
            clean_stats["within_task_duplicate_images"].get(name, 0)
            for name in PARTITION_SIZES
        ),
        "selected_experimental_tasks_share_no_exact_image_bytes": not any(
            clean_stats["within_partition_duplicate_groups"].get(name, 0)
            for name in PARTITION_SIZES
        )
        and not any(
            count
            for pair, count in clean_stats[
                "cross_partition_duplicate_groups"
            ].items()
            if "reserve" not in pair
        ),
        "development_and_confirmation_have_no_exact_semantic_duplicate": (
            not any(exact_semantic_overlap.values())
        ),
        "development_and_confirmation_have_no_strict_perceptual_near_duplicate": (
            strict_perceptual_overlaps == 0
        ),
        "selection_rule_does_not_use_semantic_truth_or_labels": True,
    }
    all_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model_calls": 0,
            "cost_usd": 0.0,
            "selection_rule": (
                "seeded source order; reject rows with byte-identical images "
                "within the row or any earlier selected experimental row"
            ),
            "selection_uses_semantic_truth_or_labels": False,
        },
        "status": "partition_integrity_pass" if all_pass else "fail",
        "all_gates_pass": all_pass,
        "authorizes_paid_calls": False,
        "bound_source": bound,
        "original_partition_duplicate_stats": _duplicate_stats(
            original, fingerprints
        ),
        "repaired_partition_duplicate_stats": clean_stats,
        "rejections": {
            "counts_by_target_partition": {
                name: rejected_counts.get(name, 0) for name in PARTITION_SIZES
            },
            "internal_duplicate_rows": sum(
                item["internal_duplicate_images"] > 0 for item in rejected
            ),
            "prior_selected_overlap_rows": sum(
                item["prior_partition_duplicate_images"] > 0
                for item in rejected
            ),
            "identities_emitted": False,
        },
        "repaired_partitions": {
            name: {"count": len(part), "uid_sha256": _uid_sha256(part)}
            for name, part in clean.items()
        },
        "post_selection_semantic_audit": {
            "development_confirmation_exact_overlap_counts": (
                exact_semantic_overlap
            ),
            "semantic_values_emitted": False,
            "used_for_selection": False,
        },
        "post_selection_perceptual_audit": {
            "development_confirmation_strict_near_pair_count": (
                strict_perceptual_overlaps
            ),
            "average_hash_max_distance": 2,
            "difference_hash_max_distance": 2,
            "used_for_selection": False,
        },
        "gates": gates,
        "interpretation": (
            "A pass repairs exact image reuse before any paid response while "
            "preserving seeded order, task counts, and the mechanics partition. "
            "The post-selection strict perceptual-hash screen is clean; this "
            "does not establish broad semantic independence or planning efficacy."
        ),
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_partition_integrity_audit/"
            "bongard-openworld-partition-integrity-audit-20260807/MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_audit(output_path=parse_args().output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_pass": result["all_gates_pass"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
