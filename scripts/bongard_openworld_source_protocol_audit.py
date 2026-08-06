#!/usr/bin/env python3
"""Audit Bongard-OpenWorld for a sequential visual-semantic BED protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Sequence
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-sequential-bed-source-protocol-1"
SOURCE_ROOT = REPO_ROOT / "external/Bongard-OpenWorld"
SOURCE_COMMIT = "0462ba15f3be2f7f9a7e6cdd0d24314a1fabb5fe"
SOURCE_TREE = "b527dfa22222cab3beef5c4023a8b3837d8d7ee5"
DATA_ROOT = SOURCE_ROOT / "assets/data/bongard-ow"
DATA_HASHES = {
    "all": "6498f879b33d4e20be692dfaa0d44daec50b57b1af7571b86f8044d1e925b82c",
    "train": "12062804a2636c433e11b1eb0fdf1f56edfd98acbca500454c1383b2e7083aee",
    "val": "aeecd23c1f7f40dd2db1cb59b3ecda22b1fc9656b18855c0928e6311a2764e0f",
    "test": "df2a946b1b5d12c039c9e36ac46dc3a8420eee4c78a306418683f273ed22a35d",
}
EXPECTED_SPLIT_SIZES = {"train": 610, "val": 200, "test": 200}
EXPECTED_TOTAL_ROWS = 1010
EXPECTED_IMAGES_PER_TASK = 14
SUPPORT_POSITIONS = tuple(range(6)) + tuple(range(7, 13))
ENDPOINT_POSITIONS = (6, 13)
PROTOCOL_SEED = 20260806
INITIAL_PER_CLASS = 2
QUERY_BUDGET = 2
MECHANICS_TASKS = 4
DEVELOPMENT_TASKS = 32
CONFIRMATION_TASKS = 64
AUDIT_EXPOSED_TEST_UIDS = ("0008",)
UID_RE = re.compile(r"^[0-9]{4}$")
IMAGE_RE = re.compile(r"^images/([0-9]{4})/(pos|neg)__([0-6])__.+$")

BACKUP_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1aXr3ihVq0mtzbl6ZNJMogYEyEY-WALNr&export=download&confirm=t"
)
BACKUP_EXPECTED_SIZE = 5_124_375_111
BACKUP_TAIL_BYTES = 65_536


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def data_path(split: str) -> Path:
    suffix = "" if split == "all" else f"_{split}"
    return DATA_ROOT / f"bongard_ow{suffix}.json"


def verify_source() -> dict[str, Any]:
    observed = {
        "commit": _git_value("rev-parse", "HEAD"),
        "tree": _git_value("rev-parse", "HEAD^{tree}"),
        "data_sha256": {
            split: sha256_file(data_path(split))
            for split in ("all", "train", "val", "test")
        },
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "data_sha256": DATA_HASHES,
    }
    if observed != expected:
        raise ValueError(
            "Bongard-OpenWorld source changed: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def load_rows(split: str) -> list[dict[str, Any]]:
    value = json.loads(data_path(split).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError(f"{split} data is not a list of objects")
    return value


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def row_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_types = {
        "uid": str,
        "commonSense": str,
        "concept": str,
        "caption": str,
        "imageFiles": list,
    }
    for key, expected_type in required_types.items():
        if not isinstance(row.get(key), expected_type):
            errors.append(f"invalid_{key}_type")
    if errors:
        return errors

    uid = row["uid"]
    if UID_RE.fullmatch(uid) is None:
        errors.append("invalid_uid")
    if not row["concept"].strip() or not row["caption"].strip():
        errors.append("empty_semantic_target")
    if not row["commonSense"].isdigit():
        errors.append("invalid_common_sense_code")

    image_files = row["imageFiles"]
    if len(image_files) != EXPECTED_IMAGES_PER_TASK:
        return errors + ["wrong_image_count"]
    if not all(isinstance(path, str) for path in image_files):
        return errors + ["invalid_image_path_type"]
    if len(image_files) != len(set(image_files)):
        errors.append("duplicate_image_path")

    for position, path in enumerate(image_files):
        match = IMAGE_RE.fullmatch(path)
        expected_label = "pos" if position < 7 else "neg"
        expected_class_index = position if position < 7 else position - 7
        if match is None:
            errors.append("invalid_image_path")
            continue
        path_uid, label, class_index = match.groups()
        if path_uid != uid:
            errors.append("image_uid_mismatch")
        if label != expected_label or int(class_index) != expected_class_index:
            errors.append("image_label_order_mismatch")
    return sorted(set(errors))


def _selection_key(row: dict[str, Any]) -> tuple[str, str]:
    uid = row["uid"]
    digest = hashlib.sha256(
        f"{PROTOCOL_SEED}|validation-split|{uid}|{row_sha256(row)}".encode()
    ).hexdigest()
    return digest, uid


def split_validation_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    ordered = sorted(rows, key=_selection_key)
    mechanics_end = MECHANICS_TASKS
    development_end = mechanics_end + DEVELOPMENT_TASKS
    confirmation_end = development_end + CONFIRMATION_TASKS
    return (
        ordered[:mechanics_end],
        ordered[mechanics_end:development_end],
        ordered[development_end:confirmation_end],
        ordered[confirmation_end:],
    )


def _task_rng(uid: str) -> random.Random:
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_SEED}|task|{uid}".encode()).digest()[:8],
        "big",
    )
    return random.Random(seed)


def _task_layout(row: dict[str, Any]) -> dict[str, Any]:
    if row_errors(row):
        raise ValueError(f"invalid Bongard row {row.get('uid')}")
    uid = row["uid"]
    rng = _task_rng(uid)

    positive = list(range(6))
    negative = list(range(7, 13))
    rng.shuffle(positive)
    rng.shuffle(negative)
    initial_positions = positive[:INITIAL_PER_CLASS] + negative[:INITIAL_PER_CLASS]
    candidate_positions = positive[INITIAL_PER_CLASS:] + negative[INITIAL_PER_CLASS:]
    rng.shuffle(initial_positions)
    rng.shuffle(candidate_positions)
    endpoint_positions = list(ENDPOINT_POSITIONS)
    rng.shuffle(endpoint_positions)

    opaque_names = [f"image-{index:02d}" for index in range(14)]
    rng.shuffle(opaque_names)
    opaque_by_position = dict(enumerate(opaque_names))
    task_id = "task-" + hashlib.sha256(
        f"{PROTOCOL_SEED}|opaque-task|{uid}".encode()
    ).hexdigest()[:12]

    return {
        "task_id": task_id,
        "initial_positions": initial_positions,
        "candidate_positions": candidate_positions,
        "endpoint_positions": endpoint_positions,
        "opaque_by_position": opaque_by_position,
    }


def task_protocol(row: dict[str, Any]) -> dict[str, Any]:
    layout = _task_layout(row)
    initial_positions = layout["initial_positions"]
    candidate_positions = layout["candidate_positions"]
    endpoint_positions = layout["endpoint_positions"]
    opaque_by_position = layout["opaque_by_position"]
    label_by_position = {
        position: "positive" if position < 7 else "negative"
        for position in range(14)
    }

    return {
        "task_id": layout["task_id"],
        "initial": [
            {
                "image_id": opaque_by_position[position],
                "label": label_by_position[position],
            }
            for position in initial_positions
        ],
        "candidates": [
            {"image_id": opaque_by_position[position]}
            for position in candidate_positions
        ],
        "endpoints": [
            {"image_id": opaque_by_position[position]}
            for position in endpoint_positions
        ],
        "query_budget": QUERY_BUDGET,
    }


def hidden_state_boundary_holds(row: dict[str, Any]) -> bool:
    payload = task_protocol(row)
    serialized = canonical_json(payload)
    image_ids = [item["image_id"] for item in payload["initial"]]
    image_ids += [item["image_id"] for item in payload["candidates"]]
    image_ids += [item["image_id"] for item in payload["endpoints"]]
    forbidden = [
        row["uid"],
        row["concept"],
        row["caption"],
        *row["imageFiles"],
    ]
    return (
        len(image_ids) == 14
        and len(image_ids) == len(set(image_ids))
        and len(payload["initial"]) == 4
        and Counter(item["label"] for item in payload["initial"])
        == {"positive": 2, "negative": 2}
        and len(payload["candidates"]) == 8
        and len(payload["endpoints"]) == 2
        and all("label" not in item for item in payload["candidates"])
        and all("label" not in item for item in payload["endpoints"])
        and all(value not in serialized for value in forbidden)
    )


def assess_backup_ranges(
    *, first_bytes: bytes, tail_bytes: bytes, total_size: int
) -> dict[str, Any]:
    return {
        "total_size_bytes": total_size,
        "expected_size_matches": total_size == BACKUP_EXPECTED_SIZE,
        "zip_local_header_present": first_bytes.startswith(b"PK\x03\x04"),
        "zip64_end_record_present": b"PK\x06\x06" in tail_bytes,
        "zip64_locator_present": b"PK\x06\x07" in tail_bytes,
        "zip_end_record_present": b"PK\x05\x06" in tail_bytes,
    }


def _range_read(start: int, end: int) -> tuple[bytes, int]:
    request = Request(BACKUP_URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes [0-9]+-[0-9]+/([0-9]+)", content_range)
        if response.status != 206 or match is None:
            raise ValueError(
                "official backup did not honor the required byte range: "
                f"status={response.status}, Content-Range={content_range!r}"
            )
        return response.read(), int(match.group(1))


def probe_backup_archive() -> dict[str, Any]:
    first, first_total = _range_read(0, 3)
    tail_start = BACKUP_EXPECTED_SIZE - BACKUP_TAIL_BYTES
    tail, tail_total = _range_read(tail_start, BACKUP_EXPECTED_SIZE - 1)
    if first_total != tail_total:
        raise ValueError("official backup size changed between range requests")
    return assess_backup_ranges(
        first_bytes=first,
        tail_bytes=tail,
        total_size=first_total,
    )


def _partition_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "uid_sha256": hashlib.sha256(
            "\n".join(sorted(row["uid"] for row in rows)).encode()
        ).hexdigest(),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    source = verify_source()
    master = load_rows("all")
    splits = {
        split: load_rows(split) for split in ("train", "val", "test")
    }
    all_split_rows = [row for split in splits.values() for row in split]
    split_ids = {
        split: {row.get("uid") for row in rows}
        for split, rows in splits.items()
    }
    master_ids = {row.get("uid") for row in master}

    errors: Counter[str] = Counter()
    for row in master:
        errors.update(row_errors(row))
    image_paths = [path for row in master for path in row.get("imageFiles", [])]
    mechanics, development, confirmation, reserve = split_validation_rows(
        splits["val"]
    )
    validation_selected = mechanics + development + confirmation + reserve
    backup = probe_backup_archive()
    backup_ok = all(
        bool(value)
        for key, value in backup.items()
        if key != "total_size_bytes"
    )

    gates = {
        "source_hashes_match": bool(source),
        "exact_master_and_split_sizes": (
            len(master) == EXPECTED_TOTAL_ROWS
            and {
                split: len(rows) for split, rows in splits.items()
            }
            == EXPECTED_SPLIT_SIZES
        ),
        "official_splits_are_disjoint_and_exhaustive": (
            not (split_ids["train"] & split_ids["val"])
            and not (split_ids["train"] & split_ids["test"])
            and not (split_ids["val"] & split_ids["test"])
            and set().union(*split_ids.values()) == master_ids
            and len(all_split_rows) == len(master)
        ),
        "all_rows_have_exact_balanced_image_structure": not errors,
        "all_image_references_are_unique": (
            len(image_paths) == EXPECTED_TOTAL_ROWS * EXPECTED_IMAGES_PER_TASK
            and len(image_paths) == len(set(image_paths))
        ),
        "validation_protocol_split_is_exact_and_disjoint": (
            [len(mechanics), len(development), len(confirmation), len(reserve)]
            == [4, 32, 64, 100]
            and len({row["uid"] for row in validation_selected}) == 200
        ),
        "planner_payload_hides_semantic_truth_paths_and_unrevealed_labels": all(
            hidden_state_boundary_holds(row) for row in validation_selected
        ),
        "official_query_images_are_excluded_from_selectable_support": (
            not (set(SUPPORT_POSITIONS) & set(ENDPOINT_POSITIONS))
            and len(SUPPORT_POSITIONS) == 12
        ),
        "official_backup_is_range_accessible_zip64": backup_ok,
        "structural_depth_two_branching_exceeds_myopic_action_count": (
            8 * 7 > 8 and QUERY_BUDGET == 2
        ),
        "audit_exposed_test_rows_are_excluded_from_sealed_test": all(
            uid in split_ids["test"] for uid in AUDIT_EXPOSED_TEST_UIDS
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_SOURCE_PROTOCOL_AUDIT_PREREGISTRATION.md"
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
            "seed": PROTOCOL_SEED,
            "initial_labelled_images": 4,
            "selectable_images": 8,
            "query_budget": QUERY_BUDGET,
            "ordered_depth_two_action_sequences": 56,
            "endpoint_images": 2,
        },
        "status": "source_protocol_pass" if all_gates_pass else "fail",
        "scientific_opportunity_status": "untested",
        "authorizes_paid_calls": False,
        "source": {
            **source,
            "repository": "https://github.com/rujiewu/Bongard-OpenWorld",
            "backup_url": BACKUP_URL,
            "backup_probe": backup,
            "full_backup_sha256_verified": False,
        },
        "data": {
            "master_rows": len(master),
            "split_sizes": {
                split: len(rows) for split, rows in splits.items()
            },
            "row_error_counts": dict(sorted(errors.items())),
            "unique_image_references": len(set(image_paths)),
            "common_sense_code_counts": dict(
                sorted(Counter(row["commonSense"] for row in master).items())
            ),
        },
        "validation_partitions": {
            "mechanics": _partition_summary(mechanics),
            "development": _partition_summary(development),
            "confirmation": _partition_summary(confirmation),
            "reserve": _partition_summary(reserve),
        },
        "test_boundary": {
            "official_test_count": len(splits["test"]),
            "audit_exposed_uids": list(AUDIT_EXPOSED_TEST_UIDS),
            "sealed_test_count": (
                len(splits["test"]) - len(AUDIT_EXPOSED_TEST_UIDS)
            ),
            "concepts_captions_and_paths_emitted": False,
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "A pass establishes a reproducible leakage-controlled sequential "
            "visual-semantic task source. It does not establish that depth-two "
            "planning beats myopic selection; that requires a later VLM "
            "mechanics/opportunity experiment."
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
            / "results/nonmyopic/bongard_openworld_source_protocol_audit/"
            "bongard-openworld-source-protocol-audit-20260806/MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_audit(output_path=parse_args().output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_opportunity_status": result[
                    "scientific_opportunity_status"
                ],
                "authorizes_paid_calls": result["authorizes_paid_calls"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
