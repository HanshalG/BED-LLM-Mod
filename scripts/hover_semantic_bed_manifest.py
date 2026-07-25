#!/usr/bin/env python3
"""Freeze content-blind HoVer splits without emitting claims or evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence


SOURCE_REPOSITORY = "https://github.com/hover-nlp/hover"
SOURCE_REVISION = "39b84697f196308f398a251a7aea9b82ae0f0562"
SOURCE_PATH = "data/hover/hover_dev_release_v1.1.json"
SOURCE_SHA256 = (
    "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d"
)
SOURCE_ROWS = 4_000
ELIGIBLE_HOPS = (3, 4)
EXPECTED_HOP_COUNTS = {3: 1_835, 4: 1_039}
SELECTION_SEED = 24_405
PER_HOP_SPLIT_SIZES = {
    "mechanics": 3,
    "opportunity": 200,
    "development": 20,
}
SPLIT_HASHES = {
    "mechanics": (
        "5ff4651736dd93dd91d793fa084ded2afd9f7b7a739020094a42be114aa17f4c"
    ),
    "opportunity": (
        "230ba25090172484250812ae0c24b089b2f3c888c7406702a47d2cd18a570fd0"
    ),
    "development": (
        "2eac3af917f9647919286ef37c1f691f0d95248d3e26e9cd353900ec3d0351f9"
    ),
    "holdout": (
        "050dc22e6700618f8929d301cfb7528f2b95bf77759077584d433064770cac6f"
    ),
}
EXPECTED_SPLIT_SIZES = {
    "mechanics": 6,
    "opportunity": 400,
    "development": 40,
    "holdout": 2_428,
}
EXPECTED_SCHEMA = {
    "claim",
    "hpqa_id",
    "label",
    "num_hops",
    "supporting_facts",
    "uid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"expected {SOURCE_ROWS} HoVer rows, got {len(rows)}")
    if any(set(row) != EXPECTED_SCHEMA for row in rows):
        raise ValueError("HoVer source schema changed")
    task_ids = [row["uid"] for row in rows]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("HoVer UIDs must be non-empty strings")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("HoVer UIDs must be unique")
    hop_counts = {
        hop: sum(row["num_hops"] == hop for row in rows)
        for hop in ELIGIBLE_HOPS
    }
    if hop_counts != EXPECTED_HOP_COUNTS:
        raise ValueError(
            f"HoVer eligible hop counts changed: {hop_counts}"
        )


def split_ids(
    metadata_rows: Sequence[dict[str, Any]],
    *,
    verify_frozen_hashes: bool = True,
) -> dict[str, list[str]]:
    splits = {
        "mechanics": [],
        "opportunity": [],
        "development": [],
        "holdout": [],
    }
    for hop in ELIGIBLE_HOPS:
        task_ids = sorted(
            str(row["uid"])
            for row in metadata_rows
            if row["num_hops"] == hop
        )
        random.Random(SELECTION_SEED + hop).shuffle(task_ids)
        mechanics_end = PER_HOP_SPLIT_SIZES["mechanics"]
        opportunity_end = (
            mechanics_end + PER_HOP_SPLIT_SIZES["opportunity"]
        )
        development_end = (
            opportunity_end + PER_HOP_SPLIT_SIZES["development"]
        )
        splits["mechanics"].extend(task_ids[:mechanics_end])
        splits["opportunity"].extend(
            task_ids[mechanics_end:opportunity_end]
        )
        splits["development"].extend(
            task_ids[opportunity_end:development_end]
        )
        splits["holdout"].extend(task_ids[development_end:])

    sizes = {name: len(values) for name, values in splits.items()}
    if verify_frozen_hashes and sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"HoVer split sizes changed: {sizes}")
    if verify_frozen_hashes:
        hashes = {
            name: ordered_hash(values) for name, values in splits.items()
        }
        if hashes != SPLIT_HASHES:
            raise ValueError("HoVer split hashes changed")
    return splits


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("HoVer source must be a JSON array of objects")
    return payload


def build_manifest(source_json: Path) -> dict[str, Any]:
    source_json = source_json.resolve()
    source_sha256 = sha256_file(source_json)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"HoVer source SHA-256 is {source_sha256}, "
            f"expected {SOURCE_SHA256}"
        )
    rows = load_rows(source_json)
    validate_rows(rows)
    metadata = [
        {"uid": str(row["uid"]), "num_hops": int(row["num_hops"])}
        for row in rows
    ]
    splits = split_ids(metadata)
    row_by_id = {str(row["uid"]): row for row in rows}

    return {
        "interface_version": "hover-semantic-bed-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
            "sha256": source_sha256,
            "row_count": len(rows),
            "schema": sorted(EXPECTED_SCHEMA),
            "eligible_hops": list(ELIGIBLE_HOPS),
            "eligible_hop_counts": {
                str(hop): EXPECTED_HOP_COUNTS[hop]
                for hop in ELIGIBLE_HOPS
            },
        },
        "selection_seed": SELECTION_SEED,
        "splits": {
            split: {
                "task_ids": task_ids,
                "ordered_sha256": ordered_hash(task_ids),
                "hop_counts": {
                    str(hop): sum(
                        row_by_id[task_id]["num_hops"] == hop
                        for task_id in task_ids
                    )
                    for hop in ELIGIBLE_HOPS
                },
                "record_sha256": {
                    task_id: hashlib.sha256(
                        json.dumps(
                            row_by_id[task_id],
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    for task_id in task_ids
                },
            }
            for split, task_ids in splits.items()
        },
        "content_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the content-blind HoVer semantic-BED manifest."
    )
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.source_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
