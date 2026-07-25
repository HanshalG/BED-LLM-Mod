#!/usr/bin/env python3
"""Freeze content-blind Bamboogle splits without emitting questions or answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence


SOURCE_REPOSITORY = (
    "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets"
)
SOURCE_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
SOURCE_PATH = "bamboogle/test.jsonl"
SOURCE_SHA256 = (
    "c9703dae6bb1ceb9e2df77be45da28cb12aa040d2f471507890a461296968f3f"
)
SELECTION_SEED = 24_404
EXPECTED_COUNT = 125
SPLIT_SIZES = {
    "mechanics": 5,
    "opportunity": 20,
    "development": 20,
    "holdout": 80,
}
SPLIT_HASHES = {
    "mechanics": (
        "f2a9e1d7587e22d16127dae8ed15dd8c58fcc9b867ce1a1a1fc13f9e01912b7e"
    ),
    "opportunity": (
        "6abfc7dc790564fa09cc6fb282851beece36306c682e038e177e39b546aa2e94"
    ),
    "development": (
        "01837e7e950110c5da05440862d443092a1edc9ce14b0032c8942d5bbfedb8a6"
    ),
    "holdout": (
        "c2da4d412ae4d7c14c3aca06033f61ba5ec1078789ab70a39811eeeed3f85856"
    ),
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


def split_ids(task_ids: Sequence[str]) -> dict[str, list[str]]:
    ordered_ids = sorted(str(task_id) for task_id in task_ids)
    if len(ordered_ids) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} Bamboogle IDs, got {len(ordered_ids)}"
        )
    if len(set(ordered_ids)) != EXPECTED_COUNT:
        raise ValueError("Bamboogle IDs must be unique")

    rng = random.Random(SELECTION_SEED)
    rng.shuffle(ordered_ids)
    splits: dict[str, list[str]] = {}
    cursor = 0
    for split, size in SPLIT_SIZES.items():
        splits[split] = ordered_ids[cursor : cursor + size]
        cursor += size
    if cursor != EXPECTED_COUNT:
        raise ValueError("Bamboogle split sizes do not cover the task universe")

    hashes = {
        split: ordered_hash(split_ids)
        for split, split_ids in splits.items()
    }
    if hashes != SPLIT_HASHES:
        raise ValueError("Bamboogle split hashes changed")
    return splits


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON on Bamboogle line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Bamboogle line {line_number} is not an object")
        if set(row) != {"id", "question", "golden_answers"}:
            raise ValueError(
                f"Bamboogle line {line_number} has an unexpected schema"
            )
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(
                f"Bamboogle line {line_number} has an invalid question"
            )
        answers = row["golden_answers"]
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer.strip() for answer in answers)
        ):
            raise ValueError(
                f"Bamboogle line {line_number} has invalid answers"
            )
        rows.append(row)
    return rows


def build_manifest(source_jsonl: Path) -> dict[str, Any]:
    source_jsonl = source_jsonl.resolve()
    source_sha256 = sha256_file(source_jsonl)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"Bamboogle source SHA-256 is {source_sha256}, "
            f"expected {SOURCE_SHA256}"
        )
    rows = load_rows(source_jsonl)
    splits = split_ids([str(row["id"]) for row in rows])
    row_by_id = {str(row["id"]): row for row in rows}

    return {
        "interface_version": "bamboogle-semantic-bed-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
            "sha256": source_sha256,
            "row_count": len(rows),
            "schema": ["id", "question", "golden_answers"],
        },
        "selection_seed": SELECTION_SEED,
        "splits": {
            split: {
                "task_ids": task_ids,
                "ordered_sha256": ordered_hash(task_ids),
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
        description="Build the content-blind Bamboogle semantic-BED manifest."
    )
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.source_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
