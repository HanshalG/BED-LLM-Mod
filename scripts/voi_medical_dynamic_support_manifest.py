#!/usr/bin/env python3
"""Create a value-blind row manifest for MedDG dynamic-support experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence


SOURCE_SHA256 = "e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169"
SOURCE_ROWS = 499
SEED = 24423
SPLIT_SIZES = {
    "mechanics": 5,
    "opportunity": 40,
    "development": 20,
    "holdout": 434,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(
    count: int,
    *,
    seed: int = SEED,
    split_sizes: dict[str, int] = SPLIT_SIZES,
) -> dict[str, list[int]]:
    if sum(split_sizes.values()) != count:
        raise ValueError("split sizes must exhaust the source")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    result = {}
    offset = 0
    for name, size in split_sizes.items():
        result[name] = indices[offset : offset + size]
        offset += size
    return result


def ordered_hash(indices: Sequence[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(source: Path) -> dict[str, Any]:
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("MedDG source hash mismatch")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != SOURCE_ROWS:
        raise ValueError("MedDG source row count mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "self_repo",
            "target",
            "conv_hist",
        }:
            raise ValueError("MedDG row schema mismatch")
        if not all(isinstance(row[key], str) for key in row):
            raise ValueError("MedDG row values must be strings")
    splits = split_indices(len(rows))
    return {
        "schema_version": 1,
        "source_sha256": SOURCE_SHA256,
        "source_rows": SOURCE_ROWS,
        "seed": SEED,
        "values_inspected_or_emitted": False,
        "splits": splits,
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "split_ordered_hashes": {
            name: ordered_hash(indices) for name, indices in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
