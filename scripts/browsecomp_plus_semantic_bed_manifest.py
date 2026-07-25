#!/usr/bin/env python3
"""Freeze content-blind BrowseComp-Plus splits from official qrels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence


SOURCE_REPOSITORY = "https://github.com/texttron/BrowseComp-Plus"
SOURCE_REVISION = "046949032b0328319cc9a02663a759ec601d9402"
SOURCE_ROWS = 830
SELECTION_SEED = 24_407
QREL_SHA256 = {
    "qrel_evidence.txt": (
        "a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1"
    ),
    "qrel_golds.txt": (
        "b875af4a745712bee7a94f464ed989232f8c77977c31824428470e11dcb28c73"
    ),
}
PARQUET_SHA256 = {
    "test-00000-of-00006.parquet": (
        "4ff9e93054eaee61b8d079a89b7ec4d02f16c9d05fa6e2fef7185b0786427a3a"
    ),
    "test-00001-of-00006.parquet": (
        "70cc3a6781693bb013c6855f231d15159222dddf826eb19947f6f85fcabccd4e"
    ),
    "test-00002-of-00006.parquet": (
        "f4fab28dcd293231da1b9da48680a19c7cdcb831df889d0691304de66016d194"
    ),
    "test-00003-of-00006.parquet": (
        "e738eb7a2e76feebe60632fc279a99773586d75ce5344e205c9c53fc3071c126"
    ),
    "test-00004-of-00006.parquet": (
        "25535ff3d41be25d8b213b903c478f34c18082af3cbfe0a4a3925989b84c87e4"
    ),
    "test-00005-of-00006.parquet": (
        "3a87031c5214322344b358db1d5f44d773b4d634ae43c60ce53ad582ccc47646"
    ),
}
EXPECTED_EVIDENCE_BINS = {"low": 231, "mid": 353, "high": 246}
SPLIT_BIN_SIZES = {
    "mechanics": {"low": 1, "mid": 2, "high": 2},
    "opportunity": {"low": 33, "mid": 51, "high": 36},
    "development": {"low": 11, "mid": 17, "high": 12},
}
EXPECTED_SPLIT_SIZES = {
    "mechanics": 5,
    "opportunity": 120,
    "development": 40,
    "holdout": 665,
}
SPLIT_HASHES = {
    "mechanics": (
        "b34afce2fb25685de253a44a1871bdb477c098bf1e44209eb5e5af301b1f4625"
    ),
    "opportunity": (
        "f5a98c9aa093ebfe63c33ef7feef522ad5f1ae0bc111157ca395a5ea1ca4cdca"
    ),
    "development": (
        "58d9782db534d122ef78ed6bda4d8703b26821cfb9864107b7170e61c19a2f44"
    ),
    "holdout": (
        "9fd7ef2bd221ad31f01dc38b261fa77b683f30f402c6e1ca863cee156abc1c4d"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid qrel line {line_number}")
        query_id, q0, doc_id, relevance = fields
        if q0 != "Q0" or relevance != "1" or not query_id or not doc_id:
            raise ValueError(f"invalid qrel fields on line {line_number}")
        if doc_id in qrels[query_id]:
            raise ValueError(f"duplicate qrel on line {line_number}")
        qrels[query_id].add(doc_id)
    return dict(qrels)


def evidence_bin(count: int) -> str:
    if count <= 4:
        return "low"
    if count <= 7:
        return "mid"
    return "high"


def split_ids(
    evidence_counts: dict[str, int],
    *,
    verify_frozen: bool = True,
) -> dict[str, list[str]]:
    bins = {name: [] for name in ("low", "mid", "high")}
    for query_id in sorted(evidence_counts, key=int):
        bins[evidence_bin(evidence_counts[query_id])].append(query_id)
    if verify_frozen:
        counts = {name: len(values) for name, values in bins.items()}
        if counts != EXPECTED_EVIDENCE_BINS:
            raise ValueError(f"evidence bins changed: {counts}")

    for offset, name in enumerate(("low", "mid", "high")):
        random.Random(SELECTION_SEED + offset).shuffle(bins[name])

    cursors = {name: 0 for name in bins}
    splits: dict[str, list[str]] = {}
    for split, quotas in SPLIT_BIN_SIZES.items():
        selected: list[str] = []
        for name in ("low", "mid", "high"):
            start = cursors[name]
            end = start + quotas[name]
            selected.extend(bins[name][start:end])
            cursors[name] = end
        splits[split] = selected
    splits["holdout"] = [
        query_id
        for name in ("low", "mid", "high")
        for query_id in bins[name][cursors[name] :]
    ]

    if verify_frozen:
        sizes = {name: len(values) for name, values in splits.items()}
        if sizes != EXPECTED_SPLIT_SIZES:
            raise ValueError(f"split sizes changed: {sizes}")
        hashes = {
            name: ordered_hash(values) for name, values in splits.items()
        }
        if hashes != SPLIT_HASHES:
            raise ValueError("split hashes changed")
    return splits


def build_manifest(qrel_dir: Path, parquet_dir: Path) -> dict[str, object]:
    for name, expected_hash in QREL_SHA256.items():
        actual_hash = sha256_file(qrel_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"{name} hash changed: {actual_hash}")
    for name, expected_hash in PARQUET_SHA256.items():
        actual_hash = sha256_file(parquet_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"{name} hash changed: {actual_hash}")

    evidence = parse_qrels(qrel_dir / "qrel_evidence.txt")
    gold = parse_qrels(qrel_dir / "qrel_golds.txt")
    if set(evidence) != set(gold) or len(evidence) != SOURCE_ROWS:
        raise ValueError("qrel task universe changed")
    if not all(gold[query_id] <= evidence[query_id] for query_id in evidence):
        raise ValueError("gold qrels are not a subset of evidence qrels")
    evidence_counts = {
        query_id: len(doc_ids) for query_id, doc_ids in evidence.items()
    }
    splits = split_ids(evidence_counts)

    return {
        "interface_version": "browsecomp-plus-semantic-bed-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "row_count": SOURCE_ROWS,
            "qrel_sha256": QREL_SHA256,
            "parquet_sha256": PARQUET_SHA256,
        },
        "selection_seed": SELECTION_SEED,
        "splits": {
            split: {
                "task_ids": task_ids,
                "ordered_sha256": ordered_hash(task_ids),
                "evidence_count_bins": {
                    name: sum(
                        evidence_bin(evidence_counts[query_id]) == name
                        for query_id in task_ids
                    )
                    for name in ("low", "mid", "high")
                },
            }
            for split, task_ids in splits.items()
        },
        "content_emitted": False,
        "document_ids_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrel-dir", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.qrel_dir, args.parquet_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
