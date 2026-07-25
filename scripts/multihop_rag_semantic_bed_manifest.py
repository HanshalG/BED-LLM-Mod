#!/usr/bin/env python3
"""Freeze content-blind MultiHop-RAG splits for semantic BED."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence


SOURCE_REPOSITORY = "https://github.com/yixuantt/MultiHop-RAG"
SOURCE_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
CODE_REVISION = "cde8e844af14b3012f20158abc2854fe8458212a"
QUERY_SHA256 = (
    "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff"
)
CORPUS_SHA256 = (
    "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f"
)
SELECTION_SEED = 24_410
PUBLIC_PREVIEW_ROWS = 20
EXPECTED_SOURCE_ROWS = 2_556
EXPECTED_CORPUS_ROWS = 609
EXPECTED_ELIGIBLE_ROWS = 2_238
EXPECTED_SPLIT_SIZES = {
    "mechanics": 5,
    "opportunity": 400,
    "development": 40,
    "holdout": 1_793,
}
SPLIT_HASHES = {
    "mechanics": (
        "59fa426ff76dc96b9bd70a831cc6d8eccced9476a04fec0476714a43c8e330e7"
    ),
    "opportunity": (
        "ca74186c4560100ab45c5fdf4a5a9d8a7fea950819848860c2b4ad8c747153e6"
    ),
    "development": (
        "58b94f9fb3fffd599c7a3bdeabf4035bf7aa1105740798e1efd51fce9d1a3ceb"
    ),
    "holdout": (
        "4022a0dd27dd1cd031c47e1373f599afa8ea9058c23bca962ead7204e29d207a"
    ),
}
EXPECTED_MECHANICS_IDS = ("2495", "1153", "89", "473", "189")


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


def split_ids(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    eligible = [
        str(index)
        for index, row in enumerate(rows)
        if index >= PUBLIC_PREVIEW_ROWS
        and row.get("question_type") != "null_query"
        and 2 <= len(row.get("evidence_list") or []) <= 4
    ]
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("eligible MultiHop-RAG row count changed")
    random.Random(SELECTION_SEED).shuffle(eligible)
    splits = {
        "mechanics": eligible[:5],
        "opportunity": eligible[5:405],
        "development": eligible[405:445],
        "holdout": eligible[445:],
    }
    sizes = {name: len(values) for name, values in splits.items()}
    if sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"split sizes changed: {sizes}")
    hashes = {
        name: ordered_hash(values) for name, values in splits.items()
    }
    if hashes != SPLIT_HASHES:
        raise ValueError("split hashes changed")
    if tuple(splits["mechanics"]) != EXPECTED_MECHANICS_IDS:
        raise ValueError("mechanics IDs changed")
    return splits


def build_manifest(
    query_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    if sha256_file(query_path) != QUERY_SHA256:
        raise ValueError("MultiHop-RAG query source hash changed")
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("MultiHop-RAG corpus source hash changed")
    rows = json.loads(query_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError("MultiHop-RAG query row count changed")
    if len(corpus) != EXPECTED_CORPUS_ROWS:
        raise ValueError("MultiHop-RAG corpus row count changed")
    splits = split_ids(rows)
    return {
        "interface_version": "multihop-rag-semantic-bed-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "dataset_revision": SOURCE_REVISION,
            "code_revision": CODE_REVISION,
            "query_sha256": QUERY_SHA256,
            "corpus_sha256": CORPUS_SHA256,
            "query_rows": EXPECTED_SOURCE_ROWS,
            "corpus_rows": EXPECTED_CORPUS_ROWS,
        },
        "selection_seed": SELECTION_SEED,
        "public_preview_rows_excluded": PUBLIC_PREVIEW_ROWS,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "splits": {
            name: {
                "task_ids": task_ids,
                "ordered_sha256": ordered_hash(task_ids),
                "query_type_counts": dict(
                    sorted(
                        Counter(
                            rows[int(task_id)]["question_type"]
                            for task_id in task_ids
                        ).items()
                    )
                ),
                "evidence_count_counts": {
                    str(count): sum(
                        len(rows[int(task_id)]["evidence_list"]) == count
                        for task_id in task_ids
                    )
                    for count in (2, 3, 4)
                },
            }
            for name, task_ids in splits.items()
        },
        "content_emitted": False,
        "answers_emitted": False,
        "document_ids_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.query_path, args.corpus_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in details.items()
                    if key != "task_ids"
                }
                for name, details in manifest["splits"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
