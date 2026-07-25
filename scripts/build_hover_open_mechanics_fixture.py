#!/usr/bin/env python3
"""Extract endpoint-free inputs for already-open HoVer mechanics tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = (
    "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d"
)
RETRIEVAL_SHA256 = (
    "b50a961f63a95ff184986af766b15ff6b1d6c98f7e86b39355b64b1a85fb3745"
)
TASK_IDS = (
    "a88d2342-f506-4b15-8578-fb7861eb54c1",
    "3cc79319-433d-49b2-97f3-953ba925d6bd",
)
CANDIDATE_COUNT = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fixture(
    source_path: Path,
    retrieval_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("HoVer source hash changed")
    if sha256_file(retrieval_path) != RETRIEVAL_SHA256:
        raise ValueError("HoVer retrieval hash changed")
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    retrieval_rows = json.loads(
        retrieval_path.read_text(encoding="utf-8")
    )
    source_by_id = {str(row["uid"]): row for row in source_rows}
    retrieval_by_id = {str(row["id"]): row for row in retrieval_rows}
    tasks = []
    for task_id in TASK_IDS:
        source_row = source_by_id[task_id]
        retrieval_row = retrieval_by_id[task_id]
        titles = [
            str(title)
            for title in retrieval_row["doc_retrieval_results"][0][0]
        ]
        if len(titles) != CANDIDATE_COUNT or len(set(titles)) != CANDIDATE_COUNT:
            raise ValueError("HoVer mechanics candidate catalog changed")
        claim = source_row["claim"]
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("HoVer mechanics claim is invalid")
        tasks.append(
            {
                "task_id": task_id,
                "num_hops": int(source_row["num_hops"]),
                "claim": claim,
                "candidate_titles": titles,
            }
        )
    return {
        "interface_version": "hover-open-mechanics-input-1",
        "source_sha256": SOURCE_SHA256,
        "retrieval_sha256": RETRIEVAL_SHA256,
        "tasks": tasks,
        "endpoint_fields_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--retrieval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = build_fixture(args.source_json, args.retrieval_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
