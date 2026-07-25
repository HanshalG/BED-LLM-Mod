#!/usr/bin/env python3
"""Freeze a fresh ClariQ mechanics task for spaced-code transport V2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clariq_dynamic_support_manifest import (
    DEVELOPMENT_TOPIC_IDS,
    PRIOR_FIXED_SUPPORT_TOPIC_IDS,
    SEALED_HOLDOUT_TOPIC_IDS,
    build_topic_structure,
)
from scripts.clariq_topic_level_train_opportunity import (
    _load_split_archive,
    _load_tar_pickle,
    build_split,
    verify_source,
)


PARENT_MANIFEST_SHA256 = (
    "8871d72aca825aa5b34cf52295eb93fe79caf631c0bba070d82abf6c3bec698d"
)
V1_MECHANICS_TOPIC_ID = "38"
V2_MECHANICS_TOPIC_ID = "60"
EXPECTED_ROOT_COUNT = 15
EXPECTED_BRANCH_COUNT = 90
EXPECTED_FULL_TREE_REQUESTS = 91


def build_manifest(
    source_root: Path,
    parent_manifest_path: Path,
) -> dict[str, Any]:
    if (
        hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest()
        != PARENT_MANIFEST_SHA256
    ):
        raise ValueError("ClariQ dynamic-support parent manifest changed")
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent["status"] != "passed":
        raise ValueError("ClariQ dynamic-support parent did not pass")
    if parent["stages"]["mechanics"][0]["topic_id"] != V1_MECHANICS_TOPIC_ID:
        raise ValueError("ClariQ V1 mechanics topic changed")
    if tuple(
        task["topic_id"] for task in parent["stages"]["development"]
    ) != DEVELOPMENT_TOPIC_IDS:
        raise ValueError("ClariQ development topic changed")
    if tuple(parent["sealed_holdout"]["topic_ids"]) != (
        SEALED_HOLDOUT_TOPIC_IDS
    ):
        raise ValueError("ClariQ sealed holdout topics changed")

    paths = verify_source(source_root)
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    split = build_split(row["topic_id"] for row in metadata)
    if V2_MECHANICS_TOPIC_ID not in split["opportunity"]:
        raise ValueError("ClariQ V2 mechanics topic changed")
    forbidden = (
        set(PRIOR_FIXED_SUPPORT_TOPIC_IDS)
        | {V1_MECHANICS_TOPIC_ID}
        | set(DEVELOPMENT_TOPIC_IDS)
        | set(SEALED_HOLDOUT_TOPIC_IDS)
    )
    if V2_MECHANICS_TOPIC_ID in forbidden:
        raise ValueError("ClariQ V2 mechanics topic was previously consumed")
    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        if row["topic_id"] == V2_MECHANICS_TOPIC_ID:
            by_topic[row["topic_id"]].append(row)
    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )["NDCG20"]
    task = build_topic_structure(
        V2_MECHANICS_TOPIC_ID,
        by_topic[V2_MECHANICS_TOPIC_ID],
        synthetic,
        evaluation,
    )
    if task["root_count"] != EXPECTED_ROOT_COUNT:
        raise ValueError("ClariQ V2 root count changed")
    if task["branch_count"] != EXPECTED_BRANCH_COUNT:
        raise ValueError("ClariQ V2 branch count changed")
    if task["expected_model_requests"] != EXPECTED_FULL_TREE_REQUESTS:
        raise ValueError("ClariQ V2 request count changed")
    return {
        "schema_version": 1,
        "status": "passed",
        "protocol": {
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "v1_mechanics_topic_rerun": False,
            "v2_mechanics_selection_is_endpoint_disclosed": True,
            "development_content_or_utility_values_read": False,
            "holdout_content_or_utility_values_read": False,
            "transport_change_only": (
                "contiguous response codes -> single-space-separated codes"
            ),
            "api_calls": 0,
        },
        "mechanics_task": task,
        "development_reference": {
            "topic_ids": list(DEVELOPMENT_TOPIC_IDS),
            "content_reopened": False,
            "utility_values_read": False,
        },
        "sealed_holdout": {
            "topic_ids": list(SEALED_HOLDOUT_TOPIC_IDS),
            "content_emitted": False,
            "utility_values_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.source_root, args.parent_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    task = result["mechanics_task"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "topic_id": task["topic_id"],
                "roots": task["root_count"],
                "branches": task["branch_count"],
                "full_tree_requests": task["expected_model_requests"],
                "development_reference": result["development_reference"],
                "sealed_holdout": result["sealed_holdout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
