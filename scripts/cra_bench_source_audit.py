#!/usr/bin/env python3
"""Audit the released CRA-Bench package without emitting task content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


SOURCE_REPOSITORY = "https://huggingface.co/datasets/l1i1p/CRA-Bench"
SOURCE_REVISION = "30d6c7c7e67e7dd21904c1aae67aa6fc8d596640"
SPLIT_SHA256 = {
    "easy": "dc5e4620fca19988488ecf6979e331ed4577929eb3343d2cb0d0fcfb763b1d82",
    "medium": "d30a9539ba634fb8e6247f28d4448f14e451432adee74cb7298ce4bc9ef9ee82",
    "hard": "4d86ecbfc7da92ddc9c5e776606d3a062d5878da9b6e02681e45751f6c9fb6da",
}
EXPECTED_SPLIT_ROWS = 250
EXPECTED_UNIQUE_TARGETS = 244
EXPECTED_DOMAINS = {
    "Clothing_Shoes_and_Jewelry": 60,
    "Electronics": 40,
    "Grocery_and_Gourmet_Food": 36,
    "Health_and_Household": 55,
    "Home_and_Kitchen": 59,
}
EXPECTED_RELEASE_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "data/easy.jsonl",
    "data/hard.jsonl",
    "data/medium.jsonl",
    "metadata/data_format.md",
    "metadata/difficulty_stats.json",
    "metadata/evaluation_protocol.md",
    "metadata/sample_task.json",
    "scripts/validate_dataset.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split, expected_hash in SPLIT_SHA256.items():
        path = root / "data" / f"{split}.jsonl"
        if sha256_file(path) != expected_hash:
            raise ValueError(f"CRA-Bench {split} source hash changed")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != EXPECTED_SPLIT_ROWS:
            raise ValueError(f"CRA-Bench {split} row count changed")
        result[split] = rows
    return result


def analyze_rows(
    splits: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    by_base_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rows in splits.values():
        for row in rows:
            by_base_user[int(row["base_user_index"])].append(row)
    hard_rows = list(splits["hard"])
    target_ids = {
        str(row["fuzzy_gt"]["evaluation_only"]["target_asin"])
        for row in hard_rows
    }
    cross_difficulty = {
        "base_users_with_three_variants": sum(
            len(rows) == 3 for rows in by_base_user.values()
        ),
        "same_user_profile": sum(
            len(
                {
                    json.dumps(row["user_profile"], sort_keys=True)
                    for row in rows
                }
            )
            == 1
            for rows in by_base_user.values()
        ),
        "same_recommender_profile": sum(
            len(
                {
                    json.dumps(row["recsys_profile"], sort_keys=True)
                    for row in rows
                }
            )
            == 1
            for rows in by_base_user.values()
        ),
        "same_target": sum(
            len(
                {
                    str(
                        row["fuzzy_gt"]["evaluation_only"]["target_asin"]
                    )
                    for row in rows
                }
            )
            == 1
            for rows in by_base_user.values()
        ),
    }
    return {
        "split_rows": {
            split: len(rows) for split, rows in splits.items()
        },
        "base_users": len(by_base_user),
        "unique_targets": len(target_ids),
        "hard_domain_counts": dict(
            sorted(Counter(row["domain"] for row in hard_rows).items())
        ),
        "hard_patience_budget_counts": {
            str(key): value
            for key, value in sorted(
                Counter(
                    row["task"]["behavior_profile"]["patience_budget"]
                    for row in hard_rows
                ).items()
            )
        },
        "cross_difficulty": cross_difficulty,
    }


def build_audit(root: Path) -> dict[str, Any]:
    splits = load_splits(root)
    summary = analyze_rows(splits)
    if summary["unique_targets"] != EXPECTED_UNIQUE_TARGETS:
        raise ValueError("CRA-Bench unique target count changed")
    if summary["hard_domain_counts"] != EXPECTED_DOMAINS:
        raise ValueError("CRA-Bench hard domain counts changed")
    release_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    if release_files != EXPECTED_RELEASE_FILES:
        raise ValueError("CRA-Bench release file inventory changed")
    return {
        "interface_version": "cra-bench-source-audit-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "split_sha256": SPLIT_SHA256,
        },
        "summary": summary,
        "release_components": {
            "task_files": True,
            "exact_evaluation_labels": True,
            "user_profiles": True,
            "recommender_profiles": True,
            "user_simulator_implementation": False,
            "user_simulator_prompt": False,
            "retrieval_runner": False,
            "catalog_reconstruction_script": False,
            "product_catalog": False,
            "reference_policy": False,
            "paper_link": False,
        },
        "bed_boundary": {
            "hard_split_has_two_turn_patience": True,
            "alternative_world_prior_released": False,
            "question_conditioned_response_map_released": False,
            "using_evaluation_targets_as_policy_support_would_leak": True,
            "decision": "close_paid_route_until_runner_and_catalog_release",
        },
        "llm_calls": 0,
        "task_content_emitted": False,
        "target_ids_emitted": False,
        "target_metadata_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": audit["summary"],
                "bed_boundary": audit["bed_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
