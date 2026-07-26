#!/usr/bin/env python3
"""Freeze target-blind cohorts for a BIRD-derived SQL-intent BED task."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

try:
    from scripts import bird_interact_source_audit as source_audit
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    import bird_interact_source_audit as source_audit


SELECTION_SEED = 24_414
EXPOSED_TASK_IDS = frozenset({"alien_1", "alien_2", "alien_3"})
SPLIT_SIZES = {
    "mechanics": 3,
    "opportunity": 24,
    "development": 12,
}
MINIMUM_ELIGIBLE_TASKS = 60
GROUNDING_TYPES = frozenset(
    {"knowledge_linking_ambiguity", "schema_linking_ambiguity"}
)
SEMANTIC_TYPES = frozenset(
    {
        "intent_ambiguity",
        "lexical_ambiguity",
        "semantic_ambiguity",
        "syntactic_ambiguity",
    }
)

# Filled after the first content-sealed invocation and enforced on replay.
EXPECTED_SOURCE_TASKS: int | None = None
EXPECTED_ELIGIBLE_TASKS: int | None = None
EXPECTED_SPLIT_HASHES: dict[str, str] | None = None
EXPECTED_MECHANICS_IDS: tuple[str, ...] | None = None


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(values).encode("utf-8")
    ).hexdigest()


def eligible_task(row: dict[str, Any]) -> bool:
    task_id = str(row.get("instance_id", ""))
    if not task_id or task_id in EXPOSED_TASK_IDS:
        return False
    if row.get("sol_sql") or row.get("test_cases") or row.get("follow_up"):
        return False
    critical = (
        row.get("user_query_ambiguity", {})
        .get("critical_ambiguity", [])
    )
    if not 3 <= len(critical) <= 4:
        return False
    if sum(item.get("is_mask") is True for item in critical) < 2:
        return False
    types = {str(item.get("type", "")) for item in critical}
    if not types.intersection(GROUNDING_TYPES):
        return False
    if not types.intersection(SEMANTIC_TYPES):
        return False
    if not row.get("knowledge_ambiguity"):
        return False
    normalized_terms = [
        " ".join(str(item.get("term", "")).lower().split())
        for item in critical
    ]
    if any(not term for term in normalized_terms):
        return False
    if len(set(normalized_terms)) != len(normalized_terms):
        return False
    return all(str(item.get("sql_snippet", "")).strip() for item in critical)


def split_tasks(rows: list[dict[str, Any]]) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, Any]],
]:
    eligible = {
        str(row["instance_id"]): row
        for row in rows
        if eligible_task(row)
    }
    task_ids = sorted(eligible)
    random.Random(SELECTION_SEED).shuffle(task_ids)
    cursor = 0
    splits: dict[str, list[str]] = {}
    for name, size in SPLIT_SIZES.items():
        splits[name] = task_ids[cursor : cursor + size]
        cursor += size
    splits["holdout"] = task_ids[cursor:]
    return splits, eligible


def _split_summary(
    task_ids: list[str],
    eligible: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [eligible[task_id] for task_id in task_ids]
    return {
        "task_ids": task_ids,
        "ordered_sha256": ordered_hash(task_ids),
        "database_counts": dict(
            sorted(
                Counter(
                    str(row["selected_database"]) for row in rows
                ).items()
            )
        ),
        "critical_count_histogram": dict(
            sorted(
                Counter(
                    len(
                        row["user_query_ambiguity"][
                            "critical_ambiguity"
                        ]
                    )
                    for row in rows
                ).items()
            )
        ),
    }


def build_manifest(
    data_path: Path,
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    digest = source_audit.sha256_file(data_path)
    if digest != source_audit.EXPECTED_DATA_SHA256:
        raise ValueError("Mini-Interact source hash changed")
    rows = source_audit.load_jsonl(data_path)
    splits, eligible = split_tasks(rows)
    split_hashes = {
        name: ordered_hash(task_ids)
        for name, task_ids in splits.items()
    }
    mechanics_ids = tuple(splits["mechanics"])

    if enforce_frozen:
        if any(
            value is None
            for value in (
                EXPECTED_SOURCE_TASKS,
                EXPECTED_ELIGIBLE_TASKS,
                EXPECTED_SPLIT_HASHES,
                EXPECTED_MECHANICS_IDS,
            )
        ):
            raise ValueError("BIRD intent-world constants are not frozen")
        if len(rows) != EXPECTED_SOURCE_TASKS:
            raise ValueError("BIRD source task count changed")
        if len(eligible) != EXPECTED_ELIGIBLE_TASKS:
            raise ValueError("BIRD eligible task count changed")
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("BIRD split hashes changed")
        if mechanics_ids != EXPECTED_MECHANICS_IDS:
            raise ValueError("BIRD mechanics IDs changed")

    gates = {
        "source_hash_matches": (
            digest == source_audit.EXPECTED_DATA_SHA256
        ),
        "at_least_60_eligible_tasks": (
            len(eligible) >= MINIMUM_ELIGIBLE_TASKS
        ),
        "all_splits_nonempty": all(splits.values()),
        "mechanics_has_three_databases": (
            len(
                {
                    str(eligible[task_id]["selected_database"])
                    for task_id in splits["mechanics"]
                }
            )
            == 3
        ),
    }
    return {
        "interface_version": "bird-intent-world-manifest-1",
        "source": {
            "repository": source_audit.SOURCE_REPOSITORY,
            "commit": source_audit.SOURCE_COMMIT,
            "dataset_repository": source_audit.DATASET_REPOSITORY,
            "dataset_revision": source_audit.DATASET_REVISION,
            "data_sha256": digest,
            "source_tasks": len(rows),
        },
        "construction": {
            "selection_seed": SELECTION_SEED,
            "excluded_semantically_inspected_task_ids": sorted(
                EXPOSED_TASK_IDS
            ),
            "critical_ambiguity_count_range": [3, 4],
            "minimum_masked_critical_ambiguities": 2,
            "requires_grounding_and_semantic_types": True,
            "requires_knowledge_ambiguity": True,
            "eligible_tasks": len(eligible),
            "minimum_eligible_tasks": MINIMUM_ELIGIBLE_TASKS,
        },
        "splits": {
            name: _split_summary(task_ids, eligible)
            for name, task_ids in splits.items()
        },
        "gates": gates,
        "passed": all(gates.values()),
        "query_content_emitted": False,
        "ambiguity_content_emitted": False,
        "sql_content_emitted": False,
        "endpoint_fields_read": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-unfrozen-constants", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(
        args.data,
        enforce_frozen=not args.show_unfrozen_constants,
    )
    if not args.show_unfrozen_constants:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
