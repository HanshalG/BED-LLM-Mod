#!/usr/bin/env python3
"""Freeze and screen fresh Hotpot link-restricted BED splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hotpot_causal_belief_smoke import _answer_and_enabling_titles
from scripts.hotpot_directional_unlock_audit import mentioned_context_titles
from scripts.hotpot_future_uplift_confirmation import (
    SPLIT_HASHES,
    metadata_splits,
    qualification,
    selected_rows,
)


INTERFACE_VERSION = "hotpot-link-restricted-split-1"
OPPORTUNITY_INTERFACE_VERSION = "hotpot-link-restricted-opportunity-2"
SPLIT_SIZES = {
    "mechanics": 100,
    "development": 500,
    "confirmation": 2_000,
    "holdout": 68_791,
}
DEVELOPMENT_SELECTION_COUNT = 20


def ordered_list_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def link_restricted_splits(
    old_holdout_ids: Sequence[str],
) -> dict[str, list[str]]:
    if len(old_holdout_ids) != sum(SPLIT_SIZES.values()):
        raise ValueError("old Hotpot holdout has unexpected size")
    splits: dict[str, list[str]] = {}
    offset = 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = list(old_holdout_ids[offset : offset + size])
        offset += size
    if offset != len(old_holdout_ids):
        raise AssertionError("link-restricted split is not exhaustive")
    return splits


def _source_splits(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    metadata, old_splits = metadata_splits(paths)
    old_holdout = old_splits["holdout"]
    if ordered_list_hash(old_holdout) != SPLIT_HASHES["holdout"]:
        raise ValueError("old sealed holdout hash does not reproduce")
    return metadata, link_restricted_splits(old_holdout)


def manifest(paths: Sequence[Path]) -> dict[str, Any]:
    metadata, splits = _source_splits(paths)
    all_ids = [task_id for values in splits.values() for task_id in values]
    gates = {
        "old_holdout_exact_71391": len(all_ids) == 71_391,
        "new_splits_disjoint": len(set(all_ids)) == len(all_ids),
        "new_splits_exhaust_old_holdout": len(all_ids)
        == sum(SPLIT_SIZES.values()),
        "only_metadata_columns_read": True,
        "zero_endpoint_rows_materialized": True,
        "zero_model_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "old_holdout_ordered_id_sha256": SPLIT_HASHES["holdout"],
            "metadata_rows": len(metadata),
            "endpoint_rows_materialized": 0,
            "model_calls": 0,
            "cost_usd": 0.0,
        },
        "splits": {
            name: {
                "count": len(values),
                "ordered_id_sha256": ordered_list_hash(values),
            }
            for name, values in splits.items()
        },
        "gates": gates,
    }


def _restricted_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    try:
        diagnostic = qualification(row)
    except ValueError as exc:
        return {
            "task_id": str(row["id"]),
            "qualifies": False,
            "malformed": True,
            "malformed_error": str(exc),
        }
    if not diagnostic["qualifies"]:
        return {
            "task_id": str(row["id"]),
            "qualifies": False,
            "malformed": False,
        }
    titles = [str(value) for value in row["context"]["title"]]
    paragraphs = [
        " ".join(str(sentence) for sentence in sentences)
        for sentences in row["context"]["sentences"]
    ]
    answer_title, enabling_title = _answer_and_enabling_titles(row)
    root_values = []
    candidate_counts = []
    root_roles = []
    for context_index in diagnostic["root_context_indices"]:
        root_title = titles[context_index]
        candidates = mentioned_context_titles(
            paragraph_text=paragraphs[context_index],
            context_titles=titles,
            root_title=root_title,
        )
        candidate_counts.append(len(candidates))
        root_roles.append(
            "enabling"
            if root_title == enabling_title
            else "answer"
            if root_title == answer_title
            else "distractor"
        )
        possible = [root_title] + candidates
        root_values.append(
            max(
                len({root_title, followup} & {answer_title, enabling_title})
                for followup in possible
            )
        )
    best = max(root_values)
    optimal = [
        index for index, value in enumerate(root_values) if value == best
    ]
    enabling_index = root_roles.index("enabling")
    return {
        "task_id": str(row["id"]),
        "qualifies": True,
        "malformed": False,
        "root_values": root_values,
        "candidate_counts": candidate_counts,
        "root_roles": root_roles,
        "unique_enabling_optimum": optimal == [enabling_index] and best == 2,
    }


def opportunity(paths: Sequence[Path]) -> dict[str, Any]:
    _metadata, splits = _source_splits(paths)
    rows_by_split = {
        name: selected_rows(paths, splits[name])
        for name in ("mechanics", "development")
    }
    diagnostics = {
        name: [_restricted_diagnostic(row) for row in rows]
        for name, rows in rows_by_split.items()
    }
    qualifying = {
        name: [row for row in values if row["qualifies"]]
        for name, values in diagnostics.items()
    }
    selected_development = qualifying["development"][
        :DEVELOPMENT_SELECTION_COUNT
    ]
    gates = {
        "exact_100_mechanics_rows_materialized": len(
            rows_by_split["mechanics"]
        )
        == 100,
        "exact_500_development_rows_materialized": len(
            rows_by_split["development"]
        )
        == 500,
        "mechanics_has_at_least_3_qualifying": len(
            qualifying["mechanics"]
        )
        >= 3,
        "development_has_at_least_20_qualifying": len(
            qualifying["development"]
        )
        >= DEVELOPMENT_SELECTION_COUNT,
        "all_qualifying_have_unique_enabling_optimum": all(
            row["unique_enabling_optimum"]
            for values in qualifying.values()
            for row in values
        ),
        "confirmation_and_holdout_endpoints_unopened": True,
        "zero_model_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": OPPORTUNITY_INTERFACE_VERSION,
            "mechanics_split_sha256": ordered_list_hash(
                splits["mechanics"]
            ),
            "development_split_sha256": ordered_list_hash(
                splits["development"]
            ),
            "confirmation_split_sha256": ordered_list_hash(
                splits["confirmation"]
            ),
            "holdout_split_sha256": ordered_list_hash(splits["holdout"]),
            "endpoint_rows_materialized": 600,
            "confirmation_endpoint_rows_materialized": 0,
            "holdout_endpoint_rows_materialized": 0,
            "model_calls": 0,
            "cost_usd": 0.0,
        },
        "metrics": {
            "qualifying_counts": {
                name: len(values) for name, values in qualifying.items()
            },
            "malformed_counts": {
                name: sum(row["malformed"] for row in values)
                for name, values in diagnostics.items()
            },
            "candidate_count_range": {
                name: [
                    min(
                        count
                        for row in values
                        for count in row["candidate_counts"]
                    ),
                    max(
                        count
                        for row in values
                        for count in row["candidate_counts"]
                    ),
                ]
                for name, values in qualifying.items()
            },
            "selected_development_task_ids": [
                row["task_id"] for row in selected_development
            ],
            "selected_development_id_sha256": ordered_list_hash(
                [row["task_id"] for row in selected_development]
            ),
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("manifest", "opportunity"), required=True
    )
    parser.add_argument(
        "--train-shard", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.train_shard) != 2:
        parser.error("exactly two --train-shard values are required")
    payload = (
        manifest(args.train_shard)
        if args.stage == "manifest"
        else opportunity(args.train_shard)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
