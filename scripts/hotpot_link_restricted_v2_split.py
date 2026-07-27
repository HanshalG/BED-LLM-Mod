#!/usr/bin/env python3
"""Freeze and screen fresh Hotpot V2 development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hotpot_future_uplift_confirmation import qualification, selected_rows
from scripts.hotpot_link_restricted_manifest import (
    _restricted_diagnostic,
    _source_splits,
    ordered_list_hash,
)


INTERFACE_VERSION = "hotpot-link-restricted-v2-split-1"
V2_DEVELOPMENT_SIZE = 500
V2_CONFIRMATION_SIZE = 1_500
SELECTION_COUNT = 20


def v2_splits(paths: Sequence[Path]) -> dict[str, list[str]]:
    _metadata, splits = _source_splits(paths)
    source = splits["confirmation"]
    if len(source) != V2_DEVELOPMENT_SIZE + V2_CONFIRMATION_SIZE:
        raise ValueError("V2 source confirmation split has unexpected size")
    return {
        "development": source[:V2_DEVELOPMENT_SIZE],
        "confirmation": source[V2_DEVELOPMENT_SIZE:],
    }


def manifest(paths: Sequence[Path]) -> dict[str, Any]:
    splits = v2_splits(paths)
    gates = {
        "exact_sizes": {
            name: len(values)
            for name, values in splits.items()
        }
        == {
            "development": V2_DEVELOPMENT_SIZE,
            "confirmation": V2_CONFIRMATION_SIZE,
        },
        "disjoint": not (
            set(splits["development"]) & set(splits["confirmation"])
        ),
        "zero_endpoint_rows_materialized": True,
        "zero_model_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "endpoint_rows_materialized": 0,
            "model_calls": 0,
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


def opportunity(paths: Sequence[Path]) -> dict[str, Any]:
    splits = v2_splits(paths)
    rows = selected_rows(paths, splits["development"])
    diagnostics = [_restricted_diagnostic(row) for row in rows]
    qualifying = [row for row in diagnostics if row["qualifies"]]
    selected = qualifying[:SELECTION_COUNT]
    gates = {
        "exact_500_development_rows_materialized": len(rows) == 500,
        "at_least_20_qualifying": len(qualifying) >= SELECTION_COUNT,
        "all_qualifying_unique_enabling_optimum": all(
            row["unique_enabling_optimum"] for row in qualifying
        ),
        "v2_confirmation_endpoints_unopened": True,
        "zero_model_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "development_split_sha256": ordered_list_hash(
                splits["development"]
            ),
            "confirmation_split_sha256": ordered_list_hash(
                splits["confirmation"]
            ),
            "development_endpoint_rows_materialized": 500,
            "confirmation_endpoint_rows_materialized": 0,
            "model_calls": 0,
        },
        "metrics": {
            "qualifying_count": len(qualifying),
            "malformed_count": sum(row["malformed"] for row in diagnostics),
            "selected_task_ids": [row["task_id"] for row in selected],
            "selected_id_sha256": ordered_list_hash(
                [row["task_id"] for row in selected]
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
