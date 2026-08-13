#!/usr/bin/env python3
"""Privately materialize the frozen V3 structural screen and bank aggregates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_module("swesmith_v1_for_structural", ROOT / "scripts/swesmith_debug_bed_source_audit.py")
V3 = load_module("swesmith_v3_for_structural", ROOT / "scripts/swesmith_debug_bed_source_v3_audit.py")
VERSION = "swesmith-debug-bed-v3-structural-screen-v1"


def hunk_count(patch: str) -> int:
    return sum(line.startswith("@@ ") for line in patch.splitlines())


def ordered_hash(values: list[str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select(rows: list[dict[str, Any]], debug_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    by_id = {row["instance_id"]: row for row in rows}
    config = V1.parse_simple_yaml_lists(debug_root / V1.BOUND_FILES["split_config"][0])
    excluded = set(config["excluded"])
    development = {value for value in config["train-789"] if value not in excluded}
    confirmation = {value for value in config["test-125"] if value not in excluded}
    composed = [
        row["instance_id"] for row in rows
        if ".combine_file__" in row["instance_id"] or ".combine_module__" in row["instance_id"]
    ]
    v2_pool = [value for value in composed if value not in excluded | development | confirmation]
    v2_ordered = sorted(v2_pool, key=lambda value: V3.digest((V1.SPLIT_SALT + value).encode()))
    image_rows: dict[str, list[str]] = defaultdict(list)
    for value in v2_ordered[72:]:
        row = by_id[value]
        image = str(row["image_name"] or "").strip()
        if image and row["FAIL_TO_PASS"] and row["PASS_TO_PASS"]:
            image_rows[image].append(value)
    selected_image, selected_pool = sorted(
        image_rows.items(), key=lambda item: (-len(item[1]), V3.digest(item[0].encode()), item[0])
    )[0]
    selected_ordered = sorted(selected_pool, key=lambda value: V3.digest((V3.SALT + value).encode()))
    screening = [by_id[value] for value in selected_ordered[:12]]
    mechanics = [row for row in screening if 2 <= hunk_count(row["patch"]) <= 4][:8]
    return screening, mechanics, selected_image


def run(data_root: Path, debug_root: Path, private_path: Path, public_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for rel in V1.SHARDS:
        table = pq.read_table(data_root / rel)
        columns = table.to_pydict()
        rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))
    screening, mechanics, image = select(rows, debug_root)
    counts = [hunk_count(row["patch"]) for row in screening]
    private = {
        "protocol_version": VERSION,
        "screening_rows": screening,
        "mechanics_rows": mechanics,
        "selected_image": image,
    }
    public = {
        "protocol_version": VERSION,
        "status": "structural_pass" if len(mechanics) == 8 else "structural_failed_closed",
        "decision": "mechanics_protocol_may_be_frozen" if len(mechanics) == 8 else "close_exact_swesmith_debug_bed_v3",
        "screening_count": len(screening),
        "hunk_counts": counts,
        "eligible_count": sum(2 <= count <= 4 for count in counts),
        "mechanics_slot_count": len(mechanics),
        "mechanics_ordered_id_sha256": ordered_hash([row["instance_id"] for row in mechanics]),
        "selected_image_sha256": hashlib.sha256(image.encode()).hexdigest(),
        "privacy": {
            "instance_ids_serialized": False, "image_name_serialized": False,
            "repositories_serialized": False, "patches_serialized": False,
            "problems_serialized": False, "test_names_serialized": False,
            "source_files_serialized": False, "endpoints_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "freeze_zero_call_mechanics_protocol_only" if len(mechanics) == 8 else "nothing",
    }
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private, sort_keys=True) + "\n")
    public_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--private-result", type=Path, required=True)
    parser.add_argument("--public-result", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_root, args.debug_root, args.private_result, args.public_result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "structural_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
