#!/usr/bin/env python3
"""Predicate-pushed structural screen for the frozen R2E-Gym prefix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts/r2e_gym_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("r2e_source_for_screen", SOURCE_PATH)
assert SPEC and SPEC.loader
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)

VERSION = "r2e-gym-debug-bed-structural-screen-v1"
PAYLOAD_COLUMNS = (
    "repo_name", "docker_image", "commit_hash", "parsed_commit_content",
    "modified_files", "modified_entity_summaries",
)
FORBIDDEN_COLUMNS = {
    "problem_statement", "prompt", "expected_output_json",
    "execution_result_content", "relevant_files",
}
ENTITY_PATTERN = re.compile(r"function|method|class", re.IGNORECASE)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def _is_test_path(value: str) -> bool:
    parts = [part.lower() for part in Path(value).parts]
    basename = parts[-1] if parts else ""
    return (
        any(part.startswith("test") for part in parts)
        or basename.endswith("_test.py")
        or basename == "conftest.py"
    )


def _qualify(row: dict[str, Any], expected_entities: int) -> tuple[bool, int, int]:
    try:
        parsed = json.loads(row["parsed_commit_content"])
    except (json.JSONDecodeError, TypeError):
        return False, 0, 0
    file_diffs = parsed.get("file_diffs") if isinstance(parsed, dict) else None
    modified_files = [str(value).strip() for value in (row["modified_files"] or []) if str(value).strip()]
    if not isinstance(file_diffs, list) or not file_diffs or not 1 <= len(set(modified_files)) <= 2:
        return False, 0, len(set(modified_files))
    identities: set[tuple[str, str, int, int]] = set()
    for entity in row["modified_entity_summaries"] or []:
        if not isinstance(entity, dict):
            continue
        kind = f"{entity.get('type', '')} {entity.get('ast_type_str', '')}"
        file_name = str(entity.get("file_name") or "").strip()
        name = str(entity.get("name") or "").strip()
        start = entity.get("start_lineno")
        end = entity.get("end_lineno")
        if (
            ENTITY_PATTERN.search(kind)
            and file_name in modified_files
            and not _is_test_path(file_name)
            and name
            and isinstance(start, int)
            and isinstance(end, int)
            and 1 <= start <= end
        ):
            identities.add((file_name, name, start, end))
    count = len(identities)
    return 2 <= count <= 4 and count == expected_entities, count, len(set(modified_files))


def _metadata_selection(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for rel, _, _ in SOURCE.SHARDS:
        parquet = pq.ParquetFile(data_root / rel)
        table = parquet.read(columns=list(SOURCE.ALLOWED_COLUMNS))
        columns = table.to_pydict()
        rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))
    commit_counts = Counter(str(row["commit_hash"] or "") for row in rows)
    image_counts = Counter(str(row["docker_image"] or "") for row in rows)
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    eligible = [
        row for row in rows
        if str(row["repo_name"] or "").strip()
        and str(row["docker_image"] or "").strip()
        and commit_pattern.fullmatch(str(row["commit_hash"] or ""))
        and 1 <= int(row["num_non_test_files"]) <= 2
        and 2 <= int(row["num_non_test_func_methods"]) <= 4
        and 4 <= int(row["num_non_test_lines"]) <= 80
        and commit_counts[str(row["commit_hash"])] == 1
        and image_counts[str(row["docker_image"])] == 1
    ]
    ordered = sorted(
        eligible,
        key=lambda row: digest((SOURCE.SALT + row["commit_hash"] + "|" + row["docker_image"]).encode()),
    )
    return ordered, ordered[:16]


def _selected_payload(data_root: Path, selected_ids: list[str]) -> list[dict[str, Any]]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    paths = [str(data_root / rel) for rel, _, _ in SOURCE.SHARDS]
    dataset = ds.dataset(paths, format="parquet")
    scanner = dataset.scanner(
        columns=list(PAYLOAD_COLUMNS),
        filter=pc.field("commit_hash").isin(selected_ids),
    )
    table = scanner.to_table()
    columns = table.to_pydict()
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def screen(data_root: Path, protocol: Path, source_manifest: Path, source_result: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest_value = json.loads(source_manifest.read_text())
    source_result_value = json.loads(source_result.read_text())
    ordered, selected = _metadata_selection(data_root)
    selected_ids = [str(row["commit_hash"]) for row in selected]
    expected = {str(row["commit_hash"]): int(row["num_non_test_func_methods"]) for row in selected}
    payload_rows = _selected_payload(data_root, selected_ids)
    by_id = {str(row["commit_hash"]): row for row in payload_rows}
    qualifications: list[dict[str, Any]] = []
    for commit_hash in selected_ids:
        row = by_id.get(commit_hash)
        qualified, entity_count, file_count = _qualify(row, expected[commit_hash]) if row else (False, 0, 0)
        qualifications.append({
            "commit_hash": commit_hash,
            "qualified": qualified,
            "entity_count": entity_count,
            "file_count": file_count,
        })
    qualifying = [item for item in qualifications if item["qualified"]]
    mechanics = qualifying[:8]
    source_hash = source_manifest_value["selection"]["split_ordered_id_sha256"]["structural_screen"]
    gates = {
        "source_v3_pass_bound": source_result_value.get("status") == "source_pass"
        and all(source_result_value.get("gates", {}).values()),
        "exact_source_prefix": len(selected_ids) == 16
        and digest(canonical(selected_ids)) == source_hash,
        "predicate_payload_exactness": len(payload_rows) == 16
        and len(by_id) == 16
        and set(by_id) == set(selected_ids),
        "minimum_structural_cohort": len(qualifying) >= 8 and len(mechanics) == 8,
        "payload_projection_boundary": not (FORBIDDEN_COLUMNS & set(PAYLOAD_COLUMNS)),
    }
    public = {
        "protocol_version": VERSION,
        "bindings": {
            "protocol_sha256": file_digest(protocol),
            "source_manifest_file_sha256": file_digest(source_manifest),
            "source_result_file_sha256": file_digest(source_result),
            "source_structural_prefix_sha256": source_hash,
        },
        "screen": {
            "screened_count": len(qualifications),
            "qualifying_count": len(qualifying),
            "mechanics_count": len(mechanics),
            "mechanics_ordered_id_sha256": digest(canonical([item["commit_hash"] for item in mechanics])),
            "qualifying_entity_count_histogram": dict(sorted(Counter(item["entity_count"] for item in qualifying).items())),
            "qualifying_file_count_histogram": dict(sorted(Counter(item["file_count"] for item in qualifying).items())),
        },
        "privacy": {
            "payload_rows_materialized": len(payload_rows),
            "nonselected_payload_rows_materialized": 0,
            "identifiers_serialized_publicly": False,
            "task_text_opened": False,
            "test_expectations_opened": False,
            "execution_logs_opened": False,
            "relevant_source_opened": False,
            "endpoints_opened": False,
        },
    }
    gates["public_shape"] = set(public) == {"protocol_version", "bindings", "screen", "privacy"}
    passed = all(gates.values())
    result = {
        **public,
        "status": "structural_screen_pass" if passed else "structural_screen_failed_closed",
        "decision": "mechanics_protocol_authorized" if passed else "close_exact_r2e_gym_debug_bed_route",
        "gates": gates,
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "separately_frozen_eight_task_mechanics_only" if passed else "nothing",
    }
    private = {
        "protocol_version": VERSION,
        "selected_ordered_ids": selected_ids,
        "qualifications": qualifications,
        "mechanics_ordered_ids": [item["commit_hash"] for item in mechanics],
        "public_result_sha256": digest(canonical(result)),
    }
    return result, private


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    result, private = screen(args.data_root, args.protocol, args.source_manifest, args.source_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "structural_screen_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
