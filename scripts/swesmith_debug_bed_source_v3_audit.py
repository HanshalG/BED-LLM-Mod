#!/usr/bin/env python3
"""Metadata-only native-amd64 V3 split for SWE-smith Debug-BED."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "scripts/swesmith_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_v1_for_v3", V1_PATH)
assert SPEC and SPEC.loader
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

VERSION = "swesmith-debug-bed-source-v3-native-amd64"
SALT = "swesmith-debug-bed-v3-native-amd64|"
PREFLIGHT_COMMIT = "e0c8c718598bb2cde638516fecdc4b7b2deac7e1"
PREFLIGHT_RESULT_COMMIT = "5150d0da"
WHEEL_SHA256 = "149d3fc54bcb0d3b3010121e7de4a13e5e6f31bee08be5918dea1f7b0a8dd5ef"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def ordered_hash(values: list[str]) -> str:
    return digest(canonical(values))


def audit(debug_root: Path, data_root: Path, protocol: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for rel in V1.SHARDS:
        table = pq.read_table(data_root / rel, columns=["instance_id", "image_name", "FAIL_TO_PASS", "PASS_TO_PASS"])
        columns = table.to_pydict()
        rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))
    by_id = {row["instance_id"]: row for row in rows}
    config = V1.parse_simple_yaml_lists(debug_root / V1.BOUND_FILES["split_config"][0])
    excluded = set(config["excluded"])
    development = [value for value in config["train-789"] if value not in excluded]
    confirmation = [value for value in config["test-125"] if value not in excluded]
    composed = [row["instance_id"] for row in rows if ".combine_file__" in row["instance_id"] or ".combine_module__" in row["instance_id"]]
    unavailable = excluded | set(development) | set(confirmation)
    v2_pool = [value for value in composed if value not in unavailable]
    v2_ordered = sorted(v2_pool, key=lambda value: digest((V1.SPLIT_SALT + value).encode()))
    v2_mechanics = v2_ordered[:8]
    v2_opportunity = v2_ordered[8:72]
    v2_reserve = v2_ordered[72:]
    image_rows: dict[str, list[str]] = defaultdict(list)
    for value in v2_reserve:
        row = by_id[value]
        image = str(row["image_name"] or "").strip()
        if image and row["FAIL_TO_PASS"] and row["PASS_TO_PASS"]:
            image_rows[image].append(value)
    selected_image, selected_pool = sorted(
        image_rows.items(), key=lambda item: (-len(item[1]), digest(item[0].encode()), item[0])
    )[0]
    selected_ordered = sorted(selected_pool, key=lambda value: digest((SALT + value).encode()))
    screening = selected_ordered[:12]
    preflight = json.loads((ROOT / "results/nonmyopic/swesmith_debug_bed_native_amd64_preflight/PREFLIGHT_RESULT.json").read_text())
    disallowed = set(v2_mechanics) | set(v2_opportunity) | unavailable
    gates = {
        "exact_source_bindings": V1.git_value(debug_root, "HEAD") == V1.DEBUG_GYM_COMMIT
        and V1.git_value(debug_root, "HEAD^{tree}") == V1.DEBUG_GYM_TREE
        and all(file_digest(data_root / rel) == expected[1] for rel, expected in V1.SHARDS.items()),
        "native_preflight_bound": preflight.get("status") == "preflight_pass"
        and preflight.get("bindings", {}).get("commit") == PREFLIGHT_COMMIT
        and preflight.get("runner") == {"host_machine": "x86_64", "docker_os": "linux", "docker_arch": "x86_64"}
        and all(preflight.get("gates", {}).values()),
        "single_image_screening_prefix": len(screening) == 12
        and len(set(screening)) == 12
        and all(by_id[value]["image_name"] == selected_image for value in screening),
        "fresh_reserve_only": set(screening) <= set(v2_reserve) and not (set(screening) & disallowed),
        "composed_with_tests": all(
            (".combine_file__" in value or ".combine_module__" in value)
            and by_id[value]["FAIL_TO_PASS"] and by_id[value]["PASS_TO_PASS"]
            for value in screening
        ),
    }
    manifest = {
        "protocol_version": VERSION,
        "bindings": {
            "debuggym_commit": V1.DEBUG_GYM_COMMIT,
            "debuggym_tree": V1.DEBUG_GYM_TREE,
            "swesmith_revision": V1.DATA_REVISION,
            "source_v2_manifest_file_sha256": file_digest(ROOT / "results/nonmyopic/swesmith_debug_bed_source_v2/MANIFEST.json"),
            "native_preflight_result_file_sha256": file_digest(ROOT / "results/nonmyopic/swesmith_debug_bed_native_amd64_preflight/PREFLIGHT_RESULT.json"),
            "native_preflight_result_commit": PREFLIGHT_RESULT_COMMIT,
            "swesmith_0_0_4_wheel_sha256": WHEEL_SHA256,
        },
        "selection": {
            "salt_sha256": digest(SALT.encode()),
            "reserve_count": len(v2_reserve),
            "candidate_image_count": len(image_rows),
            "selected_image_sha256": digest(selected_image.encode()),
            "selected_image_eligible_count": len(selected_pool),
            "screening_count": len(screening),
            "screening_ordered_id_sha256": ordered_hash(screening),
        },
        "privacy": {
            "instance_ids_serialized": False, "image_names_serialized": False,
            "repositories_serialized": False, "patches_opened": False,
            "problem_statements_opened": False, "test_names_opened": False,
            "source_files_opened": False, "endpoints_opened": False,
        },
    }
    gates["public_manifest_shape"] = set(manifest) == {"protocol_version", "bindings", "selection", "privacy"}
    passed = all(gates.values())
    result = {
        "protocol_version": VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "structural_screen_authorized" if passed else "close_exact_swesmith_debug_bed_source_v3",
        "protocol_sha256": file_digest(protocol),
        "manifest_sha256": digest(canonical(manifest)),
        "gates": gates,
        "privacy": manifest["privacy"],
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "private_structural_screen_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.debug_root, args.data_root, args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
