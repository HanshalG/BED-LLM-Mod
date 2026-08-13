#!/usr/bin/env python3
"""Admit the frozen HiddenBench reserve-prefix cohort without public values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "hiddenbench-dynamic-belief-reserve-source-v1"
SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
SOURCE_PROTOCOL_SHA256 = "5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28"
SOURCE_MANIFEST_SHA256 = "b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d"
SOURCE_AUDIT_SHA256 = "67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9"
PROTOCOL_SHA256 = "f7e414eb844cbcfb8974254b54691fff5c8350fe1efefbe61473dab573ad83a3"
SALT = "hiddenbench-adaptive-elicitation-v1|"
EXPECTED_PREFIX_SHA256 = "adcbabc78acc15d7b88d2b0552636525ea64d25840113033a469c558bae3807a"
SPLIT_SIZES = (4, 12, 16, 24, 9)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_matches = file_digest(source_path) == SOURCE_SHA256
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    population_valid = (
        isinstance(rows, list)
        and len(rows) == sum(SPLIT_SIZES)
        and all(isinstance(row, dict) and "id" in row for row in rows)
        and len({str(row["id"]) for row in rows}) == len(rows)
    )
    ordered = (
        sorted(
            rows,
            key=lambda row: digest((SALT + str(row["id"])).encode()),
        )
        if population_valid
        else []
    )
    starts = [0]
    for size in SPLIT_SIZES:
        starts.append(starts[-1] + size)
    splits = [ordered[starts[i] : starts[i + 1]] for i in range(5)]
    reserve_prefix = splits[4][:4]
    prefix_ids = [str(row["id"]) for row in reserve_prefix]
    nonreserve_ids = {
        str(row["id"]) for split in splits[:4] for row in split
    }
    prefix_sha256 = digest(canonical(prefix_ids))
    privacy = {
        "task_ids_serialized": False,
        "names_serialized": False,
        "descriptions_serialized": False,
        "facts_serialized": False,
        "answers_serialized": False,
        "rationales_serialized": False,
        "task_rows_serialized": False,
        "model_responses_opened": False,
        "endpoints_opened": False,
    }
    manifest = {
        "protocol_version": VERSION,
        "bindings": {
            "source_sha256": SOURCE_SHA256,
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "mechanics_protocol_sha256": PROTOCOL_SHA256,
        },
        "selection": {
            "population_count": len(rows) if isinstance(rows, list) else 0,
            "original_split_counts": list(SPLIT_SIZES),
            "cohort_source": "reserve_positions_0_through_3",
            "cohort_count": len(reserve_prefix),
            "cohort_ordered_id_sha256": prefix_sha256,
            "remaining_reserve_count": len(splits[4][4:]),
        },
        "privacy": privacy,
    }
    gates = {
        "exact_source_binding": source_matches,
        "exact_population_and_unique_ids": population_valid,
        "exact_original_split_counts": [len(split) for split in splits]
        == list(SPLIT_SIZES),
        "exact_reserve_prefix_count": len(reserve_prefix) == 4,
        "exact_reserve_prefix_hash": prefix_sha256 == EXPECTED_PREFIX_SHA256,
        "disjoint_from_all_original_nonreserve_splits": not (
            set(prefix_ids) & nonreserve_ids
        ),
        "public_manifest_is_aggregate_only": all(
            value is False for value in privacy.values()
        ),
    }
    passed = all(gates.values())
    result = {
        "protocol_version": VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "implementation_authorized" if passed else "close_exact_v2_cohort",
        "manifest_sha256": digest(canonical(manifest)),
        "gates": gates,
        "accounting": {
            "openrouter_calls": 0,
            "openrouter_cost_usd": 0.0,
            "oatml_cluster_use": 0,
        },
        "authorizes": "implementation_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.source.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "SOURCE_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
