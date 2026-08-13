#!/usr/bin/env python3
"""Admit the final HiddenBench reserve cohort without semantic values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "hiddenbench-dynamic-belief-v3-source-v1"
SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
SOURCE_PROTOCOL_SHA256 = "5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28"
SOURCE_MANIFEST_SHA256 = "b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d"
SOURCE_AUDIT_SHA256 = "67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9"
V2_TERMINAL_SHA256 = "7ab19c902d432e9721e9f3cde2ec20f39064979d7c4f5607449157dd92d3a224"
PROTOCOL_SHA256 = "cab055aed50b22abfc1b90e720523882d8d6320616f0ab5f474a9340bf428bba"
SALT = "hiddenbench-adaptive-elicitation-v1|"
EXPECTED_COHORT_SHA256 = "822314c2c662a2711b6d2253601b9a801071c4d53bb53b92bb115240d98bee40"
V2_COHORT_SHA256 = "adcbabc78acc15d7b88d2b0552636525ea64d25840113033a469c558bae3807a"


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
        and len(rows) == 65
        and all(isinstance(row, dict) and "id" in row for row in rows)
        and len({str(row["id"]) for row in rows}) == 65
    )
    ordered = sorted(
        rows,
        key=lambda row: digest((SALT + str(row["id"])).encode()),
    ) if population_valid else []
    v2 = ordered[56:60]
    v3 = ordered[60:64]
    unused = ordered[64:65]
    v2_ids = [str(row["id"]) for row in v2]
    v3_ids = [str(row["id"]) for row in v3]
    v3_hash = digest(canonical(v3_ids))
    v2_hash = digest(canonical(v2_ids))
    nonreserve_ids = {str(row["id"]) for row in ordered[:56]}
    privacy = {
        "task_ids_serialized": False,
        "semantic_values_serialized": False,
        "task_rows_serialized": False,
        "model_responses_opened": False,
        "registered_answers_opened": False,
        "endpoints_opened": False,
    }
    manifest = {
        "protocol_version": VERSION,
        "bindings": {
            "source_sha256": SOURCE_SHA256,
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "v2_terminal_sha256": V2_TERMINAL_SHA256,
            "v3_protocol_sha256": PROTOCOL_SHA256,
        },
        "selection": {
            "cohort_source": "original_reserve_positions_4_through_7",
            "cohort_count": len(v3),
            "cohort_ordered_id_sha256": v3_hash,
            "excluded_v2_count": len(v2),
            "excluded_v2_ordered_id_sha256": v2_hash,
            "unused_reserve_count": len(unused),
        },
        "privacy": privacy,
    }
    gates = {
        "exact_source_binding": source_matches,
        "exact_population_and_unique_ids": population_valid,
        "exact_v2_exclusion_hash": v2_hash == V2_COHORT_SHA256,
        "exact_v3_cohort_count": len(v3) == 4,
        "exact_v3_cohort_hash": v3_hash == EXPECTED_COHORT_SHA256,
        "v3_disjoint_from_v2_and_nonreserve": not (
            set(v3_ids) & (set(v2_ids) | nonreserve_ids)
        ),
        "one_reserve_row_left_unused": len(unused) == 1,
        "public_manifest_is_aggregate_only": all(
            value is False for value in privacy.values()
        ),
    }
    passed = all(gates.values())
    result = {
        "protocol_version": VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "implementation_authorized" if passed else "close_exact_v3_cohort",
        "manifest_sha256": digest(canonical(manifest)),
        "gates": gates,
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0},
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
    for name, value in (("MANIFEST.json", manifest), ("SOURCE_AUDIT.json", result)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
