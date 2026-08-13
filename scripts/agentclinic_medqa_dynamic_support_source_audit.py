#!/usr/bin/env python3
"""Fail-closed value-blind source audit for AgentClinic extended MedQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "agentclinic-medqa-dynamic-support-source-v1"
SOURCE_COMMIT = "b6570edefb940857a7c334350656b29f9d984f24"
SOURCE_TREE = "084556771761828ac0fd121d37353207de91486c"
CODE_FILE = "agentclinic.py"
CODE_SHA256 = "ee9cfb3020c7addf717ba8eb3510ba81b2b4b53071cba67d0663ad6c7c6ddbd3"
DATA_FILE = "agentclinic_medqa_extended.jsonl"
DATA_SHA256 = "54a024eb2705c6c55d1988766adf4ab02ea7bbe2a28f843107b740032200f232"
EXPECTED_COUNT = 213
EXPECTED_TOP_FIELDS = {"OSCE_Examination"}
EXPECTED_OSCE_FIELDS = {
    "Correct_Diagnosis",
    "Objective_for_Doctor",
    "Patient_Actor",
    "Physical_Examination_Findings",
    "Test_Results",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_value(root: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", expression],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: row is not an object")
        rows.append(value)
    return rows


def audit(source_root: Path, protocol_path: Path) -> dict[str, Any]:
    data_path = source_root / DATA_FILE
    code_path = source_root / CODE_FILE
    rows = load_rows(data_path)
    top_schemas = {tuple(sorted(row)) for row in rows}
    nested_schemas = {
        tuple(sorted(row.get("OSCE_Examination", {})))
        for row in rows
        if isinstance(row.get("OSCE_Examination"), dict)
    }
    exact_top_rows = sum(set(row) == EXPECTED_TOP_FIELDS for row in rows)
    exact_nested_rows = sum(
        isinstance(row.get("OSCE_Examination"), dict)
        and set(row["OSCE_Examination"]) == EXPECTED_OSCE_FIELDS
        for row in rows
    )
    source_binding = {
        "commit": git_value(source_root, "HEAD"),
        "tree": git_value(source_root, "HEAD^{tree}"),
        "code_sha256": sha256_file(code_path),
        "data_sha256": sha256_file(data_path),
    }
    binding_pass = source_binding == {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "code_sha256": CODE_SHA256,
        "data_sha256": DATA_SHA256,
    }
    gates = {
        "immutable_source_binding": binding_pass,
        "exact_physical_population_size": len(rows) == EXPECTED_COUNT,
        "single_exact_top_level_schema": len(top_schemas) == 1
        and exact_top_rows == len(rows),
        "single_exact_five_field_osce_schema": len(nested_schemas) == 1
        and exact_nested_rows == len(rows),
    }
    passed = all(gates.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_authorized" if passed else "close_exact_agentclinic_medqa_construction",
        "protocol_sha256": sha256_file(protocol_path),
        "source_binding": source_binding,
        "population_shape": {
            "expected_nonblank_rows": EXPECTED_COUNT,
            "observed_nonblank_rows": len(rows),
            "distinct_top_level_schema_count": len(top_schemas),
            "rows_with_exact_top_level_schema": exact_top_rows,
            "distinct_osce_schema_count": len(nested_schemas),
            "rows_with_exact_five_field_osce_schema": exact_nested_rows,
        },
        "gates": gates,
        "later_source_gates_opened": False,
        "privacy": {
            "individual_case_ids_serialized": False,
            "source_values_serialized": False,
            "field_values_serialized": False,
            "diagnoses_serialized": False,
            "mechanics_cases_opened": False,
            "endpoint_values_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
        "authorizes": "mechanics_only" if passed else "nothing",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_root.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
