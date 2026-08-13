#!/usr/bin/env python3
"""Value-blind source admission for AgentClinic NEJM dynamic-support BED."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "agentclinic-dynamic-support-source-v1"
SOURCE_COMMIT = "b6570edefb940857a7c334350656b29f9d984f24"
SOURCE_TREE = "084556771761828ac0fd121d37353207de91486c"
SOURCE_FILE = "agentclinic_nejm_extended.jsonl"
SOURCE_SHA256 = "d945305ee17ee1456053fbfe2e9d9c5e8b27d14538bf48ab8ace7306dc437b85"
SPLIT_SALT = "agentclinic-dynamic-support-v1|"
EXPECTED_FIELDS = {
    "answers",
    "image_url",
    "patient_info",
    "physical_exams",
    "question",
    "type",
}
DEFAULT_SPLITS = (
    ("mechanics", 6),
    ("opportunity", 24),
    ("development", 30),
    ("confirmation", 40),
    ("reserve", 20),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def git_value(root: Path, expression: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", expression],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: row must be an object")
        rows.append(value)
    return rows


def split_case_ids(
    case_ids: list[str],
    split_counts: tuple[tuple[str, int], ...] = DEFAULT_SPLITS,
) -> dict[str, list[str]]:
    if sum(count for _, count in split_counts) != len(case_ids):
        raise ValueError("split counts do not cover the population")
    ordered = sorted(case_ids, key=lambda value: sha256_bytes((SPLIT_SALT + value).encode()))
    splits: dict[str, list[str]] = {}
    offset = 0
    for name, count in split_counts:
        splits[name] = ordered[offset : offset + count]
        offset += count
    return splits


def audit_rows(
    rows: list[dict[str, Any]],
    split_counts: tuple[tuple[str, int], ...] = DEFAULT_SPLITS,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    exact_schema = all(set(row) == EXPECTED_FIELDS for row in rows)
    case_ids = [sha256_bytes(canonical_bytes(row)) for row in rows]
    canonical_unique = len(set(case_ids)) == len(rows)
    exactly_one_correct = all(
        isinstance(row.get("answers"), list)
        and sum(answer.get("correct") is True for answer in row["answers"] if isinstance(answer, dict)) == 1
        for row in rows
    )
    required_nonempty = all(
        all(row.get(field) for field in ("question", "image_url", "patient_info", "physical_exams", "type"))
        for row in rows
    )
    https_images = all(isinstance(row.get("image_url"), str) and row["image_url"].startswith("https://") for row in rows)
    splits = split_case_ids(case_ids, split_counts)
    flattened = [case_id for values in splits.values() for case_id in values]
    partition_complete_disjoint = len(flattened) == len(rows) and len(set(flattened)) == len(rows)
    expected_count = sum(count for _, count in split_counts)
    gates = {
        "exact_population_size": len(rows) == expected_count,
        "exact_six_field_schema": exact_schema,
        "canonical_rows_unique": canonical_unique,
        "exactly_one_correct_answer_per_case": exactly_one_correct,
        "required_fields_nonempty": required_nonempty,
        "all_image_urls_https": https_images,
        "partition_complete_and_disjoint": partition_complete_disjoint,
    }
    summary = {
        "population_count": len(rows),
        "canonical_unique_count": len(set(case_ids)),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "ordered_split_case_id_sha256": {
            name: sha256_bytes(canonical_bytes(values)) for name, values in splits.items()
        },
        "gates": gates,
    }
    return summary, splits


def run_audit(source_root: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data_path = source_root / SOURCE_FILE
    data_bytes = data_path.read_bytes()
    source_binding = {
        "repository": "https://github.com/SamuelSchmidgall/AgentClinic",
        "commit": git_value(source_root, "HEAD"),
        "tree": git_value(source_root, "HEAD^{tree}"),
        "file": SOURCE_FILE,
        "file_sha256": sha256_bytes(data_bytes),
    }
    source_binding_pass = source_binding == {
        "repository": "https://github.com/SamuelSchmidgall/AgentClinic",
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "file": SOURCE_FILE,
        "file_sha256": SOURCE_SHA256,
    }
    summary, _ = audit_rows(load_rows(data_path))
    gates = {"immutable_source_binding": source_binding_pass, **summary["gates"]}
    passed = all(gates.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source": source_binding,
        "population_count": summary["population_count"],
        "canonical_unique_count": summary["canonical_unique_count"],
        "split_counts": summary["split_counts"],
        "ordered_split_case_id_sha256": summary["ordered_split_case_id_sha256"],
        "privacy": {
            "individual_case_ids_serialized": False,
            "source_values_serialized": False,
            "diagnoses_serialized": False,
            "endpoint_values_opened": False,
        },
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_authorized" if passed else "close_agentclinic_source_route",
        "gates": gates,
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, result = run_audit(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
