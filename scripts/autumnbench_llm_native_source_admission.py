#!/usr/bin/env python3
"""Audit the frozen AutumnBench public source population without task payloads."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_AUTUMN_COMMIT = "0929d175c3d296de091ac4b998d09b9b55a38f67"
EXPECTED_AUTUMN_TREE = "12e0cb5acfd25f5f6483cc90e696d10c00111c25"
EXPECTED_MARA_COMMIT = "d0d4e9251151778415cc5195c5dfb8296a5ded24"
EXPECTED_MARA_TREE = "ec7872d2f5ab047ac43e37ed5c8b8ea108d493f8"
EXPECTED_MANIFEST_SHA256 = (
    "c3f17e4d51318994dcd169a8ab3c4db1212dc30b83ad6b0dec88e933edb2071a"
)
EXPECTED_PROTOCOL_SHA256 = (
    "0980ca9f9f189577c154367475a7a94bbb120dbd6cc9bb53b8e2f1dff4292856"
)
CANONICAL_TYPES = {
    "masked_frame_prediction",
    "change_detection",
    "planning",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(repo: Path, expression: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", expression], text=True
    ).strip()


def audit(
    *,
    manifest_path: Path,
    autumn_repo: Path,
    mara_repo: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != {"id", "program", "type"}
        for row in rows
    ):
        raise RuntimeError("public manifest has an invalid metadata schema")

    ids = [row["id"] for row in rows]
    programs = [row["program"] for row in rows]
    type_counts = Counter(row["type"] for row in rows)
    by_program: dict[str, list[str]] = {}
    for row in rows:
        by_program.setdefault(row["program"], []).append(row["type"])

    bindings = {
        "autumn_commit": git_value(autumn_repo, "HEAD") == EXPECTED_AUTUMN_COMMIT,
        "autumn_tree": git_value(autumn_repo, "HEAD^{tree}") == EXPECTED_AUTUMN_TREE,
        "mara_commit": git_value(mara_repo, "HEAD") == EXPECTED_MARA_COMMIT,
        "mara_tree": git_value(mara_repo, "HEAD^{tree}") == EXPECTED_MARA_TREE,
        "manifest": sha256_file(manifest_path) == EXPECTED_MANIFEST_SHA256,
        "protocol": sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
    }
    population_pass = (
        len(rows) == 129
        and len(ids) == len(set(ids)) == 129
        and len(set(programs)) == 43
    )
    complete_triplets = population_pass and all(
        len(types) == 3 and set(types) == CANONICAL_TYPES
        for types in by_program.values()
    )
    gates: dict[str, bool | None] = {
        "immutable_bindings": all(bindings.values()),
        "exact_129_tasks_43_worlds": population_pass,
        "complete_three_type_triplets": complete_triplets,
        "path_separation": None,
        "native_experiment_interface": None,
        "sealed_derived_tests": None,
        "local_deterministic_handshake": None,
        "no_privileged_publication": True,
    }
    return {
        "schema_version": 1,
        "interface_version": "autumnbench-llm-native-source-admission-v1",
        "status": "source_failed_closed",
        "decision": "close_exact_autumnbench_public_release_route",
        "authorizes": "nothing",
        "failure_gate": "exact_129_tasks_43_worlds",
        "ordering": "stopped_before_task_payload_or_fixture_execution",
        "repository_bindings": {
            "autumn_commit": EXPECTED_AUTUMN_COMMIT,
            "autumn_tree": EXPECTED_AUTUMN_TREE,
            "mara_commit": EXPECTED_MARA_COMMIT,
            "mara_tree": EXPECTED_MARA_TREE,
        },
        "manifest": {
            "sha256": sha256_file(manifest_path),
            "declared_total_count": manifest.get("total_count"),
            "observed_task_count": len(rows),
            "unique_task_count": len(set(ids)),
            "unique_base_world_count": len(set(programs)),
            "type_counts": dict(sorted(type_counts.items())),
            "all_observed_worlds_form_triplets": all(
                len(types) == 3 and set(types) == CANONICAL_TYPES
                for types in by_program.values()
            ),
        },
        "payloads": {
            "nonfixture_programs_downloaded": 0,
            "nonfixture_prompts_downloaded": 0,
            "nonfixture_answers_downloaded": 0,
            "fixture_executions": 0,
            "model_calls": 0,
            "endpoint_outcomes_opened": 0,
        },
        "bindings": bindings,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--autumn-repo", type=Path, required=True)
    parser.add_argument("--mara-repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        manifest_path=args.manifest,
        autumn_repo=args.autumn_repo,
        mara_repo=args.mara_repo,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
