#!/usr/bin/env python3
"""Diagnose a closed SWE-Interact serving artifact without rescuing its gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.swe_interact_mechanics_serving import (
    JUDGE_KEYS,
    TASK_PROBES,
    load_tasks,
)


EXPECTED_RAW_SHA256 = (
    "2a59fe5cda0517c23667f8f35e64210d9fbc32a52750b1b642ce2cb50b754a77"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_diagnostic_ids(
    value: str,
    valid_ids: tuple[str, ...],
) -> frozenset[str]:
    if value == "NONE":
        return frozenset()
    valid = set(valid_ids)
    parsed = []
    for part in value.split(","):
        match = re.fullmatch(r"R?([1-9][0-9]*)", part)
        if match is None:
            raise ValueError("Diagnostic annotation has malformed ID")
        requirement_id = f"R{int(match.group(1))}"
        if requirement_id not in valid:
            raise ValueError("Diagnostic annotation has unknown ID")
        parsed.append(requirement_id)
    if len(parsed) != len(set(parsed)):
        raise ValueError("Diagnostic annotation repeats an ID")
    return frozenset(parsed)


def parse_diagnostic_judgement(
    text: str,
    valid_ids: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    lines = text.strip().splitlines()
    if len(lines) != len(JUDGE_KEYS):
        raise ValueError("Diagnostic judgement has wrong line count")
    rows: dict[str, frozenset[str]] = {}
    for line in lines:
        if line.count("|") != 1:
            raise ValueError("Diagnostic judgement row is malformed")
        key, value = line.split("|")
        if key not in JUDGE_KEYS or key in rows:
            raise ValueError("Diagnostic judgement key is invalid")
        rows[key] = parse_diagnostic_ids(value, valid_ids)
    if set(rows) != set(JUDGE_KEYS):
        raise ValueError("Diagnostic judgement is missing a key")
    return rows


def build_diagnostic(
    raw_path: Path,
    source_repo: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw_sha256 = sha256_file(raw_path)
    if raw_sha256 != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"Closed raw artifact hash changed: {raw_sha256}"
        )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tasks = load_tasks(source_repo, manifest_path)
    task_by_id = {task.task_id: task for task in tasks}
    if set(raw["judgements"]) != set(task_by_id):
        raise ValueError("Closed raw task IDs changed")

    rows = []
    for task_id, task in task_by_id.items():
        labels = parse_diagnostic_judgement(
            raw["judgements"][task_id]["response"],
            task.valid_requirement_ids,
        )
        probe = TASK_PROBES[task_id]
        root_a_expected = set(probe["root_a_expected"])
        root_b_expected = set(probe["root_b_expected"])
        review_a_expected = set(probe["review_a_expected"])
        review_b_expected = set(probe["review_b_expected"])
        rows.append(
            {
                "task_id": task_id,
                "family": task.family,
                "requirement_count": len(task.requirements),
                "initial_requirement_count": len(labels["INITIAL"]),
                "generic_new_requirement_count": len(labels["GENERIC"]),
                "generic_requirement_fraction": (
                    len(labels["GENERIC"]) / len(task.requirements)
                ),
                "root_a_requirement_ids": sorted(labels["ROOT_A_1"]),
                "root_b_requirement_ids": sorted(labels["ROOT_B_1"]),
                "review_a_requirement_ids": sorted(labels["REVIEW_A"]),
                "review_b_requirement_ids": sorted(labels["REVIEW_B"]),
                "root_repeat_pairs_exact": int(
                    labels["ROOT_A_1"] == labels["ROOT_A_2"]
                )
                + int(labels["ROOT_B_1"] == labels["ROOT_B_2"]),
                "distinct_aligned_roots": (
                    bool(labels["ROOT_A_1"] & root_a_expected)
                    and bool(labels["ROOT_B_1"] & root_b_expected)
                    and labels["ROOT_A_1"] != labels["ROOT_B_1"]
                ),
                "distinct_aligned_reviews": (
                    bool(labels["REVIEW_A"] & review_a_expected)
                    and bool(labels["REVIEW_B"] & review_b_expected)
                    and labels["REVIEW_A"] != labels["REVIEW_B"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "status": "diagnostic_only_not_gate_rescue",
        "formal_v1_status": "failed_closed_on_noncanonical_requirement_ids",
        "formal_decision": "close_exact_swe_interact_route",
        "raw_sha256": raw_sha256,
        "model_requests": 0,
        "openrouter_cost_usd": 0.0,
        "diagnostic_normalization": (
            "accept an optional missing R prefix only for the labeled "
            "zero-call diagnostic"
        ),
        "summary": {
            "num_tasks": len(rows),
            "initial_at_most_one_count": sum(
                row["initial_requirement_count"] <= 1 for row in rows
            ),
            "generic_zero_progress_count": sum(
                row["generic_new_requirement_count"] == 0 for row in rows
            ),
            "generic_requirement_counts": [
                row["generic_new_requirement_count"] for row in rows
            ],
            "generic_requirement_fractions": [
                row["generic_requirement_fraction"] for row in rows
            ],
            "exact_root_repeat_pair_count": sum(
                row["root_repeat_pairs_exact"] for row in rows
            ),
            "distinct_aligned_root_task_count": sum(
                row["distinct_aligned_roots"] for row in rows
            ),
            "distinct_aligned_review_task_count": sum(
                row["distinct_aligned_reviews"] for row in rows
            ),
            "scientific_generic_gate_would_pass": all(
                row["generic_new_requirement_count"] == 0 for row in rows
            ),
        },
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_diagnostic(
        args.raw.resolve(),
        args.source_repo.resolve(),
        args.manifest.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
