#!/usr/bin/env python3
"""Freeze a path-only, family-stratified split of the SWE-Interact release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


EXPECTED_COMMIT = "b32f98c3b8f76ca65e84341d1f30e5af7135f85d"
SOURCE_URL = "https://github.com/scaleapi/SWE-Interact"
SEED = 24423
FAMILY_PREFIXES = {
    "deepswe": "deepswe_",
    "refactoring": "rf_task-",
    "swebench_pro": "swebenchpro_",
}
PARTITION_COUNTS = {
    "mechanics": 3,
    "development": 7,
    "confirmation": 8,
    "retained": 7,
}
REQUIRED_RELATIVE_PATHS = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/repo_exec_server.py",
    "environment/user-server/Dockerfile",
    "environment/user-server/persona.md",
    "environment/user-server/server.py",
    "solution/solve.sh",
    "tests/test.sh",
    "steps/01_plan/instruction.md",
    "steps/01_plan/tests/test.sh",
    "steps/02_implement/instruction.md",
    "steps/02_implement/tests/test.sh",
    "steps/03_handoff/instruction.md",
    "steps/03_handoff/tests/test.sh",
    "steps/04_write_tests/instruction.md",
    "steps/04_write_tests/tests/test.sh",
    "steps/05_test_handoff/instruction.md",
    "steps/05_test_handoff/tests/test.sh",
)


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _family(task_id: str) -> str:
    matches = [
        family
        for family, prefix in FAMILY_PREFIXES.items()
        if task_id.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Unrecognized or ambiguous task family: {task_id}")
    return matches[0]


def parse_task_paths(paths: Iterable[str]) -> dict[str, list[str]]:
    path_set = set(paths)
    task_ids = sorted(
        path.removeprefix("data/multiturn/").removesuffix("/task.toml")
        for path in path_set
        if path.startswith("data/multiturn/") and path.endswith("/task.toml")
        and path.count("/") == 3
    )
    if len(task_ids) != 75 or len(set(task_ids)) != 75:
        raise ValueError(f"Expected 75 unique multiturn tasks, found {len(task_ids)}")

    by_family = {family: [] for family in FAMILY_PREFIXES}
    for task_id in task_ids:
        by_family[_family(task_id)].append(task_id)
        prefix = f"data/multiturn/{task_id}/"
        missing = [
            relative
            for relative in REQUIRED_RELATIVE_PATHS
            if prefix + relative not in path_set
        ]
        if missing:
            raise ValueError(
                f"Task {task_id} is missing required release paths: {missing}"
            )
        paired_singleturn = f"data/singleturn/{task_id}/task.toml"
        if paired_singleturn not in path_set:
            raise ValueError(
                f"Task {task_id} is missing paired single-turn task.toml"
            )

    bad_counts = {
        family: len(rows)
        for family, rows in by_family.items()
        if len(rows) != 25
    }
    if bad_counts:
        raise ValueError(f"Expected 25 tasks per family, found {bad_counts}")
    return by_family


def partition_tasks(
    by_family: dict[str, list[str]],
    *,
    seed: int = SEED,
) -> dict[str, list[dict[str, str]]]:
    partitions = {name: [] for name in PARTITION_COUNTS}
    for family in FAMILY_PREFIXES:
        ordered = sorted(
            by_family[family],
            key=lambda task_id: (
                _sha256_text(f"{seed}:{family}:{task_id}"),
                task_id,
            ),
        )
        cursor = 0
        for partition, count in PARTITION_COUNTS.items():
            selected = ordered[cursor : cursor + count]
            cursor += count
            for task_id in selected:
                partitions[partition].append(
                    {
                        "family": family,
                        "task_id": task_id,
                        "task_path": f"data/multiturn/{task_id}",
                    }
                )
        if cursor != len(ordered):
            raise AssertionError(f"Partition counts do not exhaust {family}")

    for rows in partitions.values():
        rows.sort(key=lambda row: (row["family"], row["task_id"]))
    return partitions


def build_manifest(repo: Path) -> dict[str, object]:
    commit = _git_output(repo, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Expected commit {EXPECTED_COMMIT}, found {commit}")
    if _git_output(repo, "status", "--short"):
        raise ValueError("Pinned source checkout is not clean")

    tracked_paths = [
        path for path in _git_output(repo, "ls-files").splitlines() if path
    ]
    by_family = parse_task_paths(tracked_paths)
    partitions = partition_tasks(by_family)
    flattened = [
        row["task_id"] for rows in partitions.values() for row in rows
    ]
    if len(flattened) != 75 or len(set(flattened)) != 75:
        raise AssertionError("Partitions are not a disjoint cover")

    tree_listing = _git_output(repo, "ls-tree", "-r", "--full-tree", "HEAD")
    return {
        "schema_version": 1,
        "source": SOURCE_URL,
        "commit": commit,
        "git_tree": _git_output(repo, "rev-parse", "HEAD^{tree}"),
        "tracked_file_count": len(tracked_paths),
        "tracked_tree_manifest_sha256": _sha256_text(tree_listing + "\n"),
        "seed": SEED,
        "partition_rule": {
            "ordering": "SHA256(seed:family:task_id), then task_id",
            "stratified_by": list(FAMILY_PREFIXES),
            "counts_per_family": PARTITION_COUNTS,
        },
        "family_counts": {
            family: len(rows) for family, rows in by_family.items()
        },
        "counts": {
            name: len(rows) for name, rows in partitions.items()
        },
        "partitions": partitions,
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_json_atomic(
        args.output.resolve(),
        build_manifest(args.repo.resolve()),
    )


if __name__ == "__main__":
    main()
