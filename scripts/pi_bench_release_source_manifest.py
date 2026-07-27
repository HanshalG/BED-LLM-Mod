#!/usr/bin/env python3
"""Freeze a path-only chronological partition for the pinned Pi-Bench release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


EXPECTED_COMMIT = "383910b1698758a198b86037c63a111c8edc32ad"
PERSONAS = {
    "financier": ("Financier", "Financier"),
    "law_trainee": ("law_trainee", "law_trainee"),
    "marketer": ("marketer", "marketer"),
    "pharmacist": ("pharmacist", "pharmacist"),
    "researcher": ("researcher", "researcher"),
}
PARTITION_RANGES = {
    "mechanics": range(1, 5),
    "development": range(5, 11),
    "confirmation": range(11, 17),
    "retained": range(17, 21),
}


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


def parse_task_paths(paths: Iterable[str]) -> dict[str, dict[int, dict[str, str]]]:
    tasks: dict[str, dict[int, dict[str, str]]] = {
        persona: {} for persona in PERSONAS
    }
    for persona, (directory, task_prefix) in PERSONAS.items():
        pattern = re.compile(
            rf"^data/{re.escape(directory)}/tasks/"
            rf"({re.escape(task_prefix)}_task_(\d{{3}}))/task\.yaml$"
        )
        for path in paths:
            match = pattern.match(path)
            if not match:
                continue
            task_id = match.group(1)
            position = int(match.group(2))
            if position in tasks[persona]:
                raise ValueError(f"Duplicate task position {persona}:{position}")
            tasks[persona][position] = {"task_id": task_id, "path": path}

    expected_positions = set(range(1, 21))
    for persona, rows in tasks.items():
        if set(rows) != expected_positions:
            raise ValueError(
                f"Expected task positions 1-20 for {persona}, found {sorted(rows)}"
            )
    return tasks


def partition_tasks(
    tasks: dict[str, dict[int, dict[str, str]]],
) -> dict[str, list[dict[str, object]]]:
    partitions: dict[str, list[dict[str, object]]] = {
        name: [] for name in PARTITION_RANGES
    }
    for partition_name, positions in PARTITION_RANGES.items():
        for persona in PERSONAS:
            for position in positions:
                row = tasks[persona][position]
                partitions[partition_name].append(
                    {
                        "persona": persona,
                        "session_position": position,
                        "task_id": row["task_id"],
                        "task_path": row["path"],
                    }
                )
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
    tasks = parse_task_paths(tracked_paths)
    partitions = partition_tasks(tasks)
    tree_listing = _git_output(
        repo, "ls-tree", "-r", "--full-tree", "HEAD"
    )

    return {
        "schema_version": 1,
        "source": "https://github.com/Simplified-Reasoning/Pi-Bench",
        "commit": commit,
        "git_tree": _git_output(repo, "rev-parse", "HEAD^{tree}"),
        "tracked_file_count": len(tracked_paths),
        "tracked_tree_manifest_sha256": _sha256_text(tree_listing + "\n"),
        "partition_rule": {
            "ordering": "release session position within each persistent persona",
            "ranges": {
                name: [min(positions), max(positions)]
                for name, positions in PARTITION_RANGES.items()
            },
        },
        "counts": {
            name: len(rows) for name, rows in partitions.items()
        },
        "partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
