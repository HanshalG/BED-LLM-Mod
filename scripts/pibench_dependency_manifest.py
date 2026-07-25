#!/usr/bin/env python3
"""Freeze pi-Bench dependency-task splits without exposing intent text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable, Sequence

import yaml


SOURCE_REPOSITORY = "https://github.com/Simplified-Reasoning/Pi-Bench"
SOURCE_COMMIT = "383910b1698758a198b86037c63a111c8edc32ad"
SELECTION_SEED = 24_403
USER_IDS = (
    "Financier",
    "law_trainee",
    "marketer",
    "pharmacist",
    "researcher",
)
SPLIT_HASHES = {
    "mechanics": (
        "ad526a0f6505eef7c3aee1580bc2261226967ea6a419d4ff1179cde70b160156"
    ),
    "opportunity": (
        "45fe47a534aef1a4754390450b2e76327cd5441c3ed07171f25e5edb829a43d5"
    ),
    "development": (
        "469755a9fb606d3d94b5d724d2d074ab7699ed88d43c94d878614034d5135947"
    ),
    "holdout": (
        "2eeb48e535b48e7a18b140ae4322c2c0cea2b251b7a0e421b5da311fd4588851"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bundle_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def split_dependency_tasks(
    dependency_ids_by_user: dict[str, list[str]],
) -> dict[str, list[str]]:
    if tuple(dependency_ids_by_user) != USER_IDS:
        raise ValueError("pi-Bench user order changed")
    rng = random.Random(SELECTION_SEED)
    splits = {
        "mechanics": [],
        "opportunity": [],
        "development": [],
        "holdout": [],
    }
    for user_id in USER_IDS:
        dependency_ids = dependency_ids_by_user[user_id]
        if len(dependency_ids) != 6 or len(set(dependency_ids)) != 6:
            raise ValueError(
                f"expected six dependency-final tasks for {user_id}"
            )
        mechanics_id = dependency_ids[0]
        eligible = sorted(dependency_ids[1:])
        rng.shuffle(eligible)
        splits["mechanics"].append(mechanics_id)
        splits["opportunity"].extend(eligible[:2])
        splits["development"].append(eligible[2])
        splits["holdout"].extend(eligible[3:])
    expected_sizes = {
        "mechanics": 5,
        "opportunity": 10,
        "development": 5,
        "holdout": 10,
    }
    if {key: len(value) for key, value in splits.items()} != expected_sizes:
        raise ValueError("pi-Bench split sizes changed")
    hashes = {key: ordered_hash(value) for key, value in splits.items()}
    if hashes != SPLIT_HASHES:
        raise ValueError("pi-Bench split hashes changed")
    return splits


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def build_manifest(root: Path, *, verify_revision: bool = True) -> dict[str, Any]:
    root = root.resolve()
    if verify_revision:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != SOURCE_COMMIT:
            raise ValueError(
                f"pi-Bench revision is {revision}, expected {SOURCE_COMMIT}"
            )
    else:
        revision = SOURCE_COMMIT

    data_root = root / "data"
    dependencies_by_user: dict[str, list[str]] = {}
    dependency_map: dict[str, list[str]] = {}
    task_paths: dict[str, Path] = {}
    all_task_ids: list[str] = []
    episode_paths: list[Path] = []
    for user_id in USER_IDS:
        user_root = data_root / user_id
        episode_path = user_root / "episode.yaml"
        episode_paths.append(episode_path)
        episode = _load_yaml(episode_path)
        tasks = episode.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 20:
            raise ValueError(f"expected 20 episode tasks for {user_id}")
        dependency_ids: list[str] = []
        for raw_task in tasks:
            if not isinstance(raw_task, dict):
                raise ValueError(f"invalid episode task for {user_id}")
            task_id = str(raw_task.get("task_id", ""))
            depends_on = [
                str(value) for value in raw_task.get("depends_on", []) or []
            ]
            if not task_id:
                raise ValueError(f"missing task ID for {user_id}")
            all_task_ids.append(task_id)
            task_path = user_root / "tasks" / task_id / "task.yaml"
            if not task_path.is_file():
                raise ValueError(f"missing task YAML for {task_id}")
            task_paths[task_id] = task_path
            if depends_on:
                dependency_ids.append(task_id)
                dependency_map[task_id] = depends_on
        dependencies_by_user[user_id] = dependency_ids

    if len(all_task_ids) != 100 or len(set(all_task_ids)) != 100:
        raise ValueError("pi-Bench task universe changed")
    splits = split_dependency_tasks(dependencies_by_user)
    selected_ids = {
        task_id for values in splits.values() for task_id in values
    }
    if len(selected_ids) != 30:
        raise ValueError("pi-Bench dependency-task universe changed")

    metadata: dict[str, dict[str, Any]] = {}
    task_yaml_paths = []
    for task_id in sorted(selected_ids):
        task_path = task_paths[task_id]
        task_yaml_paths.append(task_path)
        task = _load_yaml(task_path)
        intent = task.get("intent")
        if not isinstance(intent, dict):
            raise ValueError(f"missing intent mapping for {task_id}")
        hidden_intents = intent.get("hidden_intent")
        if not isinstance(hidden_intents, list) or not hidden_intents:
            raise ValueError(f"missing hidden intent list for {task_id}")
        initial_input = str(intent.get("initial_input", ""))
        task_dir = task_path.parent
        asset_paths = sorted(
            path
            for path in task_dir.rglob("*")
            if path.is_file() and path != task_path
        )
        metadata[task_id] = {
            "user_id": str(task.get("user_id", "")),
            "depends_on": dependency_map[task_id],
            "dependency_count": len(dependency_map[task_id]),
            "hidden_intent_count": len(hidden_intents),
            "initial_input_sha256": hashlib.sha256(
                initial_input.encode("utf-8")
            ).hexdigest(),
            "task_yaml_sha256": sha256_file(task_path),
            "asset_count": len(asset_paths),
            "asset_name_sha256": ordered_hash(
                [
                    path.relative_to(task_dir).as_posix()
                    for path in asset_paths
                ]
            ),
            "task_bundle_sha256": bundle_hash(
                [task_path, *asset_paths],
                task_dir,
            ),
        }

    source_paths = [*episode_paths, *task_yaml_paths]
    return {
        "interface_version": "pibench-dependency-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": revision,
            "episode_and_dependency_task_yaml_sha256": bundle_hash(
                source_paths,
                root,
            ),
        },
        "selection_seed": SELECTION_SEED,
        "task_universe_count": len(all_task_ids),
        "dependency_task_count": len(selected_ids),
        "splits": {
            split: {
                "task_ids": task_ids,
                "ordered_sha256": ordered_hash(task_ids),
                "tasks": {
                    task_id: metadata[task_id] for task_id in task_ids
                },
            }
            for split, task_ids in splits.items()
        },
        "value_access": {
            "hidden_intent_text_emitted": False,
            "initial_input_text_emitted": False,
            "objective_values_emitted": False,
            "asset_contents_emitted": False,
            "metadata_only_fields": [
                "task_id",
                "user_id",
                "depends_on",
                "dependency_count",
                "hidden_intent_count",
                "asset_count",
                "content hashes",
            ],
        },
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.source_root)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
