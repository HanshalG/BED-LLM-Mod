#!/usr/bin/env python3
"""Freeze target-blind DiscoverLLM priority-world experiment splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence


SOURCE_REPOSITORY = "https://github.com/tsook/discoverllm"
SOURCE_COMMIT = "a9eb2846f60e3681ac8d325fc57fd4e58e2bdc97"
DATASET_REPOSITORY = (
    "https://huggingface.co/datasets/"
    "kixlab/DiscoverLLM-multiturn-preferences"
)
DATASET_REVISION = "c857bbf6265bdd573938eb7eac79a7a3131fa7ca"
CREATIVE_WRITING_SHA256 = (
    "e8f76a47447b442e59e0d76718228b94bba1a11d8f7f653cdf1498bb0adf7843"
)
SELECTION_SEED = 24_411
INSPECTED_ARTIFACT_IDS = {
    "artifact_1",
    "artifact_11",
    "artifact_151",
    "artifact_159",
}
NUM_WORLDS = 4
MIN_WORLD_NODES = 3
MIN_WORLD_DEPTH = 2
SPLIT_SIZES = {
    "mechanics": 3,
    "opportunity": 30,
    "development": 20,
}

# Filled after the first content-sealed run and then enforced on every replay.
EXPECTED_SOURCE_ARTIFACTS = 360
EXPECTED_ELIGIBLE_ARTIFACTS = 98
EXPECTED_SPLIT_HASHES = {
    "mechanics": (
        "e9ac1e814b19181db8707091053a3c1b318fd13ff7b40cd0728e825a3b64eb7d"
    ),
    "opportunity": (
        "c71fb00e887ec8df3de56783dbb2e8128834346b28e1f0b328d8b0130949664a"
    ),
    "development": (
        "5ea4f334c37ca33b664b31fb3cb751a24f3ba7792ae824309250371bb69522eb"
    ),
    "holdout": (
        "c7424dc308efda7fbde162fcdff401a97a29bc7e8236de3ab6cb84ffc6a781c5"
    ),
}
EXPECTED_WORLD_HASHES = {
    "mechanics": (
        "aaf94f7c9f48dd962c43080053c95f340f10327f5364905f434c7e77bf3d75ab"
    ),
    "opportunity": (
        "872e7eeed24b1776d86397033d417c948ed9375e28e0043c53e386ee3e3d30f7"
    ),
    "development": (
        "2d5a2e0385fe8cf5c8776c46f93ac61b7af64a01f0663921d2fa308ba6f29b78"
    ),
    "holdout": (
        "a7632a73a555cf1def1936e5468ad5ae421cb6da29e5e352ae91f2a30614b566"
    ),
}
EXPECTED_MECHANICS_IDS = ("artifact_257", "artifact_38", "artifact_448")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def subtree_shape(node: dict[str, Any]) -> tuple[int, int]:
    children = node.get("children", []) or []
    child_shapes = [subtree_shape(child) for child in children]
    return (
        1 + sum(count for count, _ in child_shapes),
        1 + max((depth for _, depth in child_shapes), default=0),
    )


def eligible_world_roots(
    criteria_state: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roots = [
        root
        for criterion in criteria_state
        for root in criterion.get("hierarchy", [])
    ]
    eligible = []
    for root in roots:
        node_count, depth = subtree_shape(root)
        if (
            float(root.get("aware", 0.0)) < 1.0
            and node_count >= MIN_WORLD_NODES
            and depth >= MIN_WORLD_DEPTH
        ):
            eligible.append(root)
    return eligible


def _load_turn_one_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "discoverllm_priority_world_manifest requires duckdb"
        ) from exc

    connection = duckdb.connect()
    escaped_path = str(path.resolve()).replace("'", "''")
    selected = connection.execute(
        "SELECT artifact_id, assistant_index, prompt, completion, "
        "criteria_history "
        f"FROM read_parquet('{escaped_path}') "
        "WHERE CAST(turn_id AS INTEGER) = 1 "
        "ORDER BY artifact_id, assistant_index"
    ).fetchall()
    return [
        {
            "artifact_id": str(artifact_id),
            "assistant_index": int(assistant_index),
            "prompt": prompt,
            "completion": completion,
            "criteria_history": json.loads(criteria_history),
        }
        for (
            artifact_id,
            assistant_index,
            prompt,
            completion,
            criteria_history,
        ) in selected
    ]


def _world_ids_for_artifact(
    artifact_id: str,
    roots: Sequence[dict[str, Any]],
) -> list[str]:
    root_ids = [str(root["id"]) for root in roots]
    seed_bytes = hashlib.sha256(
        f"{SELECTION_SEED}:{artifact_id}".encode("utf-8")
    ).digest()[:8]
    random.Random(int.from_bytes(seed_bytes, "big")).shuffle(root_ids)
    return root_ids[:NUM_WORLDS]


def eligible_artifacts(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["artifact_id"]), []).append(row)

    eligible: dict[str, dict[str, Any]] = {}
    for artifact_id, candidates in grouped.items():
        if artifact_id in INSPECTED_ARTIFACT_IDS or len(candidates) != 2:
            continue
        if len({candidate["completion"] for candidate in candidates}) != 2:
            continue
        histories = [candidate["criteria_history"] for candidate in candidates]
        if (
            not histories[0]
            or any(history != histories[0] for history in histories[1:])
        ):
            continue
        roots = eligible_world_roots(histories[0][0])
        if len(roots) < NUM_WORLDS:
            continue
        world_ids = _world_ids_for_artifact(artifact_id, roots)
        eligible[artifact_id] = {
            "world_ids": world_ids,
            "eligible_world_count": len(roots),
            "candidate_has_question": [
                "?" in candidate["completion"] for candidate in candidates
            ],
        }
    return eligible


def split_artifacts(
    eligible: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    artifact_ids = sorted(eligible)
    random.Random(SELECTION_SEED).shuffle(artifact_ids)
    cursor = 0
    splits: dict[str, list[str]] = {}
    for name, size in SPLIT_SIZES.items():
        splits[name] = artifact_ids[cursor : cursor + size]
        cursor += size
    splits["holdout"] = artifact_ids[cursor:]
    return splits


def build_manifest(path: Path, *, enforce_frozen: bool = True) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != CREATIVE_WRITING_SHA256:
        raise ValueError("DiscoverLLM creative-writing source hash changed")
    rows = _load_turn_one_rows(path)
    source_artifacts = len({row["artifact_id"] for row in rows})
    eligible = eligible_artifacts(rows)
    splits = split_artifacts(eligible)
    split_hashes = {
        name: ordered_hash(artifact_ids)
        for name, artifact_ids in splits.items()
    }
    world_hashes = {
        name: ordered_hash(
            [
                f"{artifact_id}:{','.join(eligible[artifact_id]['world_ids'])}"
                for artifact_id in artifact_ids
            ]
        )
        for name, artifact_ids in splits.items()
    }
    mechanics_ids = tuple(splits["mechanics"])

    if enforce_frozen:
        if EXPECTED_SOURCE_ARTIFACTS is None:
            raise ValueError("manifest constants have not been frozen")
        if source_artifacts != EXPECTED_SOURCE_ARTIFACTS:
            raise ValueError("DiscoverLLM source artifact count changed")
        if len(eligible) != EXPECTED_ELIGIBLE_ARTIFACTS:
            raise ValueError("DiscoverLLM eligible artifact count changed")
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("DiscoverLLM split hashes changed")
        if world_hashes != EXPECTED_WORLD_HASHES:
            raise ValueError("DiscoverLLM world-selection hashes changed")
        if mechanics_ids != EXPECTED_MECHANICS_IDS:
            raise ValueError("DiscoverLLM mechanics IDs changed")

    return {
        "interface_version": "discoverllm-priority-world-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "creative_writing_sha256": digest,
            "source_artifacts": source_artifacts,
        },
        "construction": {
            "selection_seed": SELECTION_SEED,
            "num_priority_worlds": NUM_WORLDS,
            "minimum_world_nodes": MIN_WORLD_NODES,
            "minimum_world_depth": MIN_WORLD_DEPTH,
            "inspected_artifacts_excluded": sorted(INSPECTED_ARTIFACT_IDS),
            "eligible_artifacts": len(eligible),
            "eligible_world_count_histogram": dict(
                sorted(
                    Counter(
                        details["eligible_world_count"]
                        for details in eligible.values()
                    ).items()
                )
            ),
            "artifacts_with_interactive_candidate": sum(
                any(details["candidate_has_question"])
                for details in eligible.values()
            ),
        },
        "splits": {
            name: {
                "artifact_ids": artifact_ids,
                "ordered_sha256": split_hashes[name],
                "world_selection_sha256": world_hashes[name],
                "artifacts_with_interactive_candidate": sum(
                    any(eligible[artifact_id]["candidate_has_question"])
                    for artifact_id in artifact_ids
                ),
            }
            for name, artifact_ids in splits.items()
        },
        "gates": {
            "at_least_100_eligible_artifacts": len(eligible) >= 100,
            "all_splits_nonempty": all(splits.values()),
            "mechanics_all_have_interactive_candidate": all(
                any(eligible[artifact_id]["candidate_has_question"])
                for artifact_id in splits["mechanics"]
            ),
            "shared_two_root_actions_per_artifact": True,
            "four_structurally_deep_hidden_worlds_per_artifact": True,
        },
        "prompt_content_emitted": False,
        "criterion_content_emitted": False,
        "completion_content_emitted": False,
        "source_metadata_emitted": False,
        "released_scores_read": False,
        "released_winner_labels_read": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--show-unfrozen-constants",
        action="store_true",
        help="Print prospective counts/hashes before freezing source constants.",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        args.creative_writing,
        enforce_frozen=not args.show_unfrozen_constants,
    )
    if not args.show_unfrozen_constants:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
