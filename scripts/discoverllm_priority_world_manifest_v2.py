#!/usr/bin/env python3
"""Freeze all-domain DiscoverLLM priority-world experiment splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

from scripts import discoverllm_priority_world_manifest as v1


SELECTION_SEED = 24_412
SOURCE_SHA256 = {
    "creative_writing": v1.CREATIVE_WRITING_SHA256,
    "technical_writing": (
        "4dcc29e2cb9f21d4dabaa7d8eeaad773a433248a2be94498968e908b6858ccd0"
    ),
    "svg_drawing": (
        "aef93ede39dbbf570bd15bda2b217d4a96dec188af77eeb5d88bcaed869a081d"
    ),
}
SPLIT_SIZES = {
    "mechanics": 3,
    "opportunity": 60,
    "development": 30,
}

# Filled after the first content-sealed run and enforced on replay.
EXPECTED_SOURCE_ARTIFACTS = 1_484
EXPECTED_ELIGIBLE_ARTIFACTS = 255
EXPECTED_SPLIT_HASHES = {
    "mechanics": (
        "81b7adde434f41001108a0f1658327085db0892d4b672abb95f5008f39048530"
    ),
    "opportunity": (
        "10f5768769de9d8c3a79aeb9d41c22a8b675c912536b9b8f4542c1db7a25dce4"
    ),
    "development": (
        "f4ed143e120fa35fbfe5e6fa7dcdede8a50dbc71d2050e9586d365fe287a89ac"
    ),
    "holdout": (
        "bce380af0e65d5cb2d5514caf8399523aa25177fac19a68e3fbcb40291b795d8"
    ),
}
EXPECTED_WORLD_HASHES = {
    "mechanics": (
        "e1210daaf4bad3b84bd923a3573d518d44e96b98db5334703303fa172e95dbeb"
    ),
    "opportunity": (
        "00232b8853f4d5e23231d7659b4e3e5b306a20265884c2fae14c1b6c41d0be76"
    ),
    "development": (
        "0ae17210ad1ff8400cd1e165f4a34232a7f6388eb71e4265088316e86ab505df"
    ),
    "holdout": (
        "545f155f69f9e851cba20416867bd5ced275a5b991cfdea51bc8c2924101731a"
    ),
}
EXPECTED_MECHANICS_IDS = (
    "technical_writing:artifact_240",
    "creative_writing:artifact_60",
    "creative_writing:artifact_58",
)


def _load_earliest_turn_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "discoverllm_priority_world_manifest_v2 requires duckdb"
        ) from exc

    connection = duckdb.connect()
    escaped_path = str(path.resolve()).replace("'", "''")
    selected = connection.execute(
        "WITH source AS ("
        " SELECT *, MIN(CAST(turn_id AS INTEGER)) OVER "
        " (PARTITION BY artifact_id) AS earliest_turn"
        f" FROM read_parquet('{escaped_path}')"
        ") SELECT artifact_id, assistant_index, prompt, completion, "
        "criteria_history FROM source "
        "WHERE CAST(turn_id AS INTEGER) = earliest_turn "
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


def _world_ids(
    domain: str,
    artifact_id: str,
    roots: Sequence[dict[str, Any]],
) -> list[str]:
    root_ids = [str(root["id"]) for root in roots]
    seed_bytes = hashlib.sha256(
        f"{SELECTION_SEED}:{domain}:{artifact_id}".encode("utf-8")
    ).digest()[:8]
    random.Random(int.from_bytes(seed_bytes, "big")).shuffle(root_ids)
    return root_ids[: v1.NUM_WORLDS]


def eligible_artifacts(
    rows_by_domain: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    eligible: dict[str, dict[str, Any]] = {}
    for domain, rows in sorted(rows_by_domain.items()):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["artifact_id"]), []).append(row)
        for artifact_id, candidates in grouped.items():
            if (
                domain == "creative_writing"
                and artifact_id in v1.INSPECTED_ARTIFACT_IDS
            ):
                continue
            if len(candidates) != 2:
                continue
            if len({candidate["completion"] for candidate in candidates}) != 2:
                continue
            histories = [
                candidate["criteria_history"] for candidate in candidates
            ]
            if (
                not histories[0]
                or any(history != histories[0] for history in histories[1:])
            ):
                continue
            roots = v1.eligible_world_roots(histories[0][-1])
            if len(roots) < v1.NUM_WORLDS:
                continue
            key = f"{domain}:{artifact_id}"
            eligible[key] = {
                "world_ids": _world_ids(domain, artifact_id, roots),
                "eligible_world_count": len(roots),
                "candidate_has_question": [
                    "?" in candidate["completion"] for candidate in candidates
                ],
            }
    return eligible


def split_artifacts(
    eligible: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    keys = sorted(eligible)
    random.Random(SELECTION_SEED).shuffle(keys)
    cursor = 0
    splits: dict[str, list[str]] = {}
    for name, size in SPLIT_SIZES.items():
        splits[name] = keys[cursor : cursor + size]
        cursor += size
    splits["holdout"] = keys[cursor:]
    return splits


def build_manifest(
    paths: dict[str, Path],
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    digests = {domain: v1.sha256_file(path) for domain, path in paths.items()}
    if digests != SOURCE_SHA256:
        raise ValueError("DiscoverLLM all-domain source hashes changed")
    rows_by_domain = {
        domain: _load_earliest_turn_rows(path)
        for domain, path in sorted(paths.items())
    }
    source_artifacts = sum(
        len({row["artifact_id"] for row in rows})
        for rows in rows_by_domain.values()
    )
    eligible = eligible_artifacts(rows_by_domain)
    splits = split_artifacts(eligible)
    split_hashes = {
        name: v1.ordered_hash(keys) for name, keys in splits.items()
    }
    world_hashes = {
        name: v1.ordered_hash(
            [
                f"{key}:{','.join(eligible[key]['world_ids'])}"
                for key in keys
            ]
        )
        for name, keys in splits.items()
    }
    mechanics_ids = tuple(splits["mechanics"])

    if enforce_frozen:
        if EXPECTED_SOURCE_ARTIFACTS is None:
            raise ValueError("V2 manifest constants have not been frozen")
        if source_artifacts != EXPECTED_SOURCE_ARTIFACTS:
            raise ValueError("DiscoverLLM V2 source artifact count changed")
        if len(eligible) != EXPECTED_ELIGIBLE_ARTIFACTS:
            raise ValueError("DiscoverLLM V2 eligible artifact count changed")
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("DiscoverLLM V2 split hashes changed")
        if world_hashes != EXPECTED_WORLD_HASHES:
            raise ValueError("DiscoverLLM V2 world hashes changed")
        if mechanics_ids != EXPECTED_MECHANICS_IDS:
            raise ValueError("DiscoverLLM V2 mechanics IDs changed")

    return {
        "interface_version": "discoverllm-priority-world-manifest-2",
        "source": {
            "repository": v1.SOURCE_REPOSITORY,
            "commit": v1.SOURCE_COMMIT,
            "dataset_repository": v1.DATASET_REPOSITORY,
            "dataset_revision": v1.DATASET_REVISION,
            "sha256_by_domain": digests,
            "source_artifacts": source_artifacts,
        },
        "construction": {
            "selection_seed": SELECTION_SEED,
            "num_priority_worlds": v1.NUM_WORLDS,
            "minimum_world_nodes": v1.MIN_WORLD_NODES,
            "minimum_world_depth": v1.MIN_WORLD_DEPTH,
            "eligible_artifacts": len(eligible),
            "eligible_domain_counts": dict(
                sorted(
                    Counter(
                        key.split(":", 1)[0] for key in eligible
                    ).items()
                )
            ),
            "eligible_world_count_histogram": dict(
                sorted(
                    Counter(
                        details["eligible_world_count"]
                        for details in eligible.values()
                    ).items()
                )
            ),
        },
        "splits": {
            name: {
                "artifact_ids": keys,
                "ordered_sha256": split_hashes[name],
                "world_selection_sha256": world_hashes[name],
                "domain_counts": dict(
                    sorted(
                        Counter(
                            key.split(":", 1)[0] for key in keys
                        ).items()
                    )
                ),
                "artifacts_with_interactive_candidate": sum(
                    any(eligible[key]["candidate_has_question"])
                    for key in keys
                ),
            }
            for name, keys in splits.items()
        },
        "gates": {
            "all_three_source_hashes_match": digests == SOURCE_SHA256,
            "at_least_250_eligible_artifacts": len(eligible) >= 250,
            "all_splits_nonempty": all(splits.values()),
            "mechanics_all_have_interactive_candidate": all(
                any(eligible[key]["candidate_has_question"])
                for key in splits["mechanics"]
            ),
            "shared_two_root_actions_per_artifact": True,
            "four_structurally_deep_hidden_worlds_per_artifact": True,
        },
        "prompt_content_emitted": False,
        "criterion_content_emitted": False,
        "completion_content_emitted": False,
        "source_metadata_emitted": False,
        "selected_world_ids_emitted": False,
        "released_scores_read": False,
        "released_winner_labels_read": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-unfrozen-constants", action="store_true")
    args = parser.parse_args()
    result = build_manifest(
        {
            "creative_writing": args.creative_writing,
            "technical_writing": args.technical_writing,
            "svg_drawing": args.svg_drawing,
        },
        enforce_frozen=not args.show_unfrozen_constants,
    )
    if not args.show_unfrozen_constants:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
