#!/usr/bin/env python3
"""Validate and freeze content-blind InfoQuest LLM-BED splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any


SOURCE_REPOSITORY = "https://huggingface.co/datasets/bryanlincoln/infoquest"
SOURCE_REVISION = "1f54a770a8bed73edcab86411254fff64ed53d25"
SOURCE_FILES = {
    "seed_messages": "seed_messages.jsonl",
    "settings": "settings.jsonl",
    "traits": "traits.jsonl",
}
SOURCE_SHA256 = {
    "seed_messages": (
        "13ce547f56a5a6c80afb1327cc3ed099a4624ce841b1fc6f913f63db4b29692d"
    ),
    "settings": (
        "f8b1a8a0e692ce9877bb216794b6c8574a04794770a4c0b924f91dfddcbc08bb"
    ),
    "traits": (
        "628553f157166d1e0c53dfb374d629c704f9cfb93db23c7f9cd2f3ab3b8503c8"
    ),
}
EXPECTED_RECORDS = 500
SELECTION_SEED = 24_416
MECHANICS_IDS = [0, 1]
SPLIT_SIZES = {
    "opportunity": 80,
    "development": 30,
}
EXPECTED_SPLIT_HASHES = {
    "mechanics": (
        "463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c"
    ),
    "opportunity": (
        "737296c449680cfb7c78aa03d44508af487eb9624d0e416d37cf5ddbaa01958b"
    ),
    "development": (
        "068587d494c71d5488da4f1d53ebe34be713c178083f6be98733085337c48b48"
    ),
    "holdout": (
        "91132054ecf67049deaf1644d8e6564f4c1538559a4735c676deafc69af6ea91"
    ),
}
EXPECTED_COMBINED_SPLITS_HASH = (
    "1d8f5adbfd30677311ec1a150e2b7c1f7d0804c6ed1d9facf7d13d82b0957edf"
)

SEED_KEYS = {
    "id",
    "persona1",
    "persona2",
    "persona3",
    "prompt",
    "prompt_tokens",
    "seed_message",
    "seed_message_tokens",
}
PERSONA_KEYS = {"id", "persona"}
SETTINGS_KEYS = {"id", "seed_message", "setting1", "setting2"}
SETTING_KEYS = {
    "description",
    "goal",
    "obstacle",
    "constraints",
    "solution",
    "checklist",
    "persona",
    "setting_tokens",
    "prompt_tokens",
}
TRAITS_KEYS = {"id", "traits1", "traits1_tokens", "traits2", "traits2_tokens"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("InfoQuest source root is not a readable git checkout") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(row)
    return rows


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} has unexpected fields")


def _require_nonempty_string(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")


def _require_nonnegative_int(value: Any, context: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")


def _validate_seed(row: dict[str, Any], record_id: int) -> None:
    _require_exact_keys(row, SEED_KEYS, f"seed record {record_id}")
    if row["id"] != record_id:
        raise ValueError(f"seed record {record_id} has a mismatched id")
    for field in ("prompt", "seed_message"):
        _require_nonempty_string(row[field], f"seed record {record_id}.{field}")
    for field in ("prompt_tokens", "seed_message_tokens"):
        _require_nonnegative_int(row[field], f"seed record {record_id}.{field}")
    for index in (1, 2, 3):
        persona = row[f"persona{index}"]
        if not isinstance(persona, dict):
            raise ValueError(f"seed record {record_id}.persona{index} is not an object")
        _require_exact_keys(
            persona,
            PERSONA_KEYS,
            f"seed record {record_id}.persona{index}",
        )
        _require_nonnegative_int(
            persona["id"],
            f"seed record {record_id}.persona{index}.id",
        )
        _require_nonempty_string(
            persona["persona"],
            f"seed record {record_id}.persona{index}.persona",
        )


def _validate_setting(
    setting: Any,
    *,
    record_id: int,
    index: int,
    expected_persona: str,
) -> None:
    context = f"settings record {record_id}.setting{index}"
    if not isinstance(setting, dict):
        raise ValueError(f"{context} is not an object")
    _require_exact_keys(setting, SETTING_KEYS, context)
    for field in (
        "description",
        "goal",
        "obstacle",
        "solution",
        "persona",
    ):
        _require_nonempty_string(setting[field], f"{context}.{field}")
    if setting["persona"] != expected_persona:
        raise ValueError(f"{context}.persona does not match the seed record")
    for field in ("setting_tokens", "prompt_tokens"):
        _require_nonnegative_int(setting[field], f"{context}.{field}")
    for field in ("constraints", "checklist"):
        values = setting[field]
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"{context}.{field} must contain exactly five items")
        for item_index, value in enumerate(values):
            _require_nonempty_string(
                value,
                f"{context}.{field}[{item_index}]",
            )


def _validate_settings(
    row: dict[str, Any],
    seed: dict[str, Any],
    record_id: int,
) -> None:
    _require_exact_keys(row, SETTINGS_KEYS, f"settings record {record_id}")
    if row["id"] != record_id:
        raise ValueError(f"settings record {record_id} has a mismatched id")
    if row["seed_message"] != seed["seed_message"]:
        raise ValueError(f"settings record {record_id} seed message does not match")
    for index in (1, 2):
        _validate_setting(
            row[f"setting{index}"],
            record_id=record_id,
            index=index,
            expected_persona=seed[f"persona{index}"]["persona"],
        )


def _validate_traits(row: dict[str, Any], record_id: int) -> None:
    _require_exact_keys(row, TRAITS_KEYS, f"traits record {record_id}")
    if row["id"] != record_id:
        raise ValueError(f"traits record {record_id} has a mismatched id")
    for field in ("traits1", "traits2"):
        _require_nonempty_string(row[field], f"traits record {record_id}.{field}")
    for field in ("traits1_tokens", "traits2_tokens"):
        _require_nonnegative_int(row[field], f"traits record {record_id}.{field}")


def split_record_ids() -> dict[str, list[int]]:
    remaining = list(range(2, EXPECTED_RECORDS))
    random.Random(SELECTION_SEED).shuffle(remaining)
    opportunity_end = SPLIT_SIZES["opportunity"]
    development_end = opportunity_end + SPLIT_SIZES["development"]
    return {
        "mechanics": list(MECHANICS_IDS),
        "opportunity": remaining[:opportunity_end],
        "development": remaining[opportunity_end:development_end],
        "holdout": remaining[development_end:],
    }


def build_manifest(
    source_root: Path,
    *,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    revision = _git_revision(source_root)
    if revision != SOURCE_REVISION:
        raise ValueError("InfoQuest source revision changed")

    paths = {
        name: source_root / filename for name, filename in SOURCE_FILES.items()
    }
    digests = {name: sha256_file(path) for name, path in paths.items()}
    if digests != SOURCE_SHA256:
        raise ValueError("InfoQuest source hashes changed")

    rows_by_name = {name: _load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != EXPECTED_RECORDS for rows in rows_by_name.values()):
        raise ValueError("InfoQuest source record count changed")

    expected_ids = list(range(EXPECTED_RECORDS))
    for name, rows in rows_by_name.items():
        if [row.get("id") for row in rows] != expected_ids:
            raise ValueError(f"InfoQuest {name} ids are not exactly 0..499 in order")

    record_hashes: dict[str, dict[str, str]] = {}
    for record_id in expected_ids:
        seed = rows_by_name["seed_messages"][record_id]
        settings = rows_by_name["settings"][record_id]
        traits = rows_by_name["traits"][record_id]
        _validate_seed(seed, record_id)
        _validate_settings(settings, seed, record_id)
        _validate_traits(traits, record_id)
        record_hashes[str(record_id)] = {
            "seed_messages": canonical_sha256(seed),
            "settings": canonical_sha256(settings),
            "traits": canonical_sha256(traits),
        }

    splits = split_record_ids()
    split_hashes = {
        name: canonical_sha256(record_ids)
        for name, record_ids in splits.items()
    }
    combined_hash = canonical_sha256(splits)
    if enforce_frozen:
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError("InfoQuest split hashes changed")
        if combined_hash != EXPECTED_COMBINED_SPLITS_HASH:
            raise ValueError("InfoQuest combined split hash changed")

    return {
        "interface_version": "infoquest-llm-bed-manifest-1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": revision,
            "files": {
                name: {
                    "filename": SOURCE_FILES[name],
                    "records": len(rows_by_name[name]),
                    "sha256": digests[name],
                }
                for name in SOURCE_FILES
            },
            "record_sha256_by_id": record_hashes,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "mechanics_ids_fixed_before_shuffle": True,
            "splits": {
                name: {
                    "record_ids": record_ids,
                    "records": len(record_ids),
                    "ordered_sha256": split_hashes[name],
                }
                for name, record_ids in splits.items()
            },
            "combined_splits_sha256": combined_hash,
        },
        "gates": {
            "source_revision_matches": revision == SOURCE_REVISION,
            "all_source_hashes_match": digests == SOURCE_SHA256,
            "exactly_500_aligned_records": all(
                len(rows) == EXPECTED_RECORDS for rows in rows_by_name.values()
            ),
            "ids_are_exactly_0_through_499": True,
            "all_records_match_frozen_schema": True,
            "all_settings_have_five_constraints_and_five_checklist_items": True,
            "all_setting_personas_and_seed_messages_cross_match": True,
            "all_splits_nonempty_and_disjoint": (
                all(splits.values())
                and len(set().union(*(set(ids) for ids in splits.values())))
                == EXPECTED_RECORDS
                and sum(len(ids) for ids in splits.values()) == EXPECTED_RECORDS
            ),
        },
        "content_emitted": False,
        "seed_message_content_emitted": False,
        "persona_content_emitted": False,
        "setting_content_emitted": False,
        "traits_content_emitted": False,
        "checklist_content_emitted": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-unfrozen-constants", action="store_true")
    args = parser.parse_args()
    result = build_manifest(
        args.source_root,
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
