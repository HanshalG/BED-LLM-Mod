#!/usr/bin/env python3
"""Freeze a value-blind, language-stratified split of ICAE-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "66bbabb20a2138d066ac7d6f7ba6768b57c2f79b"
EXPECTED_BUNDLE_SHA256 = (
    "b054e8f03b3c434ffaec3c4ee6cf712d3f25e5bc7cd7ef9eb0a913026fd827d7"
)
SOURCE_URL = "https://github.com/ALEX-nlp/ICAE-EVAL"
BUNDLE_URL = (
    "https://zenodo.org/records/21639512/files/"
    "icae_prd_bundle.tar.gz?download=1"
)
SEED = 50000
PARTITION_COUNTS_PER_LANGUAGE = {
    "mechanics": 1,
    "development": 3,
    "confirmation": 4,
    "retained": 32,
}
EXPECTED_LANGUAGES = {
    "C#",
    "C++",
    "Dart",
    "Go",
    "Java",
    "JavaScript",
    "Kotlin",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "TypeScript",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_alias_rows(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != 480:
        raise ValueError("Expected exactly 480 ICAE alias records")

    rows: list[dict[str, str]] = []
    for alias, record in raw.items():
        if not isinstance(record, dict):
            raise ValueError(f"Alias record must be an object: {alias}")
        if record.get("alias") != alias:
            raise ValueError(f"Alias field mismatch: {alias}")
        language = record.get("language")
        if language not in EXPECTED_LANGUAGES:
            raise ValueError(f"Unexpected language for {alias}: {language}")
        rows.append({"alias": alias, "language": language})

    aliases = [row["alias"] for row in rows]
    if len(set(aliases)) != 480:
        raise ValueError("ICAE aliases must be unique")
    return sorted(rows, key=lambda row: row["alias"])


def partition_aliases(
    rows: list[dict[str, str]],
    *,
    seed: int = SEED,
) -> dict[str, list[dict[str, str]]]:
    by_language: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row["alias"])
    if set(by_language) != EXPECTED_LANGUAGES:
        raise ValueError("ICAE language set does not match the frozen release")
    bad_counts = {
        language: len(aliases)
        for language, aliases in by_language.items()
        if len(aliases) != 40
    }
    if bad_counts:
        raise ValueError(f"Expected 40 aliases per language, found {bad_counts}")

    partitions = {name: [] for name in PARTITION_COUNTS_PER_LANGUAGE}
    for language in sorted(by_language):
        ordered = sorted(
            by_language[language],
            key=lambda alias: (
                sha256_text(f"{seed}:{language}:{alias}"),
                alias,
            ),
        )
        cursor = 0
        for partition, count in PARTITION_COUNTS_PER_LANGUAGE.items():
            selected = ordered[cursor : cursor + count]
            cursor += count
            partitions[partition].extend(
                {"alias": alias, "language": language}
                for alias in selected
            )
        if cursor != 40:
            raise AssertionError("Partition counts must exhaust each language")

    for partition_rows in partitions.values():
        partition_rows.sort(key=lambda row: (row["language"], row["alias"]))
    return partitions


def _validate_task_files(repo: Path, alias: str) -> dict[str, str]:
    relative_paths = {
        "oracle_record": f"user_agent/prd_json/{alias}.json",
        "normal_prd": f"fuzzy_prds/{alias}/start.md",
        "medium_prd": f"fuzzy_prds_medium/{alias}/start.md",
        "easy_prd": f"fuzzy_prds_easy/{alias}/start.md",
    }
    hashes: dict[str, str] = {}
    for label, relative in relative_paths.items():
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"Missing {label} for {alias}: {relative}")
        hashes[f"{label}_sha256"] = sha256_bytes(path.read_bytes())
    return {**relative_paths, **hashes}


def build_manifest(repo: Path, bundle: Path) -> dict[str, Any]:
    commit = git_output(repo, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Expected ICAE commit {EXPECTED_COMMIT}, found {commit}")
    bundle_sha256 = sha256_bytes(bundle.read_bytes())
    if bundle_sha256 != EXPECTED_BUNDLE_SHA256:
        raise ValueError(
            "ICAE PRD bundle hash mismatch: "
            f"expected {EXPECTED_BUNDLE_SHA256}, found {bundle_sha256}"
        )

    alias_path = repo / "repo_alias.json"
    rows = load_alias_rows(alias_path)
    partitions = partition_aliases(rows)

    task_files = {
        row["alias"]: _validate_task_files(repo, row["alias"])
        for row in rows
    }
    public_partitions: dict[str, list[dict[str, str]]] = {}
    for partition, partition_rows in partitions.items():
        public_partitions[partition] = [
            {**row, **task_files[row["alias"]]}
            for row in partition_rows
        ]

    flattened = [
        row["alias"]
        for partition_rows in public_partitions.values()
        for row in partition_rows
    ]
    if len(flattened) != len(set(flattened)) or set(flattened) != {
        row["alias"] for row in rows
    }:
        raise AssertionError("ICAE partitions must be a disjoint exhaustive cover")

    return {
        "schema_version": 1,
        "audit": "icae_bench_value_blind_release_manifest",
        "source": SOURCE_URL,
        "source_commit": commit,
        "source_tree": git_output(repo, "rev-parse", "HEAD^{tree}"),
        "bundle": BUNDLE_URL,
        "bundle_sha256": bundle_sha256,
        "repo_alias_sha256": sha256_bytes(alias_path.read_bytes()),
        "seed": SEED,
        "partition_rule": {
            "ordering": "SHA256(seed:language:alias), then alias",
            "stratified_by": "language",
            "counts_per_language": PARTITION_COUNTS_PER_LANGUAGE,
        },
        "language_counts": {
            language: sum(row["language"] == language for row in rows)
            for language in sorted(EXPECTED_LANGUAGES)
        },
        "counts": {
            partition: len(partition_rows)
            for partition, partition_rows in public_partitions.items()
        },
        "public_fields_only": [
            "alias",
            "language",
            "relative paths",
            "file hashes",
        ],
        "excluded_fields": [
            "repository identity",
            "fuzzy PRD text",
            "hidden constraints",
            "trigger keywords",
            "oracle responses",
            "public and hidden test values",
            "policy scores",
        ],
        "partitions": public_partitions,
    }


def write_json_atomic(path: Path, payload: object) -> None:
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
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(
        args.output.resolve(),
        build_manifest(args.repo.resolve(), args.bundle.resolve()),
    )


if __name__ == "__main__":
    main()
