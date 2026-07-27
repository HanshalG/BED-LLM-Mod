#!/usr/bin/env python3
"""Freeze a case-ID-only partition for the pinned MedConceal release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


EXPECTED_COMMIT = "f98d02c1eb9819325f091c0afd0dc4d63d90a21b"
PARTITION_PREFIX = "medconceal-24421:"
PARTITION_SIZES = {
    "mechanics": 20,
    "development": 80,
    "confirmation": 100,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_ids(path: Path) -> list[str]:
    case_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                raise ValueError(f"Missing case_id on line {line_number}")
            case_ids.append(case_id)
    if len(case_ids) != 300:
        raise ValueError(f"Expected 300 case IDs, found {len(case_ids)}")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs are not unique")
    return case_ids


def partition_case_ids(case_ids: Iterable[str]) -> dict[str, list[str]]:
    ordered = sorted(
        case_ids,
        key=lambda case_id: (
            _sha256_bytes(f"{PARTITION_PREFIX}{case_id}".encode("utf-8")),
            case_id,
        ),
    )
    partitions: dict[str, list[str]] = {}
    offset = 0
    for name, size in PARTITION_SIZES.items():
        partitions[name] = ordered[offset : offset + size]
        offset += size
    partitions["retained"] = ordered[offset:]
    return partitions


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def tracked_content_digest(repo: Path) -> tuple[int, str]:
    paths = [path for path in _git_output(repo, "ls-files").splitlines() if path]
    entries = [
        f"{_sha256_file(repo / relative_path)}  {relative_path}\n"
        for relative_path in paths
    ]
    return len(paths), _sha256_bytes("".join(entries).encode("utf-8"))


def build_manifest(repo: Path) -> dict[str, object]:
    commit = _git_output(repo, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Expected commit {EXPECTED_COMMIT}, found {commit}")
    if _git_output(repo, "status", "--short"):
        raise ValueError("Pinned source checkout is not clean")

    cases_path = repo / "data" / "cases.jsonl"
    case_ids = load_case_ids(cases_path)
    partitions = partition_case_ids(case_ids)
    tracked_file_count, content_digest = tracked_content_digest(repo)

    return {
        "schema_version": 1,
        "source": "https://github.com/FAIRHealth/MedConceal",
        "commit": commit,
        "tracked_file_count": tracked_file_count,
        "tracked_content_manifest_sha256": content_digest,
        "cases_file_sha256": _sha256_file(cases_path),
        "partition_rule": {
            "ordering": f'SHA256("{PARTITION_PREFIX}" + case_id), then case_id',
            "sizes": {**PARTITION_SIZES, "retained": 100},
        },
        "counts": {name: len(ids) for name, ids in partitions.items()},
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
