#!/usr/bin/env python3
"""Project the frozen HiddenBench mechanics rows without label designations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
SALT = "hiddenbench-adaptive-elicitation-v1|"
MECHANICS_COUNT = 4


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_mechanics_rows(source_path: Path) -> list[dict[str, Any]]:
    if file_digest(source_path) != SOURCE_SHA256:
        raise RuntimeError("HiddenBench benchmark binding changed")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 65:
        raise RuntimeError("HiddenBench population changed")
    ordered = sorted(
        rows,
        key=lambda row: digest((SALT + str(row["id"])).encode()),
    )
    return ordered[:MECHANICS_COUNT]


def project_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != MECHANICS_COUNT:
        raise ValueError("exactly four mechanics rows are required")
    planner: list[dict[str, Any]] = []
    router: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        slot = f"T{index + 1}"
        description = str(row["description"]).strip()
        shared = [str(value).strip() for value in row["shared_information"]]
        options = [str(value).strip() for value in row["possible_answers"]]
        private = [str(value).strip() for value in row["hidden_information"]]
        if not description or not all(shared) or not all(options) or not all(private):
            raise RuntimeError("selected HiddenBench row has an empty value")
        planner.append(
            {
                "slot": slot,
                "description": description,
                "shared_facts": [
                    {"id": f"S{i + 1}", "text": value}
                    for i, value in enumerate(shared)
                ],
                "options": [
                    {"id": f"O{i + 1}", "text": value}
                    for i, value in enumerate(options)
                ],
            }
        )
        router.append(
            {
                "slot": slot,
                "description": description,
                "private_facts": [
                    {"id": f"F{i + 1}", "text": value}
                    for i, value in enumerate(private)
                ],
            }
        )
    return {"planner": planner, "router": router}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(project_rows(select_mechanics_rows(args.source))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
