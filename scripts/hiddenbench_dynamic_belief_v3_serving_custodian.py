#!/usr/bin/env python3
"""Emit only planner and router views for the final HiddenBench cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
SALT = "hiddenbench-adaptive-elicitation-v1|"
COHORT_START = 60
COHORT_COUNT = 4
EXPECTED_ORDERED_ID_SHA256 = "822314c2c662a2711b6d2253601b9a801071c4d53bb53b92bb115240d98bee40"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_rows(source_path: Path) -> list[dict[str, Any]]:
    if file_digest(source_path) != SOURCE_SHA256:
        raise RuntimeError("HiddenBench benchmark binding changed")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 65:
        raise RuntimeError("HiddenBench population changed")
    ordered = sorted(
        rows,
        key=lambda row: digest((SALT + str(row["id"])).encode()),
    )
    selected = ordered[COHORT_START : COHORT_START + COHORT_COUNT]
    if digest(canonical([str(row["id"]) for row in selected])) != EXPECTED_ORDERED_ID_SHA256:
        raise RuntimeError("HiddenBench V3 cohort changed")
    return selected


def project(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != COHORT_COUNT:
        raise ValueError("exactly four V3 rows are required")
    planner: list[dict[str, Any]] = []
    router: list[dict[str, Any]] = []
    for task_index, row in enumerate(rows):
        slot = f"T{task_index + 1}"
        planner.append(
            {
                "slot": slot,
                "description": str(row["description"]).strip(),
                "shared_facts": [
                    {"id": f"S{index + 1}", "text": str(text).strip()}
                    for index, text in enumerate(row["shared_information"])
                ],
                "options": [
                    {"id": f"O{index + 1}", "text": str(text).strip()}
                    for index, text in enumerate(row["possible_answers"])
                ],
            }
        )
        router.append(
            {
                "slot": slot,
                "description": str(row["description"]).strip(),
                "private_facts": [
                    {"id": f"F{index + 1}", "text": str(text).strip()}
                    for index, text in enumerate(row["hidden_information"])
                ],
            }
        )
    if any(
        not task["description"]
        or not all(item["text"] for item in task["shared_facts"])
        or not all(item["text"] for item in task["options"])
        for task in planner
    ) or any(
        not all(item["text"] for item in task["private_facts"])
        for task in router
    ):
        raise RuntimeError("V3 serving projection contains an empty value")
    return {"planner": planner, "router": router}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(project(select_rows(args.source.resolve()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
