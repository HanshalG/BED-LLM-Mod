#!/usr/bin/env python3
"""Project frozen HiddenBench reserve tasks into disjoint private views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
SALT = "hiddenbench-adaptive-elicitation-v1|"
RESERVE_OFFSET = 56
COHORT_COUNT = 4
EXPECTED_ORDERED_ID_SHA256 = "adcbabc78acc15d7b88d2b0552636525ea64d25840113033a469c558bae3807a"


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


def selected_rows(source_path: Path) -> list[dict[str, Any]]:
    if file_digest(source_path) != SOURCE_SHA256:
        raise RuntimeError("HiddenBench benchmark binding changed")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 65:
        raise RuntimeError("HiddenBench population changed")
    ordered = sorted(
        rows,
        key=lambda row: digest((SALT + str(row["id"])).encode()),
    )
    selected = ordered[RESERVE_OFFSET : RESERVE_OFFSET + COHORT_COUNT]
    selected_hash = digest(canonical([str(row["id"]) for row in selected]))
    if selected_hash != EXPECTED_ORDERED_ID_SHA256:
        raise RuntimeError("HiddenBench reserve-prefix selection changed")
    return selected


def planner_router_views(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != COHORT_COUNT:
        raise ValueError("exactly four reserve rows are required")
    planner: list[dict[str, Any]] = []
    router: list[dict[str, Any]] = []
    for task_index, row in enumerate(rows):
        slot = f"T{task_index + 1}"
        description = str(row["description"]).strip()
        shared = [str(value).strip() for value in row["shared_information"]]
        options = [str(value).strip() for value in row["possible_answers"]]
        private = [str(value).strip() for value in row["hidden_information"]]
        if not description or not all(shared) or not all(options) or not all(private):
            raise RuntimeError("selected HiddenBench row contains an empty value")
        planner.append(
            {
                "slot": slot,
                "description": description,
                "shared_facts": [
                    {"id": f"S{index + 1}", "text": text}
                    for index, text in enumerate(shared)
                ],
                "options": [
                    {"id": f"O{index + 1}", "text": text}
                    for index, text in enumerate(options)
                ],
            }
        )
        router.append(
            {
                "slot": slot,
                "description": description,
                "private_facts": [
                    {"id": f"F{index + 1}", "text": text}
                    for index, text in enumerate(private)
                ],
            }
        )
    return {"planner": planner, "router": router}


def endpoint_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for task_index, row in enumerate(rows):
        options = [str(value).strip() for value in row["possible_answers"]]
        answer = str(row["correct_answer"]).strip()
        if options.count(answer) != 1:
            raise RuntimeError("selected HiddenBench answer mapping is invalid")
        endpoints.append(
            {
                "slot": f"T{task_index + 1}",
                "correct_option_id": f"O{options.index(answer) + 1}",
            }
        )
    return {"endpoints": endpoints}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--view", choices=("planner-router", "endpoint"), required=True
    )
    args = parser.parse_args()
    rows = selected_rows(args.source.resolve())
    value = (
        planner_router_views(rows)
        if args.view == "planner-router"
        else endpoint_view(rows)
    )
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
