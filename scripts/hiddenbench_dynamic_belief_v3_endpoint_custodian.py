#!/usr/bin/env python3
"""Extract opaque HiddenBench V3 endpoints after a verified pass token."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hiddenbench_dynamic_belief_v3_pass_token import verify_token


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
    return digest(path.read_bytes())


def endpoint_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != COHORT_COUNT:
        raise ValueError("exactly four synthetic endpoint rows are required")
    endpoints = []
    for index, row in enumerate(rows):
        options = [str(value).strip() for value in row["possible_answers"]]
        answer = str(row["correct_answer"]).strip()
        if options.count(answer) != 1:
            raise RuntimeError("endpoint answer mapping is invalid")
        endpoints.append(
            {"slot": f"T{index + 1}", "correct_option_id": f"O{options.index(answer) + 1}"}
        )
    return {"endpoints": endpoints}


def extract(source_path: Path) -> dict[str, Any]:
    if file_digest(source_path) != SOURCE_SHA256:
        raise RuntimeError("HiddenBench benchmark binding changed")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    ordered = sorted(rows, key=lambda row: digest((SALT + str(row["id"])).encode()))
    selected = ordered[COHORT_START : COHORT_START + COHORT_COUNT]
    if digest(canonical([str(row["id"]) for row in selected])) != EXPECTED_ORDERED_ID_SHA256:
        raise RuntimeError("HiddenBench V3 endpoint cohort changed")
    return endpoint_rows(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pass-token", type=Path, required=True)
    parser.add_argument("--execution-binding", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--raw-responses", type=Path, required=True)
    parser.add_argument("--label-free-result", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("V3 endpoint output already exists")
    verify_token(
        args.pass_token,
        execution_binding=args.execution_binding,
        source_manifest=args.source_manifest,
        source_audit=args.source_audit,
        raw_responses=args.raw_responses,
        label_free_result=args.label_free_result,
        verification=args.verification,
    )
    value = extract(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
