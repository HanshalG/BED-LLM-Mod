#!/usr/bin/env python3
"""Bind the V3 deterministic BattleSnake mechanics repair before execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-execution-opportunity-v3"
BATTLESNAKE_COMMIT = "aaf48003cfa5034a14d7053bb0ccc0bc6ae99cee"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(v2_path: Path, patch_path: Path) -> dict[str, Any]:
    v2_raw = v2_path.read_bytes()
    patch_raw = patch_path.read_bytes()
    v2 = json.loads(v2_raw)
    if v2.get("protocol_version") != "revengebench-execution-opportunity-v2":
        raise ValueError("unexpected V2 manifest")
    return {
        **v2,
        "protocol_version": PROTOCOL_VERSION,
        "predecessor_bindings": {
            **v2["predecessor_bindings"],
            "v2_manifest_sha256": sha256_bytes(v2_raw),
        },
        "mechanics_repair": {
            "arena": "battlesnake",
            "upstream_commit": BATTLESNAKE_COMMIT,
            "patch_sha256": sha256_bytes(patch_raw),
            "minimum_fresh_arm_preflight": 3,
            "v2_trajectories_reused": False,
        },
        "privacy": {
            "source_content_read": False,
            "repaired_trajectory_opened": False,
            "prior_released_outcome_opened": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.v2_manifest, args.patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arena_count": len(result["arenas"]), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
