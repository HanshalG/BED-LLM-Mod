#!/usr/bin/env python3
"""Bind the SDK-correct Halite feeder before corrected trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-execution-opportunity-v4"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(v3_path: Path, runner_path: Path) -> dict[str, Any]:
    v3_raw = v3_path.read_bytes()
    runner_raw = runner_path.read_bytes()
    v3 = json.loads(v3_raw)
    if v3.get("protocol_version") != "revengebench-execution-opportunity-v3":
        raise ValueError("unexpected V3 manifest")
    return {
        **v3,
        "protocol_version": PROTOCOL_VERSION,
        "predecessor_bindings": {
            **v3["predecessor_bindings"],
            "v3_manifest_sha256": sha256_bytes(v3_raw),
        },
        "halite_feeder_repair": {
            "runner_sha256": sha256_bytes(runner_raw),
            "getinit_contains_frame_zero": True,
            "first_getframe_contains_frame_zero": True,
            "decision_frame_index_equals_move_index": True,
            "v3_trajectories_reused": False,
        },
        "privacy": {
            "corrected_halite_trajectory_opened": False,
            "prior_released_outcome_opened": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--halite-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.v3_manifest, args.halite_runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arena_count": len(result["arenas"]), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
