#!/usr/bin/env python3
"""Project the frozen V1 manifest onto natively callable V2 arenas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-execution-opportunity-v2"
CALLABLE_ARENAS = ("battlesnake", "halite", "huskybench")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(v1_path: Path, interface_path: Path) -> dict[str, Any]:
    v1_raw = v1_path.read_bytes()
    interface_raw = interface_path.read_bytes()
    v1 = json.loads(v1_raw)
    interface = json.loads(interface_raw)
    if interface.get("status") != "infrastructure_inconclusive":
        raise ValueError("V1 interface predecessor has unexpected status")
    if tuple(interface.get("callable_arenas", ())) != CALLABLE_ARENAS:
        raise ValueError("callable arena set does not match frozen V2")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "predecessor_bindings": {
            "v1_manifest_sha256": sha256_bytes(v1_raw),
            "interface_result_sha256": sha256_bytes(interface_raw),
        },
        "source_bindings": v1["source_bindings"],
        "betas": v1["betas"],
        "max_target_decisions_per_simulation": v1["max_target_decisions_per_simulation"],
        "minimum_robust_changed_arenas": 2,
        "arenas": {arena: v1["arenas"][arena] for arena in CALLABLE_ARENAS},
        "privacy": {
            "source_content_read": False,
            "provenance_opened": False,
            "trajectory_or_outcome_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--interface-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.v1_manifest, args.interface_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arena_count": len(result["arenas"]), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
