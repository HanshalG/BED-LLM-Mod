#!/usr/bin/env python3
"""Build the frozen RevengeBench paired execution opportunity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-execution-opportunity-v1"
ARENAS = ("battlesnake", "halite", "huskybench", "robocode")
ENTRYPOINTS = {
    "battlesnake": "main.py",
    "halite": "main.c",
    "huskybench": "player.py",
    "robocode": "MyTank.java",
}
PROBE_PREFIX = "revengebench-execution-probe-20260813:"
SEED_PREFIX = "revengebench-execution-seed-20260813:"
ENDPOINT_PREFIX = "revengebench-execution-endpoint-20260813:"
PUBLIC_HASH_PREFIX = "revengebench-execution-public-20260813:"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seed(prefix: str, arena: str, index: int) -> int:
    digest = hashlib.sha256(f"{prefix}{arena}:{index}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big")


def build_manifest(source_root: Path, admission_path: Path, replay_path: Path, source_opportunity_path: Path) -> dict[str, Any]:
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    source_opportunity = json.loads(source_opportunity_path.read_text(encoding="utf-8"))
    if replay.get("status") != "pass":
        raise ValueError("replay predecessor did not pass")
    if source_opportunity.get("status") != "execution_opportunity_required":
        raise ValueError("source opportunity predecessor did not authorize execution audit")

    arenas: dict[str, Any] = {}
    for arena in ARENAS:
        arena_data = admission["targets"]["arenas"][arena]
        hypotheses = list(arena_data["splits"]["opportunity"])
        reserve = list(arena_data["splits"]["reserve"])
        probes = sorted(reserve, key=lambda target_id: sha256_text(PROBE_PREFIX + arena + ":" + target_id))[:3]
        if len(hypotheses) != 3 or len(probes) != 3 or set(hypotheses) & set(probes):
            raise ValueError(f"{arena}: malformed frozen hypothesis/probe split")
        for role, target_ids in (("hypothesis", hypotheses), ("probe", probes)):
            for target_id in target_ids:
                path = source_root / "data" / "targets" / arena / target_id / ENTRYPOINTS[arena]
                if not path.is_file():
                    raise ValueError(f"{arena}: missing {role} entrypoint")
        arenas[arena] = {
            "hypotheses": [
                {
                    "target_id": target_id,
                    "public_hash": sha256_text(PUBLIC_HASH_PREFIX + arena + ":hypothesis:" + target_id),
                }
                for target_id in hypotheses
            ],
            "probes": [
                {
                    "target_id": target_id,
                    "public_hash": sha256_text(PUBLIC_HASH_PREFIX + arena + ":probe:" + target_id),
                    "selection_hash": sha256_text(PROBE_PREFIX + arena + ":" + target_id),
                }
                for target_id in probes
            ],
            "probe_seeds": [seed(SEED_PREFIX, arena, index) for index in range(3)],
            "endpoint_seeds": [seed(ENDPOINT_PREFIX, arena, index) for index in range(3)],
            "entrypoint": ENTRYPOINTS[arena],
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_bindings": {
            "revengebench_commit": "351a5a7c2671150bae44c8bc46d7115ec996615f",
            "revengebench_tree": "8991d42d09f3f8fb095580d87a68ba21b7dc6f5c",
        },
        "betas": [0.5, 1.0, 2.0],
        "max_target_decisions_per_simulation": 64,
        "arenas": arenas,
        "privacy": {
            "source_content_read": False,
            "provenance_opened": False,
            "trajectory_or_outcome_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--source-opportunity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_manifest(args.source_root, args.admission_audit, args.replay_audit, args.source_opportunity_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arena_count": len(result["arenas"]), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
