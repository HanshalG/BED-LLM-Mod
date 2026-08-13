#!/usr/bin/env python3
"""Audit native counterfactual policy interfaces before opportunity trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-counterfactual-interface-v1"
ARENAS = ("battlesnake", "halite", "huskybench", "robocode")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def audit(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    inverse_path = source_root / "src/revenge_bench/tournaments/inverse_strategy.py"
    inverse_text = inverse_path.read_text(encoding="utf-8")
    required_fragments = {
        "battlesnake": (
            "def _process_battlesnake_traces",
            "self._query_learner(target_state)",
            "extract_state_action_pairs(sim_file, target_name)",
        ),
        "halite": (
            "def _process_halite_traces",
            "query_compiled_bot(",
            "extract_state_action_pairs(",
        ),
        "huskybench": (
            "def _process_huskybench_traces",
            "bot.get_action(round_state, remaining_chips)",
            "extract_state_action_pairs(sim_file, target_name)",
        ),
        "robocode": (
            "def _process_robocode_traces",
            "self._query_learner(target_state)",
            "fallback = (",
            ') / "main.py"',
        ),
    }
    expected_entrypoints = {
        "battlesnake": "main.py",
        "halite": "main.c",
        "huskybench": "player.py",
        "robocode": "MyTank.java",
    }
    arenas = {}
    for arena in ARENAS:
        entrypoint = manifest["arenas"][arena]["entrypoint"]
        fragments_present = all(fragment in inverse_text for fragment in required_fragments[arena])
        entrypoint_matches = entrypoint == expected_entrypoints[arena]
        callable_native = fragments_present and entrypoint_matches and arena != "robocode"
        reason = "native_target_policy_callable_on_frozen_states"
        if arena == "robocode":
            reason = (
                "released_targets_are_stateful_java_bots_but_offline_evaluator_requires_"
                "a_python_main_move_surrogate"
            )
        arenas[arena] = {
            "entrypoint": entrypoint,
            "entrypoint_matches_release": entrypoint_matches,
            "public_evaluator_contract_present": fragments_present,
            "native_counterfactual_callable": callable_native,
            "reason_code": reason,
        }

    callable_arenas = [arena for arena in ARENAS if arenas[arena]["native_counterfactual_callable"]]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "infrastructure_inconclusive",
        "decision": "freeze_callable_arena_successor_before_trajectories",
        "reason": (
            "The V1 likelihood requires every candidate target policy to act on every frozen target state. "
            "RoboCode target policies are Java engine bots, while the release's offline evaluator queries a "
            "separate learned Python move(state) surrogate that is absent from the frozen target bank."
        ),
        "arenas": arenas,
        "callable_arenas": callable_arenas,
        "v1_all_arena_gate_pass": False,
        "bindings": {
            "execution_manifest_sha256": sha256_bytes(manifest_raw),
            "inverse_strategy_sha256": sha256_bytes(inverse_path.read_bytes()),
        },
        "privacy": {
            "selected_source_content_read": False,
            "trajectory_or_outcome_opened": False,
            "provenance_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.source_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"callable_arenas": result["callable_arenas"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
