#!/usr/bin/env python3
"""Adjudicate the frozen multi-arena RevengeBench deterministic replay gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-deterministic-replay-v1"
MIN_PASSING_ARENAS = 4
MIN_TARGET_ACTIONS = 3


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _exact_pair(left: Any, right: Any) -> tuple[bool, list[str]]:
    hashes = [sha256_json(left), sha256_json(right)]
    return left == right, hashes


def audit_battlesnake(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "canonical_target_states_exact",
        "target_actions_exact",
        "terminal_result_exact",
        "normalized_score_exact",
        "action_distance_evaluation_inputs_exact",
        "trajectory_length_exact",
        "minimum_nontrivial_actions",
    }
    gates = result.get("gates", {})
    passed = result.get("status") == "pass" and all(gates.get(name) is True for name in required)
    return {
        "status": "pass" if passed else "fail",
        "counts": result.get("counts", {}),
        "canonical_hashes": result.get("canonical_hashes", {}),
        "gates": {name: gates.get(name) is True for name in sorted(required)},
    }


def _only_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{directory}: expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


def audit_halite(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    paths = [_only_file(left_dir, "*.hlt"), _only_file(right_dir, "*.hlt")]
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    exact, trace_hashes = _exact_pair(*traces)
    trace = traces[0]
    frame_count = len(trace.get("frames", []))
    moves = trace.get("moves", [])
    valid_shape = (
        trace.get("num_players") == 2
        and frame_count == trace.get("num_frames")
        and len(moves) == max(frame_count - 1, 0)
    )
    # Halite owner IDs are one-based. The frozen target is engine player 1.
    target_actions = 0
    if valid_shape:
        for frame, turn_moves in zip(trace["frames"], moves):
            for y, row in enumerate(turn_moves):
                for x, direction in enumerate(row):
                    if direction != 0 and frame[y][x][0] == 1:
                        target_actions += 1
    enough_actions = target_actions >= MIN_TARGET_ACTIONS
    passed = exact and valid_shape and enough_actions
    return {
        "status": "pass" if passed else "fail",
        "counts": {
            "frames_per_arm": [len(item.get("frames", [])) for item in traces],
            "target_nontrivial_actions_per_arm": [target_actions, target_actions] if exact else [target_actions, None],
        },
        "canonical_hashes": {"complete_trace": trace_hashes},
        "gates": {
            "canonical_target_states_exact": exact,
            "target_actions_exact": exact,
            "terminal_result_exact": exact,
            "normalized_score_exact": exact,
            "action_distance_evaluation_inputs_exact": exact,
            "trajectory_length_exact": exact and valid_shape,
            "minimum_nontrivial_actions": enough_actions,
        },
    }


def _husky_logs(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("game_log_*.json"), key=lambda path: int(path.name.split("_", 3)[2]))
    if not paths:
        raise ValueError(f"{directory}: no HuskyBench game logs")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _husky_target_action_count(logs: list[dict[str, Any]]) -> int:
    count = 0
    for game in logs:
        player_names = game.get("playerNames", {})
        if len(player_names) != 2:
            raise ValueError("HuskyBench game must have exactly two players")
        # Connection order is frozen target then opponent; insertion order is
        # retained by the release's JSON writer. Action records use this key.
        target_key = int(next(iter(player_names)))
        for round_data in game.get("rounds", {}).values():
            for action in round_data.get("action_sequence", []):
                if action.get("player") == target_key:
                    count += 1
    return count


def audit_huskybench(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    traces = [_husky_logs(left_dir), _husky_logs(right_dir)]
    exact, trace_hashes = _exact_pair(*traces)
    action_counts = [_husky_target_action_count(trace) for trace in traces]
    game_counts = [len(trace) for trace in traces]
    complete = game_counts == [3, 3]
    enough_actions = min(action_counts) >= MIN_TARGET_ACTIONS
    passed = exact and complete and action_counts[0] == action_counts[1] and enough_actions
    return {
        "status": "pass" if passed else "fail",
        "counts": {
            "games_per_arm": game_counts,
            "target_nontrivial_actions_per_arm": action_counts,
        },
        "canonical_hashes": {"ordered_complete_game_logs": trace_hashes},
        "gates": {
            "canonical_target_states_exact": exact,
            "target_actions_exact": exact and action_counts[0] == action_counts[1],
            "terminal_result_exact": exact,
            "normalized_score_exact": exact,
            "action_distance_evaluation_inputs_exact": exact,
            "trajectory_length_exact": exact and complete,
            "minimum_nontrivial_actions": enough_actions,
        },
    }


def _robocode_xml_prefix(path: Path) -> tuple[bytes, ET.Element]:
    raw = path.read_bytes()
    marker = b"</record>"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path}: missing XML record terminator")
    xml = raw[: end + len(marker)]
    return raw, ET.fromstring(xml)


def _robocode_target_action_count(root: ET.Element) -> int:
    states: list[tuple[str | None, ...]] = []
    for turn in root.findall("./turns/turn"):
        target = turn.find("./robots/robot[@id='0']")
        if target is None:
            raise ValueError("RoboCode turn missing target robot")
        states.append(
            tuple(
                target.get(name)
                for name in ("x", "y", "bodyHeading", "gunHeading", "radarHeading", "energy", "state")
            )
        )
    return sum(left != right for left, right in zip(states, states[1:]))


def audit_robocode(left_record: Path, right_record: Path, left_results: Path, right_results: Path) -> dict[str, Any]:
    left_raw, left_root = _robocode_xml_prefix(left_record)
    right_raw, right_root = _robocode_xml_prefix(right_record)
    result_raw = [left_results.read_bytes(), right_results.read_bytes()]
    trace_exact = left_raw == right_raw
    result_exact = result_raw[0] == result_raw[1]
    turn_counts = [len(root.findall("./turns/turn")) for root in (left_root, right_root)]
    action_counts = [_robocode_target_action_count(root) for root in (left_root, right_root)]
    enough_actions = min(action_counts) >= MIN_TARGET_ACTIONS
    passed = trace_exact and result_exact and turn_counts[0] == turn_counts[1] and action_counts[0] == action_counts[1] and enough_actions
    return {
        "status": "pass" if passed else "fail",
        "counts": {"turns_per_arm": turn_counts, "target_nontrivial_actions_per_arm": action_counts},
        "canonical_hashes": {
            "complete_record": [sha256_bytes(left_raw), sha256_bytes(right_raw)],
            "normalized_score": [sha256_bytes(value) for value in result_raw],
        },
        "gates": {
            "canonical_target_states_exact": trace_exact,
            "target_actions_exact": trace_exact and action_counts[0] == action_counts[1],
            "terminal_result_exact": trace_exact and result_exact,
            "normalized_score_exact": result_exact,
            "action_distance_evaluation_inputs_exact": trace_exact,
            "trajectory_length_exact": trace_exact and turn_counts[0] == turn_counts[1],
            "minimum_nontrivial_actions": enough_actions,
        },
    }


def adjudicate(arenas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(value.get("status") == "pass" for value in arenas.values())
    fail_count = sum(value.get("status") == "fail" for value in arenas.values())
    pending_count = sum(value.get("status") == "infrastructure_pending" for value in arenas.values())
    passed = len(arenas) == 5 and pass_count >= MIN_PASSING_ARENAS and fail_count == 0
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if passed else "fail",
        "decision": "advance_to_frozen_source_opportunity_audit" if passed else "close_revengebench_route",
        "summary": {
            "required_passes": MIN_PASSING_ARENAS,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "infrastructure_pending_count": pending_count,
        },
        "arenas": arenas,
        "privacy": {
            "target_source_serialized": False,
            "raw_trajectories_serialized": False,
            "released_prior_outcomes_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battlesnake-result", type=Path, required=True)
    parser.add_argument("--halite-left", type=Path, required=True)
    parser.add_argument("--halite-right", type=Path, required=True)
    parser.add_argument("--husky-left", type=Path, required=True)
    parser.add_argument("--husky-right", type=Path, required=True)
    parser.add_argument("--robocode-left-record", type=Path, required=True)
    parser.add_argument("--robocode-right-record", type=Path, required=True)
    parser.add_argument("--robocode-left-results", type=Path, required=True)
    parser.add_argument("--robocode-right-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arenas = {
        "battlesnake": audit_battlesnake(args.battlesnake_result),
        "halite": audit_halite(args.halite_left, args.halite_right),
        "huskybench": audit_huskybench(args.husky_left, args.husky_right),
        "robocode": audit_robocode(
            args.robocode_left_record,
            args.robocode_right_record,
            args.robocode_left_results,
            args.robocode_right_results,
        ),
        "robotrumble": {
            "status": "infrastructure_pending",
            "reason_code": "pinned_x86_64_binary_unavailable_on_local_arm_runtime",
            "scientific_failure": False,
        },
    }
    result = adjudicate(arenas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "status": result["status"], **result["summary"]}, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
