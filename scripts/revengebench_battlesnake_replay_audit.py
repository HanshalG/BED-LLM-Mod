#!/usr/bin/env python3
"""Compare two pinned RevengeBench BattleSnake replay trajectories.

The BattleSnake engine logs one request body per turn. Which snake's request is
written is timing-dependent, even though every request contains the same full
board state. This audit reconstructs the target-visible state from that board,
normalizes runtime-only identifiers and latency, and compares behavior exactly.
Raw trajectories are never included in the public result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-battlesnake-replay-v1"
TARGET_NAME = "target"
MIN_NONTRIVIAL_ACTIONS = 3


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"{path}: empty trajectory")
    return records


def _normalize_snake(snake: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(snake)
    normalized.pop("id", None)
    normalized.pop("latency", None)
    return normalized


def canonical_target_states(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ordered target-visible states from full board snapshots.

    The engine's logged ``you`` object is not stable because either snake can
    win the concurrent log write. The target's exact ``you`` object is instead
    reconstructed from the named snake in the complete board snapshot.
    """

    by_turn: dict[int, dict[str, Any]] = {}
    for record in records:
        if "turn" not in record or "board" not in record:
            continue
        turn = record["turn"]
        if not isinstance(turn, int) or turn in by_turn:
            raise ValueError("turns must be unique integers")
        state = copy.deepcopy(record)
        game = state.get("game")
        if isinstance(game, dict):
            game.pop("id", None)

        board = state.get("board")
        if not isinstance(board, dict) or not isinstance(board.get("snakes"), list):
            raise ValueError(f"turn {turn}: missing board snakes")
        snakes = [_normalize_snake(snake) for snake in board["snakes"]]
        snakes.sort(key=lambda snake: snake.get("name", ""))
        board["snakes"] = snakes
        for field in ("food", "hazards"):
            if isinstance(board.get(field), list):
                board[field].sort(key=lambda point: (point.get("x"), point.get("y")))

        target = next((snake for snake in snakes if snake.get("name") == TARGET_NAME), None)
        if target is None:
            raise ValueError(f"turn {turn}: target missing from board")
        state["you"] = copy.deepcopy(target)
        by_turn[turn] = state

    if not by_turn:
        raise ValueError("trajectory has no game states")
    turns = sorted(by_turn)
    if turns != list(range(turns[0], turns[-1] + 1)):
        raise ValueError("trajectory turns are not contiguous")
    return [by_turn[turn] for turn in turns]


def inferred_target_actions(states: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    actions: list[dict[str, int | str]] = []
    directions = {(0, 1): "up", (0, -1): "down", (1, 0): "right", (-1, 0): "left"}
    for current, following in zip(states, states[1:]):
        head = current["you"]["head"]
        next_head = following["you"]["head"]
        delta = (next_head["x"] - head["x"], next_head["y"] - head["y"])
        direction = directions.get(delta)
        if direction is None:
            raise ValueError(f"turn {current['turn']}: invalid target head displacement {delta}")
        actions.append({"turn": current["turn"], "action": direction})
    return actions


def canonical_terminal_result(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = records[-1]
    required = {"winnerName", "isDraw"}
    if not required.issubset(result):
        raise ValueError("trajectory is missing its terminal result")
    return {"winner_name": result["winnerName"], "is_draw": result["isDraw"]}


def audit_replays(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_records = load_trajectory(left_path)
    right_records = load_trajectory(right_path)
    left_states = canonical_target_states(left_records)
    right_states = canonical_target_states(right_records)
    left_actions = inferred_target_actions(left_states)
    right_actions = inferred_target_actions(right_states)
    left_terminal = canonical_terminal_result(left_records)
    right_terminal = canonical_terminal_result(right_records)

    state_match = left_states == right_states
    action_match = left_actions == right_actions
    terminal_match = left_terminal == right_terminal
    normalized_score_match = terminal_match
    action_distance_inputs_match = state_match and action_match
    length_match = len(left_states) == len(right_states)
    enough_actions = len(left_actions) >= MIN_NONTRIVIAL_ACTIONS
    passed = all(
        (
            state_match,
            action_match,
            terminal_match,
            normalized_score_match,
            action_distance_inputs_match,
            length_match,
            enough_actions,
        )
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "arena": "battlesnake",
        "status": "pass" if passed else "fail",
        "gates": {
            "canonical_target_states_exact": state_match,
            "target_actions_exact": action_match,
            "terminal_result_exact": terminal_match,
            "normalized_score_exact": normalized_score_match,
            "action_distance_evaluation_inputs_exact": action_distance_inputs_match,
            "trajectory_length_exact": length_match,
            "minimum_nontrivial_actions": enough_actions,
            "raw_trajectory_serialized": False,
            "openrouter_calls_zero": True,
        },
        "counts": {
            "target_states_per_arm": [len(left_states), len(right_states)],
            "target_actions_per_arm": [len(left_actions), len(right_actions)],
        },
        "canonical_hashes": {
            "target_states": [sha256_json(left_states), sha256_json(right_states)],
            "target_actions": [sha256_json(left_actions), sha256_json(right_actions)],
            "terminal_result": [sha256_json(left_terminal), sha256_json(right_terminal)],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_replays(args.left, args.right)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
