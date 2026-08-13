from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import revengebench_battlesnake_replay_audit as audit


def _state(turn: int, target_x: int, logged_you: str, game_id: str) -> dict:
    target = {
        "id": f"target-{game_id}",
        "name": "target",
        "latency": str(turn % 2),
        "head": {"x": target_x, "y": 0},
        "body": [{"x": target_x, "y": 0}],
    }
    opponent = {
        "id": f"opponent-{game_id}",
        "name": "opponent",
        "latency": str((turn + 1) % 2),
        "head": {"x": 9, "y": turn},
        "body": [{"x": 9, "y": turn}],
    }
    snakes = [target, opponent]
    return {
        "game": {"id": game_id, "ruleset": {"name": "standard"}},
        "turn": turn,
        "board": {"height": 11, "width": 11, "food": [], "hazards": [], "snakes": snakes},
        "you": copy.deepcopy(next(snake for snake in snakes if snake["name"] == logged_you)),
    }


def _write(path: Path, states: list[dict], winner: str = "target") -> None:
    rows = [{"id": "runtime-id"}, *states, {"winnerId": "runtime-winner", "winnerName": winner, "isDraw": False}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_ignores_runtime_ids_latency_and_logged_you_race(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [_state(turn, turn, "target", "left") for turn in range(5)])
    _write(
        right,
        [_state(turn, turn, "opponent" if turn == 2 else "target", "right") for turn in range(5)],
    )

    result = audit.audit_replays(left, right)

    assert result["status"] == "pass"
    assert result["counts"]["target_actions_per_arm"] == [4, 4]
    assert result["canonical_hashes"]["target_states"][0] == result["canonical_hashes"]["target_states"][1]
    assert "runtime-id" not in audit.canonical_json(result)


def test_audit_rejects_behavioral_divergence(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left_states = [_state(turn, turn, "target", "left") for turn in range(5)]
    right_states = [_state(turn, turn, "target", "right") for turn in range(5)]
    right_states[2]["board"]["snakes"][0]["head"]["y"] = 1
    right_states[2]["board"]["snakes"][0]["body"][0]["y"] = 1
    _write(left, left_states)
    _write(right, right_states)

    with pytest.raises(ValueError, match="invalid target head displacement"):
        audit.audit_replays(left, right)


def test_audit_requires_contiguous_unique_turns(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write(path, [_state(0, 0, "target", "bad"), _state(2, 1, "target", "bad")])

    with pytest.raises(ValueError, match="not contiguous"):
        audit.canonical_target_states(audit.load_trajectory(path))
