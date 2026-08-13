from __future__ import annotations

import json
from pathlib import Path

from scripts import revengebench_battlesnake_opportunity_runner as runner


def _snake(name: str, snake_id: str, head_x: int) -> dict:
    body = [{"x": head_x, "y": 2}, {"x": head_x, "y": 1}, {"x": head_x, "y": 0}]
    return {"id": snake_id, "name": name, "head": body[0], "body": body, "length": 3, "health": 100}


def test_target_states_preserve_target_identity_and_infer_actions() -> None:
    records = []
    for turn in range(4):
        target = _snake("target", "target-id", 2 + turn)
        probe = _snake("probe", "probe-id", 8)
        records.append(
            {
                "game": {"id": "runtime-id"},
                "turn": turn,
                "board": {"width": 11, "height": 11, "food": [], "hazards": [], "snakes": [probe, target]},
                "you": probe,
            }
        )
    records.append({"winnerName": "target", "isDraw": False})

    states, actions = runner.target_states_and_actions(records)

    assert actions == ["right", "right", "right"]
    assert all(state["you"]["id"] == "target-id" for state in states)
    assert all(state["you"]["name"] == "target" for state in states)


def test_candidate_actions_reseed_and_unwrap_move_dict(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "main.py").write_text(
        "import random\n"
        "def start(state): pass\n"
        "def move(state): return {'move': random.choice(['up', 'down'])}\n"
        "def end(state): pass\n",
        encoding="utf-8",
    )
    states = [{"turn": index} for index in range(5)]

    left = runner.candidate_actions(policy, states, 123)
    right = runner.candidate_actions(policy, states, 123)

    assert left == right
    assert len(left) == 5


def test_canonical_replay_states_remove_only_runtime_identity() -> None:
    target = _snake("target", "target-runtime-a", 2)
    probe = _snake("probe", "probe-runtime-a", 8)
    state = {
        "game": {"id": "game-runtime-a", "timeout": 500},
        "turn": 0,
        "board": {"food": [{"x": 2, "y": 1}], "hazards": [], "snakes": [target, probe]},
        "you": target,
    }
    changed = json.loads(json.dumps(state))
    changed["game"]["id"] = "game-runtime-b"
    changed["board"]["snakes"][0]["id"] = "target-runtime-b"
    changed["board"]["snakes"][1]["id"] = "probe-runtime-b"
    changed["you"]["id"] = "target-runtime-b"

    assert runner.canonical_replay_states([state]) == runner.canonical_replay_states([changed])
    assert state["you"]["id"] == "target-runtime-a"


def test_load_records_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    try:
        runner.load_records(path)
    except ValueError as error:
        assert "no records" in str(error)
    else:
        raise AssertionError("empty trajectory should fail")
