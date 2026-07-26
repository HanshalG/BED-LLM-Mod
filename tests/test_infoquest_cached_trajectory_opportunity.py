from __future__ import annotations

import copy
import json

import pytest

from scripts import infoquest_cached_trajectory_opportunity as audit


def _evaluation(reward: int, *, done: bool = False) -> dict:
    return {
        "done": done,
        "generation_time": 1.0,
        "invalid_responses": 0,
        "questions": {"criterion": {"response": "result"}},
        "total_reward": reward,
    }


def _row(record_id: int) -> dict:
    history = [
        {"role": "system", "content": "hidden context"},
        {"role": "user", "content": "Could you explain the options?"},
        {"role": "assistant", "content": "My budget is limited."},
        {"role": "user", "content": "Which limited budget should we use?"},
        {"role": "assistant", "content": "The schedule changes weekly."},
        {"role": "user", "content": "How does the weekly schedule change?"},
    ]
    evaluations = [
        _evaluation(0),
        _evaluation(2),
        _evaluation(5, done=True),
    ]
    return {
        "id": record_id,
        "user_history1": copy.deepcopy(history),
        "evaluations1": copy.deepcopy(evaluations),
        "generation_time1": 1.0,
        "user_history2": copy.deepcopy(history),
        "evaluations2": copy.deepcopy(evaluations),
        "generation_time2": 1.0,
    }


def test_rows_by_id_uses_explicit_ids_not_row_positions(tmp_path):
    path = tmp_path / "unordered.jsonl"
    rows = [_row(2), _row(0), _row(1)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = audit.source_manifest.sha256_file(path)
    by_id = audit._rows_by_id(
        path,
        expected_sha256=digest,
        expected_records=3,
    )
    assert list(by_id) == [2, 0, 1]
    assert by_id[0]["id"] == 0
    assert by_id[1]["id"] == 1
    assert by_id[2]["id"] == 2


def test_trajectory_metrics_capture_delayed_gain_and_answer_uptake():
    result = audit.trajectory_metrics(
        _row(0),
        seed_message="I need some advice.",
        run=0,
        record_id=0,
        world=1,
    )
    assert result["reward_trace"] == [0, 2, 5]
    assert result["turns"] == 3
    assert result["delayed_gain"] == 5
    assert result["transition_count"] == 2
    assert result["immediate_novel_token_uptake"] > 0
    assert "hidden context" not in json.dumps(result)
    assert "budget" not in json.dumps(result)


def test_trajectory_validation_rejects_nonmonotone_reward():
    row = _row(0)
    row["evaluations1"][1]["total_reward"] = 3
    row["evaluations1"][2]["total_reward"] = 2
    row["evaluations1"][2]["done"] = False
    with pytest.raises(ValueError, match="not monotone"):
        audit.trajectory_metrics(
            row,
            seed_message="request",
            run=0,
            record_id=0,
            world=1,
        )


def test_summary_requires_three_runs_and_reports_frozen_gates():
    trajectories = []
    for run in range(3):
        for world in (1, 2):
            trajectories.append(
                audit.trajectory_metrics(
                    _row(0),
                    seed_message="request",
                    run=run,
                    record_id=0,
                    world=world,
                )
            )
    metrics, gates = audit.summarize_metrics(trajectories)
    assert metrics["trajectories"] == 6
    assert metrics["mean_delayed_gain"] == 5
    assert gates["multi_turn_fraction"] is True
    assert gates["delayed_gain_at_least_two_fraction"] is True
    assert set(gates) == set(audit.THRESHOLDS)
