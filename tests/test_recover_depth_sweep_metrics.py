import json

import numpy as np
import pytest

from core import BeliefState
from helpers import Config
from scripts.recover_depth_sweep_metrics import recover_depth_sweep_metrics


def test_recover_depth_sweep_metrics_replays_decisions_without_strategy_generation(tmp_path, monkeypatch):
    from environments.location_finding.env import LocationBEDEnvironment

    def fake_hidden_state(self, trial_index, rng):
        return np.asarray([[0.0, 0.0]], dtype=float)

    def fake_initial_belief(self, model, config):
        return BeliefState(
            hypotheses=[((0.0, 0.0),), ((1.0, 0.0),)],
            probabilities=[0.6, 0.4],
        )

    def fake_update_belief_states(self, belief_states, histories, model, config):
        return list(belief_states)

    monkeypatch.setattr(LocationBEDEnvironment, "sample_hidden_state_for_trial", fake_hidden_state)
    monkeypatch.setattr(LocationBEDEnvironment, "initial_belief_state", fake_initial_belief)
    monkeypatch.setattr(LocationBEDEnvironment, "update_belief_states", fake_update_belief_states)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    decisions_path = run_dir / "fixed_root_depth_sweep_decisions.jsonl"
    records = [
        {
            "trial_index": 0,
            "policy_label": "EIG",
            "policy_kind": "EIG",
            "policy_depth": None,
            "selection_depth": None,
            "round_index": 0,
            "selected_eig": 0.25,
            "selected_root_query": [0.0, 0.0],
            "observation": {"query": [0.0, 0.0], "value": 5.0},
            "evaluations_by_depth": {},
        },
        {
            "trial_index": 0,
            "policy_label": "naive",
            "policy_kind": "naive",
            "policy_depth": None,
            "selection_depth": None,
            "round_index": 0,
            "selected_eig": 0.0,
            "selected_root_query": [1.0, 0.0],
            "observation": {"query": [1.0, 0.0], "value": 1.0},
            "evaluations_by_depth": {},
        },
    ]
    decisions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text("", encoding="utf-8")
    output_path = run_dir / "fixed_root_depth_sweep_metrics_recovered.json"

    config = Config(
        task="location_finding",
        location_num_trials=1,
        location_num_rounds=1,
        location_seed=1304,
        location_num_sources=1,
    )
    summary = recover_depth_sweep_metrics(
        config=config,
        config_path=tmp_path / "config.yaml",
        decisions_path=decisions_path,
        run_dir=run_dir,
        output_path=output_path,
        questioner=object(),
        max_depth=1,
    )

    assert output_path.exists()
    assert summary["recovered"] is True
    assert summary["recovery_warnings"]
    assert summary["aggregate"]["EIG"]["selected_eig"]["final_mean"] == pytest.approx(0.25)
    assert summary["aggregate"]["EIG"]["source_rmse"]["final_mean"] == pytest.approx(0.0)
    assert summary["per_trial"][2]["policy_label"] == "EIG"
    assert summary["per_trial"][2]["history"][0]["observation"]["value"] == pytest.approx(5.0)
