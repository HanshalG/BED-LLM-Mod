import copy
import json
from time import monotonic

import numpy as np
import pytest

from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.pilot_data import read_protocol
from scripts import chembench_horizon_pilot as pilot
from scripts import chembench_horizon_pilot_verify as verifier


def fixture():
    model = EnvelopeGaussianModel(
        [[-0.2, -0.3, -0.4, -0.5], [0.3, 0.4, 0.5, 0.6]],
        1,
        [[0, 0.2], [0.5, 0.7]],
        [0.4, 0.6],
        branch_count=4,
    )
    config, _ = read_protocol()
    observations, targets, noise = np.full(4, 0.1), np.full(2, 0.2), np.zeros((3, 4))
    roots = {
        "h1": pilot.decide(
            model, model.initial_state, tuple(range(4)), 3, "h1", config, 60
        )
    }
    record = pilot.episode(
        model, 0, observations, targets, noise, "h1", config, roots, monotonic() + 60
    )
    return record, model, observations, targets, noise, config, roots


def test_independent_physics_and_policy_replay():
    values = fixture()
    verifier.replay_episode(*values, 0, "h1")


@pytest.mark.parametrize(
    "field",
    [
        "observation",
        "forecast",
        "posterior_log_weights",
        "target_mse",
        "model_risk_after",
    ],
)
def test_corrupted_physics_is_rejected(field):
    values = list(fixture())
    record = copy.deepcopy(values[0])
    row = record["history"][0]
    row[field] = np.asarray(row[field]) + 0.1
    values[0] = record
    with pytest.raises(ValueError, match="replay mismatch"):
        verifier.replay_episode(*values, 0, "h1")


def test_incomplete_result_does_not_construct_hidden_worlds(monkeypatch, tmp_path):
    def bomb(*args, **kwargs):
        pytest.fail("incomplete result crossed the source/hidden boundary")

    monkeypatch.setattr(verifier, "load_source", bomb)
    monkeypatch.setattr(verifier, "build_hidden_worlds", bomb)
    (tmp_path / "RESULT.json").write_text(
        json.dumps({"status": "execution_failed", "records": []})
    )
    with pytest.raises(ValueError, match="complete eight-world"):
        verifier.verify(tmp_path, tmp_path)
