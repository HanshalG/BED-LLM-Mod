from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from scipy.special import ndtr

from environments.chembench_mopen.executable_belief import ExecutableBeliefPool
from environments.chembench_mopen.proposal_prediction import (
    ARMS,
    score_sealed_panel,
    seal_panel,
)
from environments.chembench_mopen.raw_belief import GaussianParticleModel


TARGETS = [[1, 0, 1, 0, 1, 310, 7], [2, 0, 1, 0, 1, 310, 7]]


def snapshot():
    pool = ExecutableBeliefPool(particles_per_law=2)
    pool.add(
        {
            "name": "linear",
            "expr": "k * C_A",
            "params": [{"name": "k", "low": 0.1, "high": 4}],
        }
    )
    result = pool.snapshot(
        history_inputs=[], observations=[], designs=TARGETS, targets=TARGETS, sigma=0.2
    )
    model = GaussianParticleModel(
        np.ones((2, 2)), 0.2, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.25, 0.75])
    )
    return replace(result, model=model, state=tuple(np.log([0.25, 0.75])))


def panel():
    result = snapshot()
    return {"case0": {arm: result for arm in ARMS}}


def outcomes():
    return {
        "case0": {
            "target_inputs": TARGETS,
            "true_log_rates": [2.0, 3.0],
            "noisy_log_rates": [2.1, 3.2],
        }
    }


def forbidden():
    raise AssertionError("test outcomes must not open")


def test_correct_matched_mse_mixture_density_and_pit(tmp_path):
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, panel(), sigma=0.2)
    accesses = []

    def load():
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        accesses.append("opened")
        return outcomes()

    result = score_sealed_panel(path, sha, load)
    assert accesses == ["opened"]
    assert result["status"] == "heldout_scores_complete"
    for arm in ARMS:
        scores = result["rows"][0]["scores"][arm]
        assert scores["mean_squared_error"] == pytest.approx(0.25)
        np.testing.assert_allclose(scores["per_target_prediction"], [2.5, 3.5])
        for i, y in enumerate([2.1, 3.2]):
            z = (y - np.array([1 + i, 3 + i])) / 0.2
            density = (
                np.array([0.25, 0.75])
                @ np.exp(-(z**2) / 2)
                / (0.2 * math.sqrt(2 * math.pi))
            )
            assert scores["per_target_negative_log_density"][i] == pytest.approx(
                -math.log(density)
            )
            assert scores["per_target_pit"][i] == pytest.approx(
                np.array([0.25, 0.75]) @ ndtr(z)
            )
    assert result["mean_paired_mse_differences"] == {arm: 0 for arm in ARMS[1:]}
    assert result["scientific_pass_authorized"] is False
    assert result["paid_calls_authorized"] is False


def test_forecast_copy_is_not_changed_by_later_model_mutation(tmp_path):
    cases = panel()
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, cases, sigma=0.2)
    cases["case0"][ARMS[0]].model.targets = np.zeros((2, 2))
    result = score_sealed_panel(path, sha, outcomes)
    assert result["rows"][0]["scores"][ARMS[0]]["mean_squared_error"] == pytest.approx(
        0.25
    )


def test_seal_is_write_once(tmp_path):
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, panel(), sigma=0.2)
    with pytest.raises(FileExistsError):
        seal_panel(path, panel(), sigma=0.2)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.mark.parametrize(
    "defect",
    ["missing_arm", "failed_arm", "history", "targets", "noise", "log_weights"],
)
def test_incomplete_or_unmatched_controls_never_seal(tmp_path, defect):
    cases = panel()
    original = cases["case0"][ARMS[1]]
    sigma = 0.2
    if defect == "missing_arm":
        cases["case0"].pop(ARMS[1])
    elif defect == "failed_arm":
        cases["case0"][ARMS[1]] = None
    elif defect == "history":
        cases["case0"][ARMS[1]] = replace(original, history_sha256="f" * 64)
    elif defect == "targets":
        cases["case0"][ARMS[1]] = replace(
            original, target_inputs=tuple(reversed(original.target_inputs))
        )
    elif defect == "noise":
        sigma = 0.3
    else:
        cases["case0"][ARMS[1]] = replace(original, state=(0.0, 0.0))
    path = tmp_path / "forecasts.json"
    with pytest.raises(ValueError):
        seal_panel(path, cases, sigma=sigma)
    assert not path.exists()


def test_changed_seal_never_opens_outcomes(tmp_path):
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, panel(), sigma=0.2)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        score_sealed_panel(path, sha, forbidden)


def test_duplicate_json_keys_rejected_before_outcomes(tmp_path):
    path = tmp_path / "forecasts.json"
    seal_panel(path, panel(), sigma=0.2)
    path.write_text(
        path.read_text().replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate JSON"):
        score_sealed_panel(path, sha, forbidden)


@pytest.mark.parametrize(
    "defect", ["version", "duplicate", "missing_arm", "nan", "bad_shape", "extra_key"]
)
def test_replay_validates_complete_panel_before_outcomes(tmp_path, defect):
    path = tmp_path / "forecasts.json"
    seal_panel(path, panel(), sigma=0.2)
    data = json.loads(path.read_text())
    if defect == "version":
        data["schema_version"] = 2
    elif defect == "duplicate":
        data["cases"].append(data["cases"][0])
    elif defect == "missing_arm":
        data["cases"][0]["forecasts"].pop(ARMS[2])
    elif defect == "nan":
        data["cases"][0]["forecasts"][ARMS[0]]["log_weights"][0] = float("nan")
    elif defect == "bad_shape":
        data["cases"][0]["forecasts"][ARMS[0]]["target_means"] = [[1]]
    else:
        data["unexpected"] = True
    path.write_text(json.dumps(data))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        score_sealed_panel(path, sha, forbidden)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_case",
        "extra_case",
        "shape",
        "nan",
        "reordered_targets",
        "negative_truth",
    ],
)
def test_outcome_mismatch_does_not_produce_partial_scores(tmp_path, defect):
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, panel(), sigma=0.2)
    data = outcomes()
    if defect == "missing_case":
        data.clear()
    elif defect == "extra_case":
        data["new"] = data["case0"]
    elif defect == "shape":
        data["case0"]["true_log_rates"] = [1]
    elif defect == "nan":
        data["case0"]["noisy_log_rates"][0] = float("nan")
    elif defect == "reordered_targets":
        data["case0"]["target_inputs"] = list(reversed(TARGETS))
    else:
        data["case0"]["true_log_rates"][0] = -1
    with pytest.raises(ValueError):
        score_sealed_panel(path, sha, lambda: data)


def test_overconfident_wrong_predictor_is_penalized_not_rewarded(tmp_path):
    cases = panel()
    old = cases["case0"][ARMS[0]]
    wrong = GaussianParticleModel(
        np.ones((2, 2)), 0.2, np.full((2, 2), 10.0), np.array([0.25, 0.75])
    )
    cases["case0"][ARMS[0]] = replace(old, model=wrong)
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, cases, sigma=0.2)
    result = score_sealed_panel(path, sha, outcomes)
    scores = result["rows"][0]["scores"]
    assert wrong.risk(old.state) == pytest.approx(0)
    assert scores[ARMS[0]]["mean_squared_error"] > scores[ARMS[1]]["mean_squared_error"]
    assert (
        scores[ARMS[0]]["mean_negative_log_density"]
        > scores[ARMS[1]]["mean_negative_log_density"]
    )
    assert result["mean_paired_mse_differences"][ARMS[1]] > 0


def test_extreme_finite_noisy_outcome_has_stable_log_density(tmp_path):
    path = tmp_path / "forecasts.json"
    sha = seal_panel(path, panel(), sigma=0.2)
    data = outcomes()
    data["case0"]["noisy_log_rates"] = [100, -100]
    result = score_sealed_panel(path, sha, lambda: data)
    assert math.isfinite(
        result["rows"][0]["scores"][ARMS[0]]["mean_negative_log_density"]
    )
