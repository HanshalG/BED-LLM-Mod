import math

import numpy as np
import pytest

from environments.gated_sensor import GatedSensorModel, SensorState
from scripts.nonmyopic_gated_sensor_oracle import OracleConfig, exact_action_values, run_oracle


def test_activation_is_zero_information_and_unlocks_precise_tests() -> None:
    model = GatedSensorModel()
    belief = model.initial_belief

    assert model.expected_information_gain(belief, "activate:A") == pytest.approx(0.0)
    assert not any(action.startswith("precise:") for action in model.legal_actions(model.initial_state))

    active = model.next_state(model.initial_state, "activate:A")
    assert active == SensorState("A")
    assert "precise:bit-0" in model.legal_actions(active)
    assert model.expected_information_gain(belief, "precise:bit-0") > model.expected_information_gain(
        belief, "screen:bit-0"
    )


def test_exact_posterior_and_likelihood_are_normalized() -> None:
    model = GatedSensorModel()
    belief = model.initial_belief
    posterior = model.posterior(belief, "screen:bit-0", "positive")

    assert float(posterior.sum()) == pytest.approx(1.0)
    assert model.outcome_probability(belief, "screen:bit-0", "positive") == pytest.approx(0.5)
    bit_one = np.asarray([hidden[0] == 1 for hidden in model.hidden_states])
    assert float(posterior[bit_one].sum()) == pytest.approx(model.screen_accuracy)


def test_depth_two_prefers_setup_while_depth_one_prefers_measurement() -> None:
    model = GatedSensorModel()
    d1, _ = exact_action_values(model, state=model.initial_state, belief=model.initial_belief, depth=1)
    d2, _ = exact_action_values(model, state=model.initial_state, belief=model.initial_belief, depth=2)
    d1_action = max(d1, key=d1.get)
    d2_action = max(d2, key=d2.get)

    assert d1_action.startswith("screen:")
    assert d2_action.startswith("activate:")
    assert d2[d2_action] > d2[d1_action]


def test_small_oracle_is_paired_finite_llm_free_and_passes_mechanics() -> None:
    summary = run_oracle(OracleConfig(num_trials=12, num_rounds=4, bootstrap_replicates=100))

    assert summary["no_llm_calls"]
    assert all(summary["mechanics"].values())
    assert math.isfinite(summary["comparison"]["entropy_auc_gain_mean"])
    assert summary["comparison"]["entropy_auc_gain_mean"] > 0.0
    assert summary["gate"]["passed"]
