import numpy as np

from environments.heart_workup import (
    INITIAL_FEATURES,
    ORDER_WORKUP_ACTION,
    HeartWorkupModel,
)


def test_heart_workup_loads_complete_cleveland_cohort() -> None:
    model = HeartWorkupModel()

    assert len(model.rows) == 297
    assert int(np.sum(model.classes == 0)) == 160
    assert int(np.sum(model.classes == 1)) == 137
    assert np.isclose(model.initial_belief.sum(), 1.0)


def test_workup_is_zero_information_and_unlocks_compact_tests() -> None:
    model = HeartWorkupModel()
    state = model.initial_state

    assert ORDER_WORKUP_ACTION in model.legal_actions(state)
    assert all(
        model.action_feature(action) in INITIAL_FEATURES
        for action in model.legal_actions(state)
        if action.startswith("query:")
    )
    assert abs(model.expected_information_gain(model.initial_belief, ORDER_WORKUP_ACTION)) <= 1e-12

    next_state = model.next_state(state, ORDER_WORKUP_ACTION)
    assert next_state.workup_ordered is True
    assert ORDER_WORKUP_ACTION not in model.legal_actions(next_state)
    assert "query:major-vessels" in model.legal_actions(next_state)
    assert "query:thal" in model.legal_actions(next_state)


def test_query_conditioning_is_exact_and_nonrepeatable() -> None:
    model = HeartWorkupModel()
    action = "query:chest-pain"
    outcome = model.observation(0, action)
    posterior = model.posterior(model.initial_belief, action, outcome)
    state = model.next_state(model.initial_state, action)

    assert posterior.sum() == 1.0
    assert np.all(model.feature_values[posterior > 0, 2] == outcome)
    assert action not in model.legal_actions(state)
