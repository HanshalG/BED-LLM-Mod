import math

import pytest

from environments.mushroom_feature_acquisition import (
    COLLECT_ACTION,
    FEATURE_NAMES,
    FIELD_FEATURES,
    MushroomFeatureModel,
)
from scripts.nonmyopic_mushroom_feature_oracle import exact_action_costs


def test_mushroom_catalog_and_initial_actions_are_frozen() -> None:
    model = MushroomFeatureModel()

    assert len(model.rows) == 8124
    assert len(FEATURE_NAMES) == 22
    assert sum(model.classes == "e") == 4208
    assert sum(model.classes == "p") == 3916
    assert model.legal_actions(model.initial_state) == (
        COLLECT_ACTION,
        "query:cap-shape",
        "query:cap-surface",
        "query:cap-color",
        "query:population",
        "query:habitat",
    )


def test_collection_has_zero_eig_and_unlocks_specimen_features() -> None:
    model = MushroomFeatureModel()
    state = model.initial_state
    belief = model.initial_belief

    assert model.expected_information_gain(belief, COLLECT_ACTION) == pytest.approx(0.0)
    collected = model.next_state(state, COLLECT_ACTION)

    assert collected.specimen_collected
    assert "query:odor" in model.legal_actions(collected)
    assert all(
        f"query:{feature}" in model.legal_actions(collected)
        for feature in FEATURE_NAMES
        if feature not in FIELD_FEATURES
    )


def test_query_observation_conditions_exact_catalog_posterior() -> None:
    model = MushroomFeatureModel()
    action = "query:odor"
    outcome = model.observation(0, action)
    posterior = model.posterior(model.initial_belief, action, outcome)

    assert posterior.sum() == pytest.approx(1.0)
    assert all(
        probability == 0.0 or model.observation(index, action) == outcome
        for index, probability in enumerate(posterior)
    )
    assert model.target_entropy(posterior) <= model.target_entropy(model.initial_belief)
    assert math.isfinite(model.truth_log_probability(posterior, 0))


def test_aligned_depth_two_values_collection_but_depth_one_does_not() -> None:
    model = MushroomFeatureModel()

    d1, _ = exact_action_costs(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        depth=1,
    )
    d2, _ = exact_action_costs(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        depth=2,
    )

    assert d1[COLLECT_ACTION] > min(
        value for action, value in d1.items() if action != COLLECT_ACTION
    )
    assert d2[COLLECT_ACTION] < min(
        value for action, value in d2.items() if action != COLLECT_ACTION
    )
