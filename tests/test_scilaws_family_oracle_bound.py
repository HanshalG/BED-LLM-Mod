from itertools import product

import numpy as np
import pytest

from environments.scilaws.family_oracle_bound import family_oracle_bound
from environments.scilaws.regression_belief import RegressionBelief
from environments.scilaws.regression_mixture import RegressionMixture


def model():
    b = RegressionBelief([0.0], [[1.0]], 3.0, 2.0)
    return RegressionMixture(
        [[[1.0], [2.0]]],
        [[[1.0], [3.0]]],
        [b],
        [1.0],
        target_weights=[0.25, 0.75],
        include_observation_noise=True,
    )


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_known_family_analytic_solution(depth):
    m = model()
    result = family_oracle_bound(m, m.initial_state, depth)
    assert result["value"] == pytest.approx(1 + 7 / (1 + 4 * depth))
    assert result["component_sequences"] == [[1] * depth]
    assert not result["quadrature_pruning_authorized"]


def test_forced_first_action_and_conditioned_noise():
    m = model()
    state = m.condition(m.initial_state, 0, 3.0)
    result = family_oracle_bound(m, state, 3, first_action=0)
    assert result["value"] == pytest.approx(
        state.components[0].noise_variance * (1 + 7 / 11)
    )
    assert result["component_sequences"] == [[0, 1, 1]]


def test_multiset_enumeration_matches_ordered_independent_target_formula():
    a = RegressionBelief([0.0, 0.0], [[1.0, 0.2], [0.2, 2.0]], 3.0, 1.0)
    b = RegressionBelief([1.0], [[2.0]], 4.0, 2.0)
    m = RegressionMixture(
        [[[1.0, -1.0], [1.0, 1.0]], [[1.0], [2.0]]],
        [[[1.0, 0.3], [1.0, 0.7]], [[1.0], [3.0]]],
        [a, b],
        [0.4, 0.6],
        target_weights=[0.2, 0.8],
    )
    expected = []
    for component, actions, targets in zip(
        m.components, m.action_features, m.target_features
    ):
        values = []
        for sequence in product(range(2), repeat=3):
            precision = np.asarray(component.precision) + sum(
                np.outer(actions[i], actions[i]) for i in sequence
            )
            variances = component.noise_variance * np.diag(
                targets @ np.linalg.inv(precision) @ targets.T
            )
            values.append(variances @ m.target_weights)
        expected.append(min(values))
    result = family_oracle_bound(m, m.initial_state, 3)
    assert result["value"] == pytest.approx(np.dot([0.4, 0.6], expected))
    assert result["sequence_count_per_component"] == 4


def test_invalid_depth_fails():
    m = model()
    with pytest.raises(ValueError):
        family_oracle_bound(m, m.initial_state, 4)
    with pytest.raises(ValueError):
        family_oracle_bound(m, m.initial_state, 0, first_action=0)
