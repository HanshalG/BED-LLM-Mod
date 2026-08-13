from __future__ import annotations

import itertools
import math

import pytest

from scripts import revengebench_execution_opportunity_math as planner


DISTANCES = [
    [0.0, 0.8, 0.4],
    [0.6, 0.0, 0.7],
    [0.2, 0.5, 0.0],
]


def _brute_final_entropy(prior, first, second):
    total = 0.0
    for y1, y2 in itertools.product(range(len(first)), range(len(second))):
        joint = [prior[t] * first[y1][t] * second[y2][t] for t in range(len(prior))]
        predictive = math.fsum(joint)
        posterior = [value / predictive for value in joint]
        total += predictive * planner.entropy(posterior)
    return total


def test_distance_softmax_is_column_stochastic() -> None:
    likelihood = planner.likelihood_from_distances(DISTANCES, beta=1.0)

    assert [sum(row[column] for row in likelihood) for column in range(3)] == pytest.approx([1.0] * 3)
    assert likelihood[0][0] > likelihood[2][0] > likelihood[1][0]


def test_fixed_two_step_matches_full_observation_enumeration() -> None:
    prior = [1 / 3] * 3
    first = planner.likelihood_from_distances(DISTANCES, beta=0.5)
    second = planner.likelihood_from_distances(
        [[0.0, 0.2, 0.9], [0.5, 0.0, 0.1], [0.7, 0.8, 0.0]], beta=2.0
    )

    expected = planner.entropy(prior) - _brute_final_entropy(prior, first, second)

    assert planner.fixed_two_step_value(prior, first, second) == pytest.approx(expected, abs=1e-12)


def test_adaptive_value_selects_each_branch_independently() -> None:
    likelihoods = [
        planner.likelihood_from_distances(DISTANCES, beta=1.0),
        planner.likelihood_from_distances(
            [[0.0, 0.9, 0.2], [0.1, 0.0, 0.8], [0.7, 0.4, 0.0]], beta=1.0
        ),
        planner.likelihood_from_distances(
            [[0.0, 0.1, 0.9], [0.8, 0.0, 0.2], [0.4, 0.7, 0.0]], beta=1.0
        ),
    ]
    result = planner.adaptive_two_step_value([1 / 3] * 3, likelihoods, 0)

    manual_entropy = 0.0
    for y1 in range(3):
        predictive, posterior = planner.observation_update([1 / 3] * 3, likelihoods[0], y1)
        remaining = [1, 2]
        expected_entropies = [planner.expected_posterior_entropy(posterior, likelihoods[q]) for q in remaining]
        selected = remaining[min(range(2), key=lambda i: expected_entropies[i])]
        assert result["continuation_probes"][y1] == selected
        manual_entropy += predictive * min(expected_entropies)

    assert result["utility"] == pytest.approx(math.log(3) - manual_entropy, abs=1e-12)


def test_policy_report_is_compute_matched_and_tie_breaks_by_manifest_order() -> None:
    likelihood = planner.likelihood_from_distances(DISTANCES, beta=1.0)
    result = planner.evaluate_policies([likelihood, likelihood, likelihood])

    assert result["depth_two"]["first_probe"] == 0
    assert result["receding_myopic"]["first_probe"] == 0
    assert result["fixed"]["probes"] == [0, 1]
    assert not result["changed_first_action"]
    assert result["depth_two_margin"] == pytest.approx(0.0)
    assert result["random"]["utility"] == pytest.approx(result["fixed"]["utility"])


@pytest.mark.parametrize(
    "distances,beta",
    [([[0.0]], 1.0), (DISTANCES, 0.0), ([[0.0, 2.0], [0.0, 0.0]], 1.0)],
)
def test_invalid_distance_models_fail_closed(distances, beta) -> None:
    with pytest.raises(ValueError):
        planner.likelihood_from_distances(distances, beta)
