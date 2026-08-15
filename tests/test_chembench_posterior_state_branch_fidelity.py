from __future__ import annotations

import numpy as np

from scripts.chembench_posterior_state_branch_fidelity import (
    deterministic_belief_medoids,
    evaluate_gates,
    posterior_moments_for_observations,
    posterior_state_branch_expected_risk,
)


def test_posterior_moments_match_direct_risk() -> None:
    outcomes = np.asarray([-0.2, 0.7])
    means = np.asarray([0.0, 1.0])
    sigmas = np.asarray([0.5, 0.5])
    targets = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    prior = np.asarray([0.25, 0.75])
    risks, child_means, child_variances = posterior_moments_for_observations(
        outcomes, means, sigmas, targets, prior, batch_size=1
    )
    assert np.allclose(risks, np.mean(child_variances, axis=1))
    assert child_means.shape == child_variances.shape == (2, 2)


def test_deterministic_medoids_cover_separated_groups() -> None:
    features = np.asarray(
        [[-4.0, 0.0], [-3.9, 0.1], [0.0, 4.0], [0.1, 3.9], [4.0, 0.0], [3.9, -0.1]]
    )
    first = deterministic_belief_medoids(features, 3)
    second = deterministic_belief_medoids(features, 3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    labels, medoids, probabilities = first
    assert len(set(labels.tolist())) == 3
    assert len(set(medoids.tolist())) == 3
    assert np.allclose(np.sort(probabilities), [1 / 3, 1 / 3, 1 / 3])


def test_deterministic_medoids_retain_zero_mass_duplicate_centers() -> None:
    labels, medoids, probabilities = deterministic_belief_medoids(
        np.zeros((12, 2)), 3
    )
    assert labels.tolist() == [0] * 12
    assert medoids.tolist() == [0, 0, 0]
    assert probabilities.tolist() == [1.0, 0.0, 0.0]


def test_posterior_state_branches_are_finite_and_normalized() -> None:
    result = posterior_state_branch_expected_risk(
        outcomes=np.linspace(-1.0, 2.0, 27),
        predictive_means=np.asarray([0.0, 1.0]),
        predictive_sigmas=np.asarray([0.4, 0.4]),
        target_values=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
        prior_weights=np.asarray([0.5, 0.5]),
    )
    assert result["reference_expected_risk"] >= 0
    assert result["posterior_state_expected_risk"] >= 0
    assert len(result["medoid_indices"]) == 9
    assert np.isclose(sum(result["branch_probabilities"]), 1.0)


def _case(index: int) -> dict:
    return {
        "finite_and_reproducible": True,
        "posterior_state": {
            "spearman": 0.95,
            "normalized_top_one_regret": 0.005,
        },
        "raw_quantile": {
            "spearman": 0.90,
            "normalized_top_one_regret": 0.01,
        },
        "component_bank_regret": {"bank_1": 0.005, "bank_2": 0.006},
    }


def test_posterior_state_gate_passes_exact_panel() -> None:
    gates = evaluate_gates([_case(index) for index in range(36)])
    assert gates["pass"]


def test_posterior_state_gate_fails_component_regret() -> None:
    cases = [_case(index) for index in range(36)]
    for case in cases[:5]:
        case["component_bank_regret"]["bank_2"] = 0.1
    gates = evaluate_gates(cases)
    assert not gates["pass"]
    assert not gates["conditions"]["component_bank_regret_passes"]
