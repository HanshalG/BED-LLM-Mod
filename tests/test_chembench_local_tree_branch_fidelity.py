from __future__ import annotations

import numpy as np

from scripts.chembench_local_tree_branch_fidelity import (
    evaluate_gates,
    expected_action_risks,
    posterior_risks_for_observations,
    quantile_branch_observations,
    summarize_bank_cases,
)


def test_posterior_risk_matches_manual_reweighting() -> None:
    outcomes = np.asarray([-0.2, 0.7])
    means = np.asarray([0.0, 1.0])
    sigmas = np.asarray([0.5, 0.5])
    targets = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    prior = np.asarray([0.25, 0.75])

    actual = posterior_risks_for_observations(
        outcomes, means, sigmas, targets, prior, batch_size=1
    )
    expected = []
    for outcome in outcomes:
        likelihood = np.exp(-0.5 * np.square((outcome - means) / sigmas)) / sigmas
        weights = prior * likelihood
        weights /= weights.sum()
        first = weights @ targets
        second = weights @ np.square(targets)
        expected.append(np.mean(second - np.square(first)))
    assert np.allclose(actual, expected)


def test_quantile_branches_use_equal_mass_bin_medians() -> None:
    representatives, probabilities = quantile_branch_observations(
        np.asarray([8.0, 0.0, 7.0, 2.0, 3.0, 6.0, 5.0, 1.0, 4.0]), 3
    )
    assert representatives.tolist() == [1.0, 4.0, 7.0]
    assert np.allclose(probabilities, 1.0 / 3.0)


def test_expected_action_risk_reuses_reference_outcomes() -> None:
    result = expected_action_risks(
        predictive_means=np.asarray([0.0, 1.0]),
        predictive_sigmas=np.asarray([0.4, 0.4]),
        target_values=np.asarray([[0.0], [2.0]]),
        prior_weights=np.asarray([0.5, 0.5]),
        reference_outcomes=np.linspace(-0.5, 1.5, 12),
        branch_counts=(3,),
    )
    assert result["reference_expected_risk"] >= 0
    approximation = result["approximations"]["3"]
    assert len(approximation["representatives"]) == 3
    assert np.isclose(sum(approximation["probabilities"]), 1.0)


def _case(bank: str, index: int, *, rho9: float = 0.95, regret9: float = 0.005):
    branches = {
        "3": {"spearman": 0.8, "normalized_top_one_regret": 0.02},
        "5": {"spearman": 0.9, "normalized_top_one_regret": 0.01},
        "9": {
            "spearman": rho9,
            "normalized_top_one_regret": regret9,
            "selected_action_index": index % 3,
        },
    }
    return {
        "bank": bank,
        "difficulty": ("easy", "medium", "hard")[index // 12],
        "domain": f"domain-{index % 12}",
        "finite_and_reproducible": True,
        "branches": branches,
    }


def test_gate_summary_passes_exact_72_case_panel() -> None:
    cases = [_case(bank, index) for bank in ("bank_1", "bank_2") for index in range(36)]
    gates = evaluate_gates(cases)
    assert gates["pass"]
    assert gates["bank_selected_action_agreement"] == 1.0
    assert summarize_bank_cases(cases[:36], 9)["median_spearman"] == 0.95


def test_gate_summary_fails_regret_threshold() -> None:
    cases = [_case(bank, index) for bank in ("bank_1", "bank_2") for index in range(36)]
    cases[0]["branches"]["9"]["normalized_top_one_regret"] = 0.5
    gates = evaluate_gates(cases)
    assert not gates["pass"]
    assert not gates["conditions"]["mean_normalized_regret_at_most_001_both_banks"]
