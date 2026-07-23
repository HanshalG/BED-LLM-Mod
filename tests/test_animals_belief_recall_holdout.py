import pytest

from scripts.animals_belief_recall_holdout import (
    bootstrap_mean_gain,
    evaluate_gates,
)


def _summary(*, active=20, wins=10, losses=2, ranker_regret=0.1, eig_regret=0.2):
    return {
        "num_states": 60,
        "num_active_states": active,
        "ranker_immediate_wins_ties_losses": [wins, 60 - wins - losses, losses],
        "mean_active_state_regret_belief_recall": ranker_regret,
        "mean_active_state_regret_immediate_eig": eig_regret,
    }


def test_bootstrap_mean_gain_is_deterministic():
    first = bootstrap_mean_gain([0.2, 0.1, 0.0], seed=17, replicates=200)
    second = bootstrap_mean_gain([0.2, 0.1, 0.0], seed=17, replicates=200)

    assert first == second
    assert first["mean"] == pytest.approx(0.1)


def test_holdout_gates_require_positive_interval_and_mechanism():
    passing = evaluate_gates(_summary(), {"ci95": [0.01, 0.2]})
    assert passing["all_pass"]

    failed_interval = evaluate_gates(_summary(), {"ci95": [-0.01, 0.2]})
    assert not failed_interval["paired_bootstrap_lower_bound_positive"]
    assert not failed_interval["all_pass"]

    failed_mechanism = evaluate_gates(
        _summary(active=14),
        {"ci95": [0.01, 0.2]},
    )
    assert not failed_mechanism["at_least_fifteen_active_states"]
    assert not failed_mechanism["all_pass"]
