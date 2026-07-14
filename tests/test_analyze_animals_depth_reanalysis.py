from __future__ import annotations

from scripts.analyze_animals_depth_reanalysis import (
    bootstrap_arm,
    bootstrap_comparison,
    mean_curve,
    q_at_threshold,
    recover_trial_accuracy,
)


def test_recover_trial_accuracy_inverts_cumulative_means() -> None:
    trials = (
        (0, 1, 0, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 1),
    )
    cumulative = [mean_curve(trials[:index]) for index in range(1, len(trials) + 1)]
    assert recover_trial_accuracy(cumulative) == trials


def test_q80_uses_tolerance_and_censors_non_crossing_curves() -> None:
    assert q_at_threshold([0.2, 0.7999999999999999, 0.7]) == 2
    assert q_at_threshold([0.2, 0.79, 0.7]) == 4


def test_bootstrap_arm_is_deterministic() -> None:
    trials = (
        (0, 1, 1, 1),
        (1, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 1, 1, 1),
    )
    first = bootstrap_arm(trials, replicates=200, seed=1304)
    second = bootstrap_arm(trials, replicates=200, seed=1304)
    assert first == second
    assert first["accuracy_auc"] == 0.75
    assert first["q80"] == 4


def test_paired_bootstrap_uses_trial_pairs_and_reports_wins() -> None:
    baseline = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
    )
    better = (
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 1, 0),
        (1, 1, 0, 0),
    )
    result = bootstrap_comparison(
        better,
        baseline,
        paired=True,
        replicates=500,
        seed=1304,
        label="test",
    )
    assert result["accuracy_auc_delta"] == 0.1875
    assert (result["wins"], result["ties"], result["losses"]) == (3, 1, 0)
    assert result == bootstrap_comparison(
        better,
        baseline,
        paired=True,
        replicates=500,
        seed=1304,
        label="test",
    )


def test_recovery_rejects_nonbinary_inversion() -> None:
    try:
        recover_trial_accuracy(((0.0, 1.0), (0.25, 0.5)))
    except ValueError as error:
        assert "binary outcome" in str(error)
    else:
        raise AssertionError("Expected malformed cumulative traces to be rejected")
