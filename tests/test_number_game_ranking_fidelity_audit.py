import pytest

from scripts.number_game_ranking_fidelity_audit import (
    bootstrap_mean_interval,
    pairwise_concordance,
    spearman_correlation,
)


def test_pairwise_concordance_detects_order_and_reversal():
    assert pairwise_concordance([1, 2, 3], [10, 20, 30]) == 1.0
    assert pairwise_concordance([1, 2, 3], [30, 20, 10]) == 0.0


def test_pairwise_concordance_ignores_ties():
    assert pairwise_concordance([1, 1, 2], [3, 4, 5]) == 1.0


def test_spearman_correlation_handles_order_reversal_and_ties():
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == -1.0
    assert spearman_correlation([1, 1, 2], [3, 3, 4]) == 1.0


def test_bootstrap_mean_interval_is_reproducible():
    first = bootstrap_mean_interval(
        [0.1, 0.2, 0.3],
        seed=10,
        samples=200,
    )
    second = bootstrap_mean_interval(
        [0.1, 0.2, 0.3],
        seed=10,
        samples=200,
    )

    assert first == second
    assert first[0] == pytest.approx(0.1)
    assert first[1] <= 0.3
