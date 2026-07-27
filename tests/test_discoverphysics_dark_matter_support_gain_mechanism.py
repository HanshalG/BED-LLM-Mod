from __future__ import annotations

import numpy as np

from scripts.discoverphysics_dark_matter_support_gain_mechanism import (
    spearman_correlation,
)


def test_spearman_correlation_handles_ties():
    left = np.array([0.0, 0.0, 1.0, 2.0])
    right = np.array([0.0, 0.0, 2.0, 1.0])

    value = spearman_correlation(left, right)

    assert np.isfinite(value)
    assert 0.0 < value < 1.0


def test_spearman_correlation_returns_zero_for_constant_input():
    assert spearman_correlation(
        np.ones(4),
        np.arange(4, dtype=float),
    ) == 0.0
