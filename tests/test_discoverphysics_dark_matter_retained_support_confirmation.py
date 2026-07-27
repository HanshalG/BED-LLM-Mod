from __future__ import annotations

import numpy as np

from scripts.discoverphysics_dark_matter_grounded_policy import (
    HIDDEN_MAP_SEEDS,
)
from scripts.discoverphysics_dark_matter_retained_support_confirmation import (
    CONFIRMATION_MAP_SEEDS,
    INITIAL_COMPONENT_MASS,
    REFRESH_COMPONENT_MASS,
    stratified_bootstrap_interval,
)


def test_confirmation_seeds_are_fresh_and_mixture_is_frozen():
    assert len(CONFIRMATION_MAP_SEEDS) == 16
    assert len(set(CONFIRMATION_MAP_SEEDS)) == 16
    assert set(CONFIRMATION_MAP_SEEDS).isdisjoint(HIDDEN_MAP_SEEDS)
    assert INITIAL_COMPONENT_MASS == 0.95
    assert REFRESH_COMPONENT_MASS == 0.05
    assert INITIAL_COMPONENT_MASS + REFRESH_COMPONENT_MASS == 1.0


def test_stratified_bootstrap_interval_preserves_constant_difference():
    regions = [
        region
        for region in ("NE", "NW", "SW", "SE")
        for _ in range(4)
    ]
    lower, upper = stratified_bootstrap_interval(
        np.full(len(regions), 0.25),
        regions,
    )

    assert np.isclose(lower, 0.25)
    assert np.isclose(upper, 0.25)
