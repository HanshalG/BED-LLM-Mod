from __future__ import annotations

import numpy as np
import pytest

from scripts.discoverphysics_dark_matter_asymmetric_opportunity import (
    REGION_PRIOR,
    hypothesis_prior,
)
from scripts.discoverphysics_dark_matter_opportunity import evaluate_roots


def test_asymmetric_hypothesis_prior_preserves_region_masses():
    prior = hypothesis_prior()

    assert prior.shape == (24,)
    assert np.all(prior > 0.0)
    assert np.isclose(prior.sum(), 1.0)
    assert np.allclose(prior.reshape(4, 6).sum(axis=1), REGION_PRIOR)


def test_root_evaluator_rejects_invalid_prior():
    observation_means = np.zeros((2, 2, 1))
    heldout_features = np.zeros((2, 1))

    with pytest.raises(ValueError, match="sum to one"):
        evaluate_roots(
            observation_means,
            heldout_features,
            noise_seed=1,
            prior=np.array([0.2, 0.2]),
            root_samples_per_hypothesis=1,
            continuation_samples_per_hypothesis=1,
        )
