from __future__ import annotations

from itertools import permutations
import math

import numpy as np

from core import BeliefState
from environments.location_finding.types import SourceConfig


def canonicalize_source_config(hypothesis: SourceConfig | np.ndarray) -> np.ndarray:
    """Return an order-stable source array for permutation-invariant averaging."""
    sources = np.asarray(hypothesis, dtype=float)
    if sources.ndim != 2:
        raise ValueError("source configuration must have shape (num_sources, dim)")
    keys = tuple(sources[:, column] for column in reversed(range(sources.shape[1])))
    return sources[np.lexsort(keys)]


def posterior_mean_source_config(
    belief_state: BeliefState[SourceConfig],
) -> SourceConfig | None:
    """Compute a permutation-stable posterior mean source configuration."""
    if not belief_state.hypotheses:
        return None
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    if probabilities.shape != (len(belief_state.hypotheses),):
        raise ValueError("belief hypotheses and probabilities must have matching lengths")
    probabilities = np.where(np.isfinite(probabilities) & (probabilities > 0.0), probabilities, 0.0)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return None
    probabilities /= total
    canonical = np.stack(
        [canonicalize_source_config(hypothesis) for hypothesis in belief_state.hypotheses],
        axis=0,
    )
    estimate = np.sum(canonical * probabilities[:, None, None], axis=0)
    return tuple(tuple(float(value) for value in source) for source in estimate)


def posterior_expected_source_rmse(
    belief_state: BeliefState[SourceConfig],
) -> float:
    """Return posterior expected permutation-matched RMSE to its mean estimate.

    This is a decision-time task loss: it depends only on the posterior and never on
    the environment's hidden truth.
    """
    estimate = posterior_mean_source_config(belief_state)
    if estimate is None:
        return float("inf")

    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    probabilities = np.where(np.isfinite(probabilities) & (probabilities > 0.0), probabilities, 0.0)
    probabilities /= float(np.sum(probabilities))
    estimate_array = np.asarray(estimate, dtype=float)
    hypotheses = np.asarray(belief_state.hypotheses, dtype=float)
    if hypotheses.ndim != 3 or hypotheses.shape[1:] != estimate_array.shape:
        raise ValueError("all source hypotheses must have the same shape")

    permutation_indices = np.asarray(
        list(permutations(range(estimate_array.shape[0]))),
        dtype=int,
    )
    matched_mses = []
    for indices in permutation_indices:
        differences = hypotheses[:, indices, :] - estimate_array[None, :, :]
        matched_mses.append(np.mean(differences * differences, axis=(1, 2)))
    rmses = np.sqrt(np.min(np.stack(matched_mses, axis=1), axis=1))
    result = float(np.sum(probabilities * rmses))
    return result if math.isfinite(result) else float("inf")
