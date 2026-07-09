from __future__ import annotations

import math

import pytest

from core import BeliefState
from environments.location_finding.task_loss import (
    posterior_expected_source_rmse,
    posterior_mean_source_config,
)


def test_posterior_task_loss_is_permutation_invariant() -> None:
    state = BeliefState(
        hypotheses=[
            ((-1.0, 0.0), (1.0, 0.0)),
            ((1.0, 0.0), (-1.0, 0.0)),
        ],
        probabilities=[0.5, 0.5],
    )

    assert posterior_mean_source_config(state) == ((-1.0, 0.0), (1.0, 0.0))
    assert posterior_expected_source_rmse(state) == pytest.approx(0.0)


def test_posterior_task_loss_matches_hand_computed_one_source_case() -> None:
    state = BeliefState(
        hypotheses=[((0.0, 0.0),), ((2.0, 0.0),)],
        probabilities=[0.25, 0.75],
    )

    assert posterior_mean_source_config(state) == ((1.5, 0.0),)
    expected = 0.25 * 1.5 / math.sqrt(2.0) + 0.75 * 0.5 / math.sqrt(2.0)
    assert posterior_expected_source_rmse(state) == pytest.approx(expected)


def test_posterior_task_loss_empty_state_is_infinite() -> None:
    state = BeliefState(hypotheses=[], probabilities=[])
    assert posterior_mean_source_config(state) is None
    assert math.isinf(posterior_expected_source_rmse(state))
