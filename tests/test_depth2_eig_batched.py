"""Tests for batched LLM depth-2 location EIG."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from environments.location_finding.depth2_eig import score_location_candidates_depth2_batched
from environments.location_finding.runner import (
    LocationBeliefState,
    LocationObservation,
    build_location_belief_state,
    expected_information_gain,
    normalize_source_config,
)
from tests.test_location_finding import FakeLocationModel, _location_config


def test_depth2_batched_uses_immediate_scores_and_batches_llm_helpers():
    config = _location_config(
        location_search_depth=2,
        location_posterior_mode="llm_distribution",
        location_eig_quadrature_order=5,
        num_mc_samples=50,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    candidates = [(0.0, 0.0), (1.0, 1.0)]
    immediate = [
        expected_information_gain(belief_state, candidate, config.location_noise_sd, 5)
        for candidate in candidates
    ]
    model = FakeLocationModel([])

    with (
        patch(
            "environments.location_finding.depth2_eig._generate_location_hypotheses_many",
            return_value=[[hypothesis_a], [hypothesis_b]],
        ) as gen_many,
        patch(
            "environments.location_finding.depth2_eig.build_location_posteriors_many",
            side_effect=lambda _q, hypotheses_many, *_a, **_k: [
                build_location_belief_state(hypotheses, [], config)
                for hypotheses in hypotheses_many
            ],
        ) as posterior_many,
        patch(
            "environments.location_finding.depth2_eig.generate_location_candidates_many",
            return_value=[[(0.5, 0.5)], [(1.5, 1.5)]],
        ) as candidates_many,
    ):
        scores = score_location_candidates_depth2_batched(
            belief_state,
            candidates,
            config,
            model,
            [],
            immediate_scores=immediate,
        )

    assert len(scores) == 2
    assert gen_many.call_count == 1
    assert posterior_many.call_count == 1
    assert candidates_many.call_count == 1
    assert gen_many.call_args[0][1]  # observations_many
    assert all(isinstance(obs[-1], LocationObservation) for obs in gen_many.call_args[0][1])
