"""Tests for batched LLM depth-2 hyperbolic EIG."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from environments.hyperbolic_discounting.depth2_eig import score_hyperbolic_candidates_depth2_batched
from environments.hyperbolic_discounting.runner import (
    HyperbolicBeliefState,
    HyperbolicDesign,
    HyperbolicObservation,
    HyperbolicParams,
    build_hyperbolic_belief_state,
    expected_information_gain,
)
from tests.test_hyperbolic_discounting import _RoutingHyperbolicModel, _htd_config


def test_depth2_batched_batches_llm_helpers():
    config = _htd_config(
        htd_search_depth=2,
        htd_posterior_mode="llm_distribution",
        num_mc_samples=50,
    )
    hypothesis_a = HyperbolicParams(k=0.5, alpha=1.0)
    hypothesis_b = HyperbolicParams(k=2.0, alpha=0.8)
    belief_state = build_hyperbolic_belief_state([hypothesis_a, hypothesis_b], [], config)
    candidates = [
        HyperbolicDesign(immediate_reward=1.0, delayed_reward=40.0, days=7),
        HyperbolicDesign(immediate_reward=5.0, delayed_reward=20.0, days=30),
    ]
    immediate = [
        expected_information_gain(belief_state, candidate, config.htd_noise_sd, config.htd_eig_quadrature_order)
        for candidate in candidates
    ]
    model = _RoutingHyperbolicModel()

    with (
        patch(
            "environments.hyperbolic_discounting.depth2_eig._generate_hyperbolic_hypotheses_many",
            return_value=[[hypothesis_a], [hypothesis_b]],
        ) as gen_many,
        patch(
            "environments.hyperbolic_discounting.depth2_eig.build_hyperbolic_posteriors_many",
            side_effect=lambda _q, hypotheses_many, *_a, **_k: [
                build_hyperbolic_belief_state(hypotheses, [], config)
                for hypotheses in hypotheses_many
            ],
        ) as posterior_many,
        patch(
            "environments.hyperbolic_discounting.depth2_eig.generate_hyperbolic_candidates_many",
            return_value=[
                [HyperbolicDesign(immediate_reward=2.0, delayed_reward=35.0, days=10)],
                [HyperbolicDesign(immediate_reward=8.0, delayed_reward=25.0, days=20)],
            ],
        ) as candidates_many,
    ):
        scores = score_hyperbolic_candidates_depth2_batched(
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
    assert all(isinstance(obs[-1], HyperbolicObservation) for obs in gen_many.call_args[0][1])
