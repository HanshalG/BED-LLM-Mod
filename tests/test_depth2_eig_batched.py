"""Tests for batched LLM depth-2 location EIG."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core import BeliefState
from methods.continuous_eig import score_continuous_forward_search
from environments.location_finding.depth2_eig import score_location_candidates_depth2_batched
from environments.location_finding.beliefs import build_location_belief_state
from environments.location_finding.eig import expected_information_gain
from environments.location_finding.types import LocationObservation, normalize_source_config
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


def test_continuous_eig_calls_depth2_hook_with_adapter_signature():
    class _HookEnv:
        def __init__(self):
            self.args = None

        def predictive_means(self, hypotheses, action):
            return pytest.importorskip("numpy").array([0.0 for _ in hypotheses], dtype=float)

        def score_continuous_forward_search_depth2_batched(
            self,
            belief_state,
            candidates,
            model,
            history,
            config,
            *,
            noise_sd,
            quadrature_order,
            immediate_scores=None,
        ):
            self.args = (belief_state, candidates, model, history, config, noise_sd, quadrature_order)
            return [1.0 for _candidate in candidates]

    config = type("Config", (), {"location_posterior_mode": "llm_distribution"})()
    env = _HookEnv()
    belief = BeliefState(hypotheses=("h0", "h1"), probabilities=(0.5, 0.5))
    model = object()

    scores = score_continuous_forward_search(
        belief,
        candidates=["a", "b"],
        environment=env,
        model=model,
        history=[],
        config=config,
        noise_sd=0.5,
        quadrature_order=5,
        search_depth=2,
    )

    assert scores == [1.0, 1.0]
    assert env.args[2] is model
