"""Tests for hyperbolic temporal discounting (continuous observation surrogate)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import core.defaults as core_defaults
from core import BEDRunner, build_environment, build_method
from helpers import Config
from hyperbolic_discounting import (
    HyperbolicBeliefState,
    HyperbolicDesign,
    HyperbolicDiscountingEnv,
    HyperbolicObservation,
    HyperbolicParams,
    build_hyperbolic_belief_state,
    expected_information_gain,
    implied_choice_accuracy,
    implied_prefers_delayed,
    latent_mean,
    run_hyperbolic_finding,
    score_candidate_designs,
)


@pytest.fixture(autouse=True)
def _register_defaults():
    core_defaults.register_defaults(force=True)
    yield
    core_defaults.register_defaults(force=True)


def _htd_config(**overrides) -> Config:
    base = dict(
        task="hyperbolic_discounting",
        htd_num_rounds=2,
        htd_num_trials=1,
        htd_noise_sd=0.25,
        htd_target_num_candidates=3,
        htd_search_depth=1,
        htd_eig_quadrature_order=5,
        htd_posterior_mode="analytical_likelihood",
        htd_max_total_beliefs=20,
        htd_max_llm_prompt_beliefs=8,
        htd_num_generated_hypotheses=4,
        generation_temperature_diverse=0.0,
    )
    base.update(overrides)
    return Config(**base)


class _RoutingHyperbolicModel:
    def __init__(self) -> None:
        self.batched_calls: list[list[list[dict[str, str]]]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        prompt = "\n".join(message["content"] for message in messages)
        if "candidate parameter pairs" in prompt or "finite Bayesian" in prompt.lower():
            return [
                json.dumps(
                    {
                        "hypotheses": [
                            {"k": 0.5, "alpha": 1.0},
                            {"k": 1.0, "alpha": 0.8},
                            {"k": 2.0, "alpha": 1.2},
                        ]
                    }
                )
            ]
        if "experiment designs" in prompt:
            return [
                json.dumps(
                    {
                        "designs": [
                            {"iR": 10.0, "dR": 20.0, "days": 7},
                            {"iR": 5.0, "dR": 30.0, "days": 14},
                            {"iR": 1.0, "dR": 40.0, "days": 30},
                        ]
                    }
                )
            ]
        return [json.dumps({"hypotheses": [{"k": 1.0, "alpha": 1.0}]})]

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        self.batched_calls.append(batch_messages)
        return [self.chat_complete(messages, temperature)[0] for messages in batch_messages]


def test_latent_mean_and_simulator():
    params = HyperbolicParams(k=1.0, alpha=2.0)
    design = HyperbolicDesign(immediate_reward=10.0, delayed_reward=20.0, days=5)
    expected = (20.0 / (1.0 + 5.0) - 10.0) / 2.0
    assert latent_mean(design, params) == pytest.approx(expected)
    env = HyperbolicDiscountingEnv(noise_sd=0.1, true_params=params, rng=np.random.default_rng(0))
    observation = env.run_experiment(design)
    assert isinstance(observation.value, float)


def test_analytical_belief_update():
    config = _htd_config()
    truth = HyperbolicParams(k=1.2, alpha=0.9)
    hypotheses = [
        HyperbolicParams(k=1.0, alpha=1.0),
        HyperbolicParams(k=1.5, alpha=0.8),
        HyperbolicParams(k=0.8, alpha=1.1),
    ]
    design = HyperbolicDesign(immediate_reward=5.0, delayed_reward=25.0, days=10)
    value = latent_mean(design, truth)
    observations = [HyperbolicObservation(design=design, value=value)]
    belief = build_hyperbolic_belief_state(hypotheses, observations, config)
    assert len(belief.hypotheses) == 3
    assert sum(belief.probabilities) == pytest.approx(1.0)
    assert max(belief.probabilities) > min(belief.probabilities)


def test_expected_information_gain_positive():
    config = _htd_config()
    belief = HyperbolicBeliefState(
        hypotheses=[
            HyperbolicParams(k=0.5, alpha=1.0),
            HyperbolicParams(k=2.0, alpha=1.0),
        ],
        probabilities=[0.5, 0.5],
    )
    design = HyperbolicDesign(immediate_reward=1.0, delayed_reward=50.0, days=30)
    eig = expected_information_gain(belief, design, config.htd_noise_sd, config.htd_eig_quadrature_order)
    assert eig > 0.0


def test_score_candidate_designs_with_fixed_candidates():
    config = _htd_config()
    belief = HyperbolicBeliefState(
        hypotheses=[
            HyperbolicParams(k=0.5, alpha=1.0),
            HyperbolicParams(k=2.0, alpha=1.0),
        ],
        probabilities=[0.5, 0.5],
    )
    candidates = [
        HyperbolicDesign(immediate_reward=1.0, delayed_reward=40.0, days=7),
        HyperbolicDesign(immediate_reward=5.0, delayed_reward=20.0, days=30),
    ]
    scores = score_candidate_designs(belief, candidates, config)
    assert len(scores) == 2
    assert all(score >= 0.0 for score in scores)


def test_implied_choice_accuracy_on_holdout_designs():
    truth = HyperbolicParams(k=1.0, alpha=1.0)
    estimate = HyperbolicParams(k=1.0, alpha=1.0)
    designs = [
        HyperbolicDesign(immediate_reward=1.0, delayed_reward=50.0, days=30),
        HyperbolicDesign(immediate_reward=40.0, delayed_reward=50.0, days=7),
    ]
    assert implied_choice_accuracy(estimate, truth, designs) == pytest.approx(1.0)
    assert implied_prefers_delayed(truth, designs[0]) != implied_prefers_delayed(truth, designs[1])


def test_depth2_analytical_forward_search():
    config = _htd_config(htd_search_depth=2, htd_posterior_mode="analytical_likelihood")
    belief = HyperbolicBeliefState(
        hypotheses=[
            HyperbolicParams(k=0.3, alpha=1.0),
            HyperbolicParams(k=3.0, alpha=1.0),
        ],
        probabilities=[0.5, 0.5],
    )
    candidates = [
        HyperbolicDesign(immediate_reward=1.0, delayed_reward=60.0, days=14),
        HyperbolicDesign(immediate_reward=10.0, delayed_reward=15.0, days=14),
    ]
    scores = score_candidate_designs(belief, candidates, config, questioner=None, observations=[])
    assert len(scores) == 2
    assert all(score >= 0.0 for score in scores)


def test_run_hyperbolic_finding_batches_across_trials():
    model = _RoutingHyperbolicModel()
    config = _htd_config(
        htd_num_trials=3,
        htd_num_rounds=1,
        htd_trial_batch_size=3,
        htd_target_num_candidates=2,
    )
    metrics = run_hyperbolic_finding(
        model,
        config,
        rng=np.random.default_rng(1),
        method_name="EIG",
    )
    assert len(metrics.parameter_rmse) == 1
    assert len(metrics.implied_choice_accuracy) == 1
    assert len(model.batched_calls) >= 2


def test_bed_runner_smoke_with_stub_llm():
    config = _htd_config(htd_num_rounds=1)
    model = _RoutingHyperbolicModel()
    env = build_environment("hyperbolic_discounting", config, model, None)
    method = build_method("hyperbolic_discounting", "EIG", config)
    runner = BEDRunner(
        environment=env,
        method=method,
        model=model,
        config=config,
        num_trials=1,
        num_rounds=1,
        rng=np.random.default_rng(0),
    )
    result = runner.run()
    assert len(result.trials) == 1
    assert len(result.trials[0].rounds) == 1
