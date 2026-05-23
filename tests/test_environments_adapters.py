"""Tests for the Environment adapters that wrap animals and location_finding."""

from __future__ import annotations

import math
import json
from typing import Any

import numpy as np
import pytest

from core import BeliefState
from environments.animals import AnimalsBEDEnvironment
from environments.hyperbolic_discounting import HyperbolicBEDEnvironment
from environments.hyperbolic_discounting.runner import HyperbolicDesign, HyperbolicObservation, HyperbolicParams
from environments.location_finding import LocationBEDEnvironment
from helpers import Config


# ---------------------------------------------------------------------------
# Stub LLMs
# ---------------------------------------------------------------------------


class _StubLLM:
    """Programmable stub Model for unit-testing the adapters."""

    def __init__(self) -> None:
        self.complete_responses: list[str] = []
        self.batched_responses: list[list[str]] = []
        self.probability_responses: list[list[dict[str, float]]] = []
        self.complete_calls: list[list[dict[str, str]]] = []
        self.batched_calls: list[list[list[dict[str, str]]]] = []
        self.probability_calls: list[list[list[dict[str, str]]]] = []

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        num_responses: int = 1,
    ) -> list[str]:
        self.complete_calls.append([dict(m) for m in messages])
        if not self.complete_responses:
            raise AssertionError("StubLLM.chat_complete ran out of canned responses")
        return [self.complete_responses.pop(0)]

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        block_size: int = 50,
        max_new_tokens: int = 8192,
    ) -> list[str]:
        self.batched_calls.append([[dict(m) for m in msgs] for msgs in batch_messages])
        if not self.batched_responses:
            raise AssertionError("StubLLM.chat_complete_messages_batched ran out of canned responses")
        responses = self.batched_responses.pop(0)
        if len(responses) != len(batch_messages):
            raise AssertionError(
                f"Canned batched response length ({len(responses)}) != batch size ({len(batch_messages)})"
            )
        return list(responses)

    def chat_probabilities_messages_batched(
        self,
        messages: list[list[dict[str, str]]],
        responses: list[str],
        temperature: float = 0.0,
        block_size: int = 50,
    ) -> list[dict[str, float]]:
        self.probability_calls.append([[dict(m) for m in msgs] for msgs in messages])
        if not self.probability_responses:
            raise AssertionError("StubLLM.chat_probabilities_messages_batched ran out of canned responses")
        canned = self.probability_responses.pop(0)
        if len(canned) != len(messages):
            raise AssertionError(
                f"Canned probability response length ({len(canned)}) != batch size ({len(messages)})"
            )
        return list(canned)


# ---------------------------------------------------------------------------
# LocationBEDEnvironment — analytical likelihood, no LLM needed for math.
# ---------------------------------------------------------------------------


def _location_config() -> Config:
    return Config(
        task="location_finding",
        location_num_rounds=2,
        location_num_trials=1,
        location_num_sources=2,
        location_dim=2,
        location_noise_sd=0.5,
        location_query_bounds=[-2.0, 2.0],
        location_max_total_beliefs=50,
        location_max_llm_prompt_beliefs=10,
        location_target_num_candidates=2,
        location_search_depth=1,
        location_eig_quadrature_order=5,
        generation_temperature_diverse=0.0,
    )


def test_location_adapter_sample_hidden_state_respects_fixed_theta():
    config = _location_config()
    theta = np.array([[0.0, 0.0], [1.0, 1.0]])
    env = LocationBEDEnvironment(config=config, true_theta=theta)

    rng = np.random.default_rng(0)
    sampled = env.sample_hidden_state(rng)

    assert sampled.shape == (2, 2)
    np.testing.assert_array_equal(sampled, theta)


def test_location_adapter_observe_returns_observation_with_query_intact():
    config = _location_config()
    theta = np.array([[0.0, 0.0], [1.0, 1.0]])
    env = LocationBEDEnvironment(config=config, true_theta=theta)

    rng = np.random.default_rng(0)
    obs = env.observe((0.5, 0.5), theta, rng)

    assert obs.query == (0.5, 0.5)
    assert isinstance(obs.value, float)


def test_location_adapter_log_prior_matches_legacy_helper():
    import location_finding as legacy

    config = _location_config()
    env = LocationBEDEnvironment(config=config)
    hypothesis = ((0.0, 0.0), (1.0, 0.0))

    assert env.log_prior(hypothesis) == pytest.approx(legacy._hypothesis_log_prior(hypothesis))


def test_location_adapter_log_likelihood_single_matches_vectorised():
    config = _location_config()
    env = LocationBEDEnvironment(config=config)
    from location_finding import LocationObservation

    hyp1 = ((0.0, 0.0), (1.0, 0.0))
    hyp2 = ((-1.0, 0.5), (0.5, -1.0))
    action = (0.25, 0.25)
    observation = LocationObservation(query=action, value=2.3)

    single = [env.log_likelihood(h, action, observation) for h in (hyp1, hyp2)]
    batched = env.log_likelihood_many((hyp1, hyp2), action, observation)

    np.testing.assert_allclose(np.array(single), batched)


def test_location_adapter_round_metrics_reports_rmse_and_top_probability():
    config = _location_config()
    env = LocationBEDEnvironment(config=config)
    true_theta = np.array([[0.0, 0.0], [1.0, 1.0]])

    belief = BeliefState(
        hypotheses=(((0.0, 0.0), (1.0, 1.0)), ((-0.5, 0.0), (0.5, 1.0))),
        probabilities=(0.7, 0.3),
    )

    metrics = env.round_metrics(belief, history=[], hidden_state=true_theta)

    assert metrics["top_probability"] == pytest.approx(0.7)
    assert metrics["source_rmse"] == pytest.approx(0.0)
    assert metrics["support_size"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# AnimalsBEDEnvironment — answerer is a stub LLM; questioner is a stub LLM.
# ---------------------------------------------------------------------------


def _animals_config(**overrides: Any) -> Config:
    defaults = dict(
        task="animals",
        animals=[["dog", "cat", "lion"]],
        target_num_questions=4,
        num_mc_samples=4,
        max_num_samples=10,
        min_num_samples=2,
        belief_state_mode="uniform",
        belief_generation_enabled=False,
        belief_filtering_enabled=False,
        generation_temperature_diverse=0.0,
        generation_temperature_simple=0.0,
        answer_temperature=0.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_animals_adapter_requires_non_empty_animal_pool():
    config = _animals_config(animals=[], version=0)
    with pytest.raises(ValueError, match="target_animals"):
        AnimalsBEDEnvironment(config=config, answerer=_StubLLM(), target_animals=None)


def test_animals_adapter_sample_hidden_state_uses_target_animals():
    answerer = _StubLLM()
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=answerer,
        target_animals=["wolverine"],
    )
    rng = np.random.default_rng(0)

    sampled = env.sample_hidden_state(rng)

    assert sampled == "wolverine"


def test_animals_adapter_observe_calls_answerer_with_yesnocorrect_prompt():
    answerer = _StubLLM()
    answerer.complete_responses = ["Yes"]
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=answerer,
        target_animals=["wolverine"],
    )

    result = env.observe("Is it a mammal?", "wolverine", np.random.default_rng(0))

    assert result == "Yes"
    # Verify the system prompt that was sent contains the goal animal.
    sent = answerer.complete_calls[0]
    assert sent[0]["role"] == "system"
    assert "wolverine" in sent[0]["content"]


def test_animals_adapter_log_prior_uniform_over_pool_when_no_prior_configured():
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=_StubLLM(),
        target_animals=["dog", "cat", "lion"],
    )

    assert env.log_prior("dog") == pytest.approx(-math.log(3))
    assert env.log_prior("not-in-pool") == float("-inf")


def test_animals_adapter_log_likelihood_many_returns_log_probability_of_observed_answer():
    answerer = _StubLLM()
    questioner = _StubLLM()
    questioner.probability_responses = [
        [
            {"Yes": 0.8, "No": 0.2},
            {"Yes": 0.1, "No": 0.9},
        ]
    ]
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=answerer,
        target_animals=["dog", "cat"],
    )
    env.set_questioner(questioner)

    result = env.log_likelihood_many(["dog", "cat"], "Is it big?", "Yes")

    assert result[0] == pytest.approx(math.log(0.8))
    assert result[1] == pytest.approx(math.log(0.1))


def test_animals_adapter_log_likelihood_returns_zero_for_correct_observation():
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=_StubLLM(),
        target_animals=["dog"],
    )
    env.set_questioner(_StubLLM())

    assert env.log_likelihood("dog", "Is it dog?", "Correct!") == 0.0


def test_animals_adapter_round_metrics_computes_correct_belief_mass():
    answerer = _StubLLM()
    answerer.complete_responses = ["No"]
    questioner = _StubLLM()
    questioner.complete_responses = ["lion"]
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=answerer,
        target_animals=["dog", "cat", "lion"],
    )
    env.set_questioner(questioner)
    belief = BeliefState(
        hypotheses=("dog", "Cat", "lion"),  # case-insensitive match
        probabilities=(0.5, 0.3, 0.2),
    )

    metrics = env.round_metrics(belief, history=[], hidden_state="CAT")

    assert metrics["correct_belief_mass"] == pytest.approx(0.3)
    assert metrics["top_belief_correct"] == 0.0  # top is "dog"
    assert metrics["support_size"] == pytest.approx(3.0)


def test_animals_adapter_early_stops_on_correct_observation():
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=_StubLLM(),
        target_animals=["dog"],
    )
    belief = BeliefState.uniform(["dog", "cat"])

    assert env.early_stop(belief, history=[], hidden_state="dog", latest_observation="Correct!")
    assert not env.early_stop(belief, history=[], hidden_state="dog", latest_observation="Yes")


def _hyperbolic_config() -> Config:
    return Config(
        task="hyperbolic_discounting",
        htd_num_rounds=2,
        htd_num_trials=1,
        htd_noise_sd=0.25,
        htd_target_num_candidates=2,
        htd_search_depth=1,
        htd_eig_quadrature_order=5,
        htd_posterior_mode="analytical_likelihood",
        htd_max_total_beliefs=20,
        htd_max_llm_prompt_beliefs=8,
        generation_temperature_diverse=0.0,
    )


def test_hyperbolic_adapter_sample_hidden_state_respects_fixed_theta():
    truth = HyperbolicParams(k=1.1, alpha=0.7)
    env = HyperbolicBEDEnvironment(config=_hyperbolic_config(), true_theta=truth)
    sampled = env.sample_hidden_state(np.random.default_rng(0))
    assert sampled == truth


def test_hyperbolic_adapter_observe_and_likelihood():
    import hyperbolic_discounting as legacy

    config = _hyperbolic_config()
    truth = HyperbolicParams(k=1.0, alpha=1.0)
    env = HyperbolicBEDEnvironment(config=config, true_theta=truth)
    design = HyperbolicDesign(immediate_reward=2.0, delayed_reward=30.0, days=10)
    rng = np.random.default_rng(0)
    observation = env.observe(design, truth, rng)
    assert observation.design == design
    hypothesis = HyperbolicParams(k=1.2, alpha=0.9)
    single = env.log_likelihood(hypothesis, design, observation)
    batched = env.log_likelihood_many((hypothesis,), design, observation)
    np.testing.assert_allclose(np.array([single]), batched)
    mean = legacy.latent_mean(design, hypothesis)
    assert env.predictive_means((hypothesis,), design)[0] == pytest.approx(mean)


def test_hyperbolic_adapter_round_metrics():
    config = _hyperbolic_config()
    truth = HyperbolicParams(k=1.0, alpha=1.0)
    env = HyperbolicBEDEnvironment(config=config)
    belief = BeliefState(
        hypotheses=(HyperbolicParams(k=1.0, alpha=1.0), HyperbolicParams(k=2.0, alpha=1.0)),
        probabilities=(0.8, 0.2),
    )
    metrics = env.round_metrics(belief, history=[], hidden_state=truth)
    assert metrics["parameter_rmse"] == pytest.approx(0.0)
    assert metrics["top_probability"] == pytest.approx(0.8)
