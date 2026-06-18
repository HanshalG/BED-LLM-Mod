"""Tests for the Environment adapters that wrap animals and location_finding."""

from __future__ import annotations

import math
import json
from typing import Any

import numpy as np
import pytest

from core import BeliefState
from environments.animals import AnimalsBEDEnvironment
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


def test_location_adapter_log_prior_matches_physics_helper():
    from environments.location_finding.physics import hypothesis_log_prior

    config = _location_config()
    env = LocationBEDEnvironment(config=config)
    hypothesis = ((0.0, 0.0), (1.0, 0.0))

    assert env.log_prior(hypothesis) == pytest.approx(hypothesis_log_prior(hypothesis))


def test_location_adapter_log_likelihood_single_matches_vectorised():
    config = _location_config()
    env = LocationBEDEnvironment(config=config)
    from environments.location_finding.types import LocationObservation

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


def test_location_naive_belief_includes_current_posterior_in_prompt():
    config = _location_config()
    model = _StubLLM()
    model.complete_responses = ['{"location":[0.1,0.2]}', '{"location":[0.3,0.4]}']
    env = LocationBEDEnvironment(config=config)
    belief = BeliefState(
        hypotheses=(((0.0, 0.0), (1.0, 1.0)), ((-1.0, 0.0), (0.0, -1.0))),
        probabilities=(0.8, 0.2),
    )

    env.generate_naive_action(belief, [], model, config, method_name="naive")
    plain_prompt = model.complete_calls[-1][-1]["content"]
    env.generate_naive_action(belief, [], model, config, method_name="naive+belief")
    belief_prompt = model.complete_calls[-1][-1]["content"]

    assert env.naive_requires_belief_state("naive") is False
    assert env.naive_requires_belief_state("naive+belief") is True
    assert "Current belief summary" not in plain_prompt
    assert "Current belief summary" in belief_prompt
    assert '"probability": 0.8' in belief_prompt


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


def test_animals_trial_index_covers_target_pool_without_replacement():
    env = AnimalsBEDEnvironment(
        config=_animals_config(animals=[["dog", "cat", "lion"]]),
        answerer=_StubLLM(),
        target_animals=["dog", "cat", "lion"],
    )
    rng = np.random.default_rng(0)

    sampled = [env.sample_hidden_state_for_trial(index, rng) for index in range(3)]

    assert sampled == ["dog", "cat", "lion"]


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


def test_animals_strategy_eig_generates_strategy_questions_without_plain_eig(monkeypatch):
    def fail_plain_eig(*args, **kwargs):
        raise AssertionError("StrategyEIG should not call plain animals forward-search EIG")

    monkeypatch.setattr("environments.animals.env.evaluate_questions_forward_search", fail_plain_eig)
    questioner = _StubLLM()
    questioner.complete_responses = [
        '{"strategies":["Separate pets from wild cats","Test size before habitat"]}',
        "Is it commonly kept as a pet?",
        "Is it larger than a house cat?",
    ]
    questioner.probability_responses = [
        [{"Yes": 0.9, "No": 0.1}, {"Yes": 0.8, "No": 0.2}, {"Yes": 0.1, "No": 0.9}],
        [{"Yes": 0.1, "No": 0.9}, {"Yes": 0.2, "No": 0.8}, {"Yes": 0.9, "No": 0.1}],
    ]
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=_StubLLM(),
        target_animals=["dog", "cat", "lion"],
    )
    env.set_questioner(questioner)
    belief = BeliefState(
        hypotheses=("dog", "cat", "lion"),
        probabilities=(1 / 3, 1 / 3, 1 / 3),
    )

    action, score, evaluation = env.choose_strategy_action(
        belief,
        history=[],
        model=questioner,
        config=_animals_config(
            animals_strategy_num_rollouts=1,
            animals_strategy_planning_depth=1,
        ),
        rng=np.random.default_rng(0),
        round_index=0,
        fixed_root=False,
    )

    assert action in {"Is it commonly kept as a pet?", "Is it larger than a house cat?"}
    assert score > 0.0
    assert evaluation["fixed_root"] is False
    assert evaluation["num_rollouts"] == 1
    assert evaluation["planning_depth"] == 1
    assert "Strategy to follow" in questioner.complete_calls[1][-1]["content"]


def test_animals_strategy_eig_root_asks_selected_root_question():
    questioner = _StubLLM()
    questioner.complete_responses = [
        json.dumps(
            {
                "candidates": [
                    {
                        "strategy": "First split domestic from wild animals.",
                        "root_question": "Is it commonly kept as a pet?",
                    },
                    {
                        "strategy": "First split very large animals.",
                        "root_question": "Is it larger than a person?",
                    },
                ]
            }
        )
    ]
    questioner.probability_responses = [
        [{"Yes": 0.9, "No": 0.1}, {"Yes": 0.8, "No": 0.2}, {"Yes": 0.1, "No": 0.9}],
        [{"Yes": 0.1, "No": 0.9}, {"Yes": 0.2, "No": 0.8}, {"Yes": 0.9, "No": 0.1}],
    ]
    env = AnimalsBEDEnvironment(
        config=_animals_config(),
        answerer=_StubLLM(),
        target_animals=["dog", "cat", "lion"],
    )
    env.set_questioner(questioner)
    belief = BeliefState(
        hypotheses=("dog", "cat", "lion"),
        probabilities=(1 / 3, 1 / 3, 1 / 3),
    )

    action, score, evaluation = env.choose_strategy_action(
        belief,
        history=[],
        model=questioner,
        config=_animals_config(
            animals_strategy_num_rollouts=1,
            animals_strategy_planning_depth=1,
        ),
        rng=np.random.default_rng(0),
        round_index=0,
        fixed_root=True,
    )

    assert action in {"Is it commonly kept as a pet?", "Is it larger than a person?"}
    assert score > 0.0
    assert evaluation["fixed_root"] is True
    assert "root_question" in questioner.complete_calls[0][-1]["content"]


def test_animals_strategy_eig_rollout_uses_planning_depth_for_followups():
    questioner = _StubLLM()
    questioner.complete_responses = [
        '{"strategies":["Start broad, then follow the likely branch."]}',
        "Is it commonly kept as a pet?",
        "Does it usually live indoors?",
    ]
    questioner.probability_responses = [
        [{"Yes": 0.9, "No": 0.1}, {"Yes": 0.8, "No": 0.2}, {"Yes": 0.1, "No": 0.9}],
        [{"Yes": 0.9, "No": 0.1}, {"Yes": 0.5, "No": 0.5}, {"Yes": 0.2, "No": 0.8}],
    ]
    config = _animals_config(
        animals_strategy_num_retrieved=0,
        animals_strategy_num_mutation=0,
        animals_strategy_num_crossover=0,
        animals_strategy_num_diverse=1,
        animals_strategy_num_rollouts=1,
        animals_strategy_planning_depth=2,
    )
    env = AnimalsBEDEnvironment(
        config=config,
        answerer=_StubLLM(),
        target_animals=["dog", "cat", "lion"],
    )
    env.set_questioner(questioner)
    belief = BeliefState(
        hypotheses=("dog", "cat", "lion"),
        probabilities=(1 / 3, 1 / 3, 1 / 3),
    )

    action, score, evaluation = env.choose_strategy_action(
        belief,
        history=[],
        model=questioner,
        config=config,
        rng=np.random.default_rng(1),
        round_index=0,
        fixed_root=False,
    )

    assert action == "Is it commonly kept as a pet?"
    assert score > 0.0
    assert evaluation["planning_depth"] == 2
    assert evaluation["rollout_scores"][0]
    assert len(questioner.probability_calls) == 2
    assert "Does it usually live indoors?" in questioner.probability_calls[1][0][-1]["content"]
