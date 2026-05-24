"""Tests for the ``environments.<env>.prompts`` public surfaces."""

from __future__ import annotations

import pytest

from environments.animals import prompts as animals_prompts
from environments.location_finding import prompts as location_prompts


# ---------------------------------------------------------------------------
# Animals
# ---------------------------------------------------------------------------


def test_animals_prompts_exports_expected_public_names():
    expected = {
        "answer_likelihood_messages",
        "answer_likelihood_system_prompt",
        "answer_likelihood_user_prompt",
        "answer_question_yesno_system_prompt",
        "answer_question_yesnocorrect_system_prompt",
        "belief_distribution_system_prompt",
        "belief_distribution_user_prompt",
        "candidate_generation_system_message",
        "candidate_generation_system_message_naive",
        "conditional_question_generation_prompt",
        "convert_to_prompt_message",
        "generate_animals_system_prompt",
        "generate_animals_user_prompt",
        "generate_more_animals_system_prompt",
        "generate_original_animals_system_prompt",
        "greedy_sample_animal_system_prompt",
        "greedy_sample_animal_system_prompt_naive",
        "greedy_sample_animal_user_prompt",
        "greedy_sample_animal_user_prompt_naive",
        "is_answer_likelihood_messages",
        "probability_answer_scores_prompt",
        "question_generation_prompt_naive",
        "unconditional_question_generation_prompt",
        "validate_animal_name_system_prompt",
        "validate_animal_name_user_prompt",
        "weighted_conditional_question_generation_prompt",
        "weighted_greedy_sample_animal_user_prompt_naive",
        "weighted_question_generation_prompt_naive",
        "weighted_unconditional_question_generation_prompt",
    }

    assert set(animals_prompts.__all__) == expected
    assert all(callable(getattr(animals_prompts, name)) for name in expected)


def test_animals_belief_distribution_system_prompt_returns_message_dict():
    message = animals_prompts.belief_distribution_system_prompt()
    assert message["role"] == "system"
    assert "20 Questions" in message["content"]


def test_animals_answer_likelihood_messages_round_trips_through_validator():
    messages = animals_prompts.answer_likelihood_messages(
        "Wolverine", "Is it native to North America?", ["Yes", "No"]
    )
    assert animals_prompts.is_answer_likelihood_messages(messages)


# ---------------------------------------------------------------------------
# Location finding
# ---------------------------------------------------------------------------


def test_location_prompts_exports_expected_public_names():
    expected = {
        "belief_generation_messages",
        "belief_output_contract",
        "belief_system_prompt",
        "candidate_generation_messages",
        "location_posterior_distribution_messages",
        "naive_location_messages",
        "naive_source_estimate_messages",
        "naive_source_estimate_repair_messages",
        "strategy_crossover_messages",
        "strategy_diverse_messages",
        "strategy_location_messages",
        "strategy_mutation_messages",
        "strategy_root_crossover_messages",
        "strategy_root_diverse_messages",
        "strategy_root_mutation_messages",
        "strategy_root_system_preamble",
        "strategy_system_preamble",
    }

    assert set(location_prompts.__all__) == expected
    assert all(callable(getattr(location_prompts, name)) for name in expected)


def test_location_belief_system_prompt_includes_dimension_and_source_count():
    from helpers import Config

    config = Config(
        task="location_finding",
        location_num_sources=3,
        location_dim=2,
        location_noise_sd=0.5,
    )
    text = location_prompts.belief_system_prompt(config, update=False)

    assert "2D" in text
    assert "noise_sd=0.5" in text


def test_location_strategy_system_preamble_includes_bounds_and_count():
    text = location_prompts.strategy_system_preamble(
        bounds=(-2.0, 2.0),
        num_strategies=5,
        task_instruction="Be creative.",
    )

    assert "Generate exactly 5 strategies" in text
    assert "[-2.0, 2.0]" in text
    assert "Be creative." in text
