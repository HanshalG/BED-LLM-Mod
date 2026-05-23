"""Tests for the new ``environments.<env>.prompts`` public surfaces.

These verify the re-export shims so new code can rely on the stable import
path even before the bodies of the prompt functions are physically moved.
"""

from __future__ import annotations

import pytest

from environments.animals import prompts as animals_prompts
from environments.location_finding import prompts as location_prompts


# ---------------------------------------------------------------------------
# Animals
# ---------------------------------------------------------------------------


def test_animals_prompts_exports_match_legacy_module():
    import prompts as legacy_prompts

    for name in animals_prompts.__all__:
        legacy = getattr(legacy_prompts, name)
        new = getattr(animals_prompts, name)
        assert new is legacy, (
            f"environments.animals.prompts.{name} should re-export prompts.{name}"
        )


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


def test_location_prompts_exports_match_legacy_module():
    import location_finding as legacy_location

    name_map = {
        "belief_generation_messages": "_belief_generation_messages",
        "belief_output_contract": "_belief_output_contract",
        "belief_system_prompt": "_belief_system_prompt",
        "candidate_generation_messages": "_candidate_generation_messages",
        "location_posterior_distribution_messages": "_location_posterior_distribution_messages",
        "naive_location_messages": "_naive_location_messages",
        "naive_source_estimate_messages": "_naive_source_estimate_messages",
        "naive_source_estimate_repair_messages": "_naive_source_estimate_repair_messages",
        "strategy_crossover_messages": "_strategy_crossover_messages",
        "strategy_diverse_messages": "_strategy_diverse_messages",
        "strategy_location_messages": "_strategy_location_messages",
        "strategy_mutation_messages": "_strategy_mutation_messages",
        "strategy_root_crossover_messages": "_strategy_root_crossover_messages",
        "strategy_root_diverse_messages": "_strategy_root_diverse_messages",
        "strategy_root_mutation_messages": "_strategy_root_mutation_messages",
        "strategy_root_system_preamble": "_strategy_root_system_preamble",
        "strategy_system_preamble": "_strategy_system_preamble",
    }

    for public_name, legacy_name in name_map.items():
        new = getattr(location_prompts, public_name)
        legacy = getattr(legacy_location, legacy_name)
        assert new is legacy, (
            f"environments.location_finding.prompts.{public_name} should re-export "
            f"location_finding.{legacy_name}"
        )


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
