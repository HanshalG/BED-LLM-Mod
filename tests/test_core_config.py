"""Tests for the typed configuration views over the legacy flat ``Config``."""

from __future__ import annotations

import pytest

from core import (
    AnimalsConfig,
    BaseConfig,
    HyperbolicConfig,
    LocationConfig,
    animals_view,
    base_view,
    hyperbolic_view,
    location_view,
)
from helpers import Config


def test_base_view_extracts_run_and_model_fields():
    config = Config(
        run_id="abc",
        batched_block_size=64,
        generation_temperature_simple=0.4,
        gpu_memory_utilization=0.7,
    )

    view = base_view(config)

    assert isinstance(view, BaseConfig)
    assert view.run_id == "abc"
    assert view.batched_block_size == 64
    assert view.generation_temperature_simple == pytest.approx(0.4)
    assert view.gpu_memory_utilization == pytest.approx(0.7)


def test_animals_view_extracts_animals_fields():
    config = Config(
        task="animals",
        version=2,
        animals=[["dog", "cat"]],
        target_num_questions=8,
        num_mc_samples=5,
        belief_state_mode="categorical",
        belief_distribution_num_calls=3,
    )

    view = animals_view(config)

    assert isinstance(view, AnimalsConfig)
    assert view.version == 2
    assert view.animals == [["dog", "cat"]]
    assert view.target_num_questions == 8
    assert view.num_mc_samples == 5
    assert view.belief_state_mode == "categorical"
    assert view.belief_distribution_num_calls == 3


def test_location_view_extracts_location_fields():
    config = Config(
        task="location_finding",
        location_num_rounds=10,
        location_num_trials=3,
        location_num_sources=4,
        location_dim=3,
        location_noise_sd=0.25,
        location_query_bounds=[-4.0, 4.0],
        location_max_total_beliefs=500,
        location_max_llm_prompt_beliefs=30,
        location_num_generated_hypotheses=25,
        location_eig_quadrature_order=11,
        location_strategy_num_retrieved=3,
        location_strategy_num_mutation=2,
        location_strategy_num_crossover=2,
        location_strategy_num_diverse=1,
    )

    view = location_view(config)

    assert isinstance(view, LocationConfig)
    assert view.num_rounds == 10
    assert view.num_trials == 3
    assert view.num_sources == 4
    assert view.dim == 3
    assert view.noise_sd == pytest.approx(0.25)
    assert view.query_bounds == (-4.0, 4.0)
    assert view.max_total_beliefs == 500
    assert view.max_llm_prompt_beliefs == 30
    assert view.num_generated_hypotheses == 25
    assert view.eig_quadrature_order == 11
    # Derived: 3 + 2 + 2 + 1 = 8
    assert view.strategy_num_candidates == 8


def test_hyperbolic_view_extracts_hyperbolic_fields():
    config = Config(
        task="hyperbolic_discounting",
        htd_num_rounds=6,
        htd_num_trials=2,
        htd_noise_sd=0.3,
        htd_ir_bounds=[1.0, 50.0],
        htd_dr_bounds=[2.0, 80.0],
        htd_days_bounds=[7, 90],
        htd_max_total_beliefs=200,
        htd_target_num_candidates=9,
        htd_search_depth=2,
        htd_posterior_mode="llm_distribution",
        htd_k_mean=0.1,
        htd_k_std=0.9,
        htd_alpha_scale=1.5,
    )
    view = hyperbolic_view(config)
    assert isinstance(view, HyperbolicConfig)
    assert view.num_rounds == 6
    assert view.num_trials == 2
    assert view.noise_sd == pytest.approx(0.3)
    assert view.ir_bounds == (1.0, 50.0)
    assert view.dr_bounds == (2.0, 80.0)
    assert view.days_bounds == (7, 90)
    assert view.posterior_mode == "llm_distribution"
    assert view.alpha_scale == pytest.approx(1.5)


def test_location_view_rejects_malformed_query_bounds():
    config = Config(
        task="location_finding",
        location_query_bounds=[1.0, 2.0, 3.0],
    )
    with pytest.raises(ValueError, match="query_bounds"):
        location_view(config)


def test_views_are_immutable():
    config = Config(task="animals", target_num_questions=7)
    view = animals_view(config)
    with pytest.raises((AttributeError, TypeError)):
        view.target_num_questions = 99  # type: ignore[misc]


def test_views_reflect_legacy_config_updates_each_time_they_are_built():
    # The legacy Config is mutable; views are constructed on demand so they
    # always reflect the current state.
    config = Config(task="animals", target_num_questions=5)
    view1 = animals_view(config)
    assert view1.target_num_questions == 5

    config.target_num_questions = 12
    view2 = animals_view(config)
    assert view2.target_num_questions == 12

    # The old view is still its own (immutable) snapshot.
    assert view1.target_num_questions == 5


def test_base_view_works_on_a_minimally_specified_object():
    # The views deliberately use getattr with defaults so they work on
    # configuration-like objects that don't implement the full Config surface.
    class Minimal:
        run_id = "xyz"

    view = base_view(Minimal())

    assert view.run_id == "xyz"
    # All other fields fall back to defaults.
    assert view.batched_block_size == 50
    assert view.gpu_memory_utilization == pytest.approx(0.88)
