"""Tests for the typed configuration views over the flat flat ``Config``."""

from __future__ import annotations

import pytest

from core import AnimalsConfig, BaseConfig, LocationConfig, animals_view, base_view, location_view
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
        belief_generation_num_calls=4,
        belief_distribution_num_calls=3,
    )

    view = animals_view(config)

    assert isinstance(view, AnimalsConfig)
    assert view.version == 2
    assert view.animals == [["dog", "cat"]]
    assert view.target_num_questions == 8
    assert view.num_mc_samples == 5
    assert view.belief_state_mode == "categorical"
    assert view.belief_generation_num_calls == 4
    assert view.belief_distribution_num_calls == 3


def test_location_view_extracts_location_fields():
    config = Config(
        task="location_finding",
        location_num_rounds=10,
        location_num_trials=3,
        location_source_prior="branch_decoy",
        location_source_radius=2.2,
        location_num_sources=4,
        location_dim=3,
        location_noise_sd=0.25,
        location_signal_model="local_bump",
        location_signal_lengthscale=0.5,
        location_signal_amplitude=8.0,
        location_max_step_radius=0.75,
        location_max_total_beliefs=500,
        location_max_llm_prompt_beliefs=30,
        location_num_generated_hypotheses=25,
        location_belief_support_refresh_enabled=False,
        location_candidate_generation_mode="support_grid",
        location_eig_quadrature_order=11,
        location_strategy_num_retrieved=3,
        location_strategy_num_mutation=2,
        location_strategy_num_crossover=2,
        location_strategy_num_diverse=1,
        location_strategy_rollout_refresh_hypotheses_each_step=True,
        location_strategy_rollout_scoring_support_mode="truth_plus_sampled",
        location_strategy_rollout_scoring_support_size=17,
        location_strategy_rollout_score_mode="future_step_support_sum",
        location_strategy_rollout_final_refresh_enabled=False,
        location_strategy_rollout_query_mode="analytic_eig",
        location_eig_bounds_enabled=True,
        location_eig_bounds_inner_samples=101,
        location_eig_bounds_seed=44,
        location_eig_bounds_chunk_size=2048,
    )

    view = location_view(config)

    assert isinstance(view, LocationConfig)
    assert view.num_rounds == 10
    assert view.num_trials == 3
    assert view.source_prior == "branch_decoy"
    assert view.source_radius == pytest.approx(2.2)
    assert view.num_sources == 4
    assert view.dim == 3
    assert view.noise_sd == pytest.approx(0.25)
    assert view.signal_model == "local_bump"
    assert view.signal_lengthscale == pytest.approx(0.5)
    assert view.signal_amplitude == pytest.approx(8.0)
    assert view.max_step_radius == pytest.approx(0.75)
    assert view.max_total_beliefs == 500
    assert view.max_llm_prompt_beliefs == 30
    assert view.num_generated_hypotheses == 25
    assert view.belief_support_refresh_enabled is False
    assert view.candidate_generation_mode == "support_grid"
    assert view.eig_quadrature_order == 11
    assert view.eig_bounds_enabled is True
    assert view.eig_bounds_inner_samples == 101
    assert view.eig_bounds_seed == 44
    assert view.eig_bounds_chunk_size == 2048
    assert view.strategy_rollout_refresh_hypotheses_each_step is True
    assert view.strategy_rollout_scoring_support_mode == "truth_plus_sampled"
    assert view.strategy_rollout_scoring_support_size == 17
    assert view.strategy_rollout_score_mode == "future_step_support_sum"
    assert view.strategy_rollout_final_refresh_enabled is False
    assert view.strategy_rollout_query_mode == "analytic_eig"
    # Derived: 3 + 2 + 2 + 1 = 8
    assert view.strategy_num_candidates == 8

def test_views_are_immutable():
    config = Config(task="animals", target_num_questions=7)
    view = animals_view(config)
    with pytest.raises((AttributeError, TypeError)):
        view.target_num_questions = 99  # type: ignore[misc]


def test_views_reflect_flat_config_updates_each_time_they_are_built():
    # The flat Config is mutable; views are constructed on demand so they
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


def test_config_exposes_typed_view_properties():
    config = Config(
        run_id="run-1",
        task="location_finding",
        location_num_rounds=3,
        animals_num_rounds=7,
    )

    assert config.base.run_id == "run-1"
    assert config.location_config.num_rounds == 3
    assert config.animals_config.animals_num_rounds == 7
