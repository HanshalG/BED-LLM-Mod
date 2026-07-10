"""Typed configuration views over the flat ``Config`` god-object.

The repo currently has a single mutable ``helpers.Config`` dataclass with ~70
fields mixing run/model/logging concerns with both animals- and
location-specific parameters.  We can't rip that apart without touching 200+
call sites, so this module takes a different tack:

The flat ``Config`` remains the *single source of truth* used by all existing
code.  This module exposes **typed, immutable views** that select only the
fields relevant to one slice of the system:

- :class:`BaseConfig` — run/model/logging fields shared by every environment
- :class:`AnimalsConfig` — animals-game-specific fields
- :class:`LocationConfig` — location-finding-specific fields
- :class:`PaprikaConfig` — Paprika customer-service-specific fields

New environments and new code paths should consume these typed views via
:func:`base_view`, :func:`animals_view`, and :func:`location_view` — that way
adding a third environment doesn't require touching the flat ``Config``.

Each view is a frozen dataclass.  Because they're built from the flat
``Config`` *at call time*, they always reflect the current values; there is no
sync issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared fields — every BED experiment uses these regardless of environment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseConfig:
    """Run, model, and logging fields shared across all environments."""

    run_id: str = ""
    log_path: Path | None = None
    batched_block_size: int = 50
    generation_temperature_diverse: float = 1.0
    generation_temperature_simple: float = 0.7
    answer_temperature: float = 0.7
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 4096


# ---------------------------------------------------------------------------
# Animals (20 Questions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnimalsConfig:
    """All fields used by the animals (20 Questions) environment."""

    version: int = 0
    animals: list[list[str]] = field(default_factory=list)
    animals_num_rounds: int = 20
    search_depth: int = 1
    forward_search_verbose: bool = True
    target_num_questions: int = 15
    num_mc_samples: int = 15
    max_num_samples: int = 50
    min_num_samples: int = 15
    threshold_rejection_probability: float = 0.2
    belief_state_mode: str = "uniform"
    belief_probability_temperature: float = 0.0
    belief_distribution_num_calls: int = 1
    belief_distribution_permute_history: bool = False
    probability_parse_fallback_to_uniform: bool = True
    belief_prior_mode: str = "none"
    belief_prior_exponential_rate: float = 0.0
    belief_generation_enabled: bool = True
    belief_filtering_enabled: bool = True
    belief_guess_threshold: float | None = 0.99
    answerer_sample_from_prior: bool = False
    answerer_prior_mode: str = "inherit"
    answerer_prior_exponential_rate: float | None = None
    answerer_randomize_prior_order_per_trial: bool = False
    answerer_num_prior_trials: int | None = None
    answerer_prior_seed: int | None = None
    strategy_num_retrieved: int = 2
    strategy_num_mutation: int = 1
    strategy_num_crossover: int = 1
    strategy_num_diverse: int = 2
    strategy_num_rollouts: int = 8
    strategy_planning_depth: int = 8
    strategy_belief_summary_top_k: int = 5


# ---------------------------------------------------------------------------
# Location finding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocationConfig:
    """All fields used by the location-finding environment."""

    num_rounds: int = 20
    num_trials: int = 1
    trial_batch_size: int = 1
    seed: int | None = None
    source_prior: str = "normal"
    source_radius: float = 1.0
    num_sources: int = 3
    dim: int = 2
    noise_sd: float = 0.5
    signal_model: str = "inverse_square"
    signal_lengthscale: float = 0.75
    signal_amplitude: float = 5.0
    max_step_radius: float | None = None
    max_total_beliefs: int = 1000
    max_llm_prompt_beliefs: int = 40
    num_generated_hypotheses: int = 40
    belief_support_refresh_enabled: bool = True
    candidate_generation_mode: str = "llm"
    target_num_candidates: int = 15
    search_depth: int = 2
    eig_quadrature_order: int = 15
    plot_trials: bool = False
    strategy_num_retrieved: int = 2
    strategy_num_mutation: int = 1
    strategy_num_crossover: int = 1
    strategy_num_diverse: int = 2
    strategy_num_rollouts: int = 8
    strategy_planning_depth: int = 8
    strategy_discount_factor: float = 1.0
    strategy_belief_summary_top_k: int = 5
    strategy_rollout_refresh_hypotheses_each_step: bool = False
    strategy_rollout_scoring_support_mode: str = "union"
    strategy_rollout_scoring_support_size: int = 32
    strategy_rollout_score_mode: str = "start_final_entropy_drop"
    strategy_rollout_final_refresh_enabled: bool = True
    strategy_rollout_query_mode: str = "llm_strategy"
    posterior_mode: str = "analytical_likelihood"
    eig_bounds_enabled: bool = False
    eig_bounds_inner_samples: int = 5000
    eig_bounds_seed: int | None = None
    eig_bounds_chunk_size: int = 8192
    max_new_tokens: int = 4096
    belief_distribution_permute_history: bool = False  # shared with animals

    @property
    def strategy_num_candidates(self) -> int:
        return (
            self.strategy_num_retrieved
            + self.strategy_num_mutation
            + self.strategy_num_crossover
            + self.strategy_num_diverse
        )


@dataclass(frozen=True)
class PaprikaConfig:
    data_path: str | None = None
    split: str = "eval"
    verify_official_hash: bool = True
    num_trials: int = 5
    num_rounds: int = 20
    trial_batch_size: int = 1
    task_offset: int = 0
    seed: int | None = None
    num_hypotheses: int = 12
    num_candidates: int = 5
    shared_call_cache_enabled: bool = True
    belief_refresh_enabled: bool = True
    num_refresh_hypotheses: int = 6
    max_hypotheses: int = 24


# ---------------------------------------------------------------------------
# Factories that project a flat ``Config`` into a typed view.
# ---------------------------------------------------------------------------


def base_view(config: Any) -> BaseConfig:
    """Project the run/model/logging fields out of a flat ``Config``."""
    return BaseConfig(
        run_id=getattr(config, "run_id", ""),
        log_path=getattr(config, "log_path", None),
        batched_block_size=getattr(config, "batched_block_size", 50),
        generation_temperature_diverse=getattr(config, "generation_temperature_diverse", 1.0),
        generation_temperature_simple=getattr(config, "generation_temperature_simple", 0.7),
        answer_temperature=getattr(config, "answer_temperature", 0.7),
        tensor_parallel_size=getattr(config, "tensor_parallel_size", None),
        gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.88),
        max_model_len=getattr(config, "max_model_len", 4096),
    )


def animals_view(config: Any) -> AnimalsConfig:
    """Project the animals-specific fields out of a flat ``Config``."""
    return AnimalsConfig(
        version=getattr(config, "version", 0),
        animals=list(getattr(config, "animals", []) or []),
        animals_num_rounds=int(getattr(config, "animals_num_rounds", 20)),
        search_depth=getattr(config, "search_depth", 1),
        forward_search_verbose=getattr(config, "forward_search_verbose", True),
        target_num_questions=getattr(config, "target_num_questions", 15),
        num_mc_samples=getattr(config, "num_mc_samples", 15),
        max_num_samples=getattr(config, "max_num_samples", 50),
        min_num_samples=getattr(config, "min_num_samples", 15),
        threshold_rejection_probability=getattr(config, "threshold_rejection_probability", 0.2),
        belief_state_mode=getattr(config, "belief_state_mode", "uniform"),
        belief_probability_temperature=getattr(config, "belief_probability_temperature", 0.0),
        belief_distribution_num_calls=getattr(config, "belief_distribution_num_calls", 1),
        belief_distribution_permute_history=getattr(config, "belief_distribution_permute_history", False),
        probability_parse_fallback_to_uniform=getattr(config, "probability_parse_fallback_to_uniform", True),
        belief_prior_mode=getattr(config, "belief_prior_mode", "none"),
        belief_prior_exponential_rate=getattr(config, "belief_prior_exponential_rate", 0.0),
        belief_generation_enabled=getattr(config, "belief_generation_enabled", True),
        belief_filtering_enabled=getattr(config, "belief_filtering_enabled", True),
        belief_guess_threshold=getattr(config, "belief_guess_threshold", 0.99),
        answerer_sample_from_prior=getattr(config, "answerer_sample_from_prior", False),
        answerer_prior_mode=getattr(config, "answerer_prior_mode", "inherit"),
        answerer_prior_exponential_rate=getattr(config, "answerer_prior_exponential_rate", None),
        answerer_randomize_prior_order_per_trial=getattr(config, "answerer_randomize_prior_order_per_trial", False),
        answerer_num_prior_trials=getattr(config, "answerer_num_prior_trials", None),
        answerer_prior_seed=getattr(config, "answerer_prior_seed", None),
        strategy_num_retrieved=getattr(config, "animals_strategy_num_retrieved", 2),
        strategy_num_mutation=getattr(config, "animals_strategy_num_mutation", 1),
        strategy_num_crossover=getattr(config, "animals_strategy_num_crossover", 1),
        strategy_num_diverse=getattr(config, "animals_strategy_num_diverse", 2),
        strategy_num_rollouts=getattr(config, "animals_strategy_num_rollouts", 8),
        strategy_planning_depth=getattr(config, "animals_strategy_planning_depth", 8),
        strategy_belief_summary_top_k=getattr(config, "animals_strategy_belief_summary_top_k", 5),
    )


def location_view(config: Any) -> LocationConfig:
    """Project the location-finding-specific fields out of a flat ``Config``."""
    return LocationConfig(
        num_rounds=getattr(config, "location_num_rounds", 20),
        num_trials=getattr(config, "location_num_trials", 1),
        trial_batch_size=getattr(config, "location_trial_batch_size", 1),
        seed=getattr(config, "location_seed", None),
        source_prior=getattr(config, "location_source_prior", "normal"),
        source_radius=getattr(config, "location_source_radius", 1.0),
        num_sources=getattr(config, "location_num_sources", 3),
        dim=getattr(config, "location_dim", 2),
        noise_sd=getattr(config, "location_noise_sd", 0.5),
        signal_model=getattr(config, "location_signal_model", "inverse_square"),
        signal_lengthscale=getattr(config, "location_signal_lengthscale", 0.75),
        signal_amplitude=getattr(config, "location_signal_amplitude", 5.0),
        max_step_radius=getattr(config, "location_max_step_radius", None),
        max_total_beliefs=getattr(config, "location_max_total_beliefs", 1000),
        max_llm_prompt_beliefs=getattr(config, "location_max_llm_prompt_beliefs", 40),
        num_generated_hypotheses=getattr(config, "location_num_generated_hypotheses", 40),
        belief_support_refresh_enabled=getattr(
            config,
            "location_belief_support_refresh_enabled",
            True,
        ),
        candidate_generation_mode=getattr(
            config,
            "location_candidate_generation_mode",
            "llm",
        ),
        target_num_candidates=getattr(config, "location_target_num_candidates", 15),
        search_depth=getattr(config, "location_search_depth", 2),
        eig_quadrature_order=getattr(config, "location_eig_quadrature_order", 15),
        plot_trials=getattr(config, "location_plot_trials", False),
        strategy_num_retrieved=getattr(config, "location_strategy_num_retrieved", 2),
        strategy_num_mutation=getattr(config, "location_strategy_num_mutation", 1),
        strategy_num_crossover=getattr(config, "location_strategy_num_crossover", 1),
        strategy_num_diverse=getattr(config, "location_strategy_num_diverse", 2),
        strategy_num_rollouts=getattr(config, "location_strategy_num_rollouts", 8),
        strategy_planning_depth=getattr(config, "location_strategy_planning_depth", 8),
        strategy_discount_factor=getattr(config, "location_strategy_discount_factor", 1.0),
        strategy_belief_summary_top_k=getattr(config, "location_strategy_belief_summary_top_k", 5),
        strategy_rollout_refresh_hypotheses_each_step=getattr(
            config,
            "location_strategy_rollout_refresh_hypotheses_each_step",
            False,
        ),
        strategy_rollout_scoring_support_mode=getattr(
            config,
            "location_strategy_rollout_scoring_support_mode",
            "union",
        ),
        strategy_rollout_scoring_support_size=getattr(
            config,
            "location_strategy_rollout_scoring_support_size",
            32,
        ),
        strategy_rollout_score_mode=getattr(
            config,
            "location_strategy_rollout_score_mode",
            "start_final_entropy_drop",
        ),
        strategy_rollout_final_refresh_enabled=getattr(
            config,
            "location_strategy_rollout_final_refresh_enabled",
            True,
        ),
        strategy_rollout_query_mode=getattr(
            config,
            "location_strategy_rollout_query_mode",
            "llm_strategy",
        ),
        posterior_mode=getattr(config, "location_posterior_mode", "analytical_likelihood"),
        eig_bounds_enabled=getattr(config, "location_eig_bounds_enabled", False),
        eig_bounds_inner_samples=getattr(config, "location_eig_bounds_inner_samples", 5000),
        eig_bounds_seed=getattr(config, "location_eig_bounds_seed", None),
        eig_bounds_chunk_size=getattr(config, "location_eig_bounds_chunk_size", 8192),
        max_new_tokens=getattr(config, "location_max_new_tokens", None) or getattr(config, "max_model_len", 4096),
        belief_distribution_permute_history=getattr(config, "belief_distribution_permute_history", False),
    )


def paprika_view(config: Any) -> PaprikaConfig:
    """Project Paprika customer-service fields out of the flat config."""
    return PaprikaConfig(
        data_path=getattr(config, "paprika_data_path", None),
        split=getattr(config, "paprika_split", "eval"),
        verify_official_hash=getattr(config, "paprika_verify_official_hash", True),
        num_trials=int(getattr(config, "paprika_num_trials", 5)),
        num_rounds=int(getattr(config, "paprika_num_rounds", 20)),
        trial_batch_size=int(getattr(config, "paprika_trial_batch_size", 1)),
        task_offset=int(getattr(config, "paprika_task_offset", 0)),
        seed=getattr(config, "paprika_seed", None),
        num_hypotheses=int(getattr(config, "paprika_num_hypotheses", 12)),
        num_candidates=int(getattr(config, "paprika_num_candidates", 5)),
        shared_call_cache_enabled=bool(
            getattr(config, "paprika_shared_call_cache_enabled", True)
        ),
        belief_refresh_enabled=bool(
            getattr(config, "paprika_belief_refresh_enabled", True)
        ),
        num_refresh_hypotheses=int(
            getattr(config, "paprika_num_refresh_hypotheses", 6)
        ),
        max_hypotheses=int(getattr(config, "paprika_max_hypotheses", 24)),
    )
