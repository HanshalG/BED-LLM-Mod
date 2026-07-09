from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

from core.belief import BeliefState as _BeliefState, deduped_belief_state, uniform_deduped

if TYPE_CHECKING:
    from model import Model


ReasoningEffort = Literal["low", "medium", "high"]
TaskMode = Literal["animals", "location_finding"]
BeliefStateMode = Literal["uniform", "categorical"]
BeliefPriorMode = Literal["none", "uniform", "exponential_rank"]
AnswererPriorMode = Literal["inherit", "none", "uniform", "exponential_rank"]
LocationPosteriorMode = Literal["analytical_likelihood", "llm_distribution"]
LocationStrategyRolloutScoringSupportMode = Literal["union", "truth_plus_sampled", "truth_start_end", "fixed_common"]
LocationStrategyRolloutScoreMode = Literal["start_final_entropy_drop", "future_step_support_sum"]
LocationStrategyRolloutQueryMode = Literal["llm_strategy", "analytic_eig"]
LocationCandidateGenerationMode = Literal["llm", "support_grid"]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    thinking: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    thinking_max_new_tokens: int | None = None
    thinking_final_max_new_tokens: int | None = None
    use_logprobs: bool = False
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    cuda_visible_devices: str | None = None


@dataclass(frozen=True)
class ModelPair:
    questioner: ModelSpec
    answerer: ModelSpec


@dataclass
class Config:
    version: int = 0
    task: TaskMode = "animals"
    model_pairs: list[ModelPair] = field(default_factory=list)
    method_names: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    animals: list[list[str]] = field(default_factory=list)
    batched_block_size: int = 50
    generation_temperature_diverse: float = 1.0
    generation_temperature_simple: float = 0.7
    answer_temperature: float = 0.7
    search_depth: int = 1
    forward_search_verbose: bool = True
    target_num_questions: int = 15
    num_mc_samples: int = 15
    max_num_samples: int = 50
    min_num_samples: int = 15
    threshold_rejection_probability: float = 0.2
    belief_state_mode: BeliefStateMode = "uniform"
    belief_probability_temperature: float = 0.0
    belief_distribution_num_calls: int = 1
    belief_distribution_permute_history: bool = False
    probability_parse_fallback_to_uniform: bool = True
    belief_prior_mode: BeliefPriorMode = "none"
    belief_prior_exponential_rate: float = 0.0
    belief_generation_enabled: bool = True
    belief_filtering_enabled: bool = True
    belief_guess_threshold: float | None = 0.99
    answerer_sample_from_prior: bool = False
    answerer_prior_mode: AnswererPriorMode = "inherit"
    answerer_prior_exponential_rate: float | None = None
    answerer_randomize_prior_order_per_trial: bool = False
    answerer_num_prior_trials: int | None = None
    answerer_prior_seed: int | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 4096
    run_id: str = ""
    log_path: Path | None = None
    active_prior_animals: list[str] | None = None
    active_answerer_prior_animals: list[str] | None = None
    animals_num_rounds: int = 20
    animals_strategy_num_retrieved: int = 2
    animals_strategy_num_mutation: int = 1
    animals_strategy_num_crossover: int = 1
    animals_strategy_num_diverse: int = 2
    animals_strategy_num_rollouts: int = 8
    animals_strategy_planning_depth: int = 8
    animals_strategy_belief_summary_top_k: int = 5
    location_num_rounds: int = 20
    location_num_trials: int = 1
    location_trial_batch_size: int = 1
    location_seed: int | None = None
    location_source_prior: str = "normal"
    location_source_radius: float = 1.0
    location_num_sources: int = 3
    location_dim: int = 2
    location_noise_sd: float = 0.5
    location_signal_model: str = "inverse_square"
    location_signal_lengthscale: float = 0.75
    location_signal_amplitude: float = 5.0
    location_max_step_radius: float | None = None
    location_max_total_beliefs: int = 1000
    location_max_llm_prompt_beliefs: int = 40
    location_num_generated_hypotheses: int = 0  # 0 = inherit from location_max_llm_prompt_beliefs
    location_belief_support_refresh_enabled: bool = True
    location_candidate_generation_mode: LocationCandidateGenerationMode = "llm"
    location_target_num_candidates: int = 15
    location_search_depth: int = 2
    location_eig_quadrature_order: int = 15
    location_plot_trials: bool = False
    location_strategy_num_retrieved: int = 2
    location_strategy_num_mutation: int = 1
    location_strategy_num_crossover: int = 1
    location_strategy_num_diverse: int = 2
    location_strategy_num_rollouts: int = 8
    location_strategy_planning_depth: int = 8
    location_strategy_discount_factor: float = 1.0
    location_strategy_belief_summary_top_k: int = 5
    location_strategy_rollout_refresh_hypotheses_each_step: bool = False
    location_strategy_rollout_scoring_support_mode: LocationStrategyRolloutScoringSupportMode = "union"
    location_strategy_rollout_scoring_support_size: int = 32
    location_strategy_rollout_score_mode: LocationStrategyRolloutScoreMode = "start_final_entropy_drop"
    location_strategy_rollout_final_refresh_enabled: bool = True
    location_strategy_rollout_query_mode: LocationStrategyRolloutQueryMode = "llm_strategy"
    location_posterior_mode: LocationPosteriorMode = "analytical_likelihood"
    location_eig_bounds_enabled: bool = False
    location_eig_bounds_inner_samples: int = 5000
    location_eig_bounds_seed: int | None = None
    location_eig_bounds_chunk_size: int = 8192
    location_max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.environment:
            self._project_environment_settings()
        # When location_num_generated_hypotheses is left at the sentinel (0), link it to
        # location_max_llm_prompt_beliefs so that direct Config() construction matches the
        # load_config() behaviour of defaulting the two together.
        if self.location_num_generated_hypotheses == 0:
            self.location_num_generated_hypotheses = self.location_max_llm_prompt_beliefs
        if not isinstance(self.location_belief_support_refresh_enabled, bool):
            raise ValueError("location_belief_support_refresh_enabled must be a boolean")
        if self.location_candidate_generation_mode not in {"llm", "support_grid"}:
            raise ValueError(
                "location_candidate_generation_mode must be one of: llm, support_grid"
            )
        if self.location_source_prior not in {"normal", "branch_decoy"}:
            raise ValueError("location_source_prior must be one of: normal, branch_decoy")
        self.location_source_radius = float(self.location_source_radius)
        if not math.isfinite(self.location_source_radius) or self.location_source_radius <= 0.0:
            raise ValueError("location_source_radius must be positive")
        if self.location_signal_model not in {"inverse_square", "local_bump"}:
            raise ValueError("location_signal_model must be one of: inverse_square, local_bump")
        self.location_signal_lengthscale = float(self.location_signal_lengthscale)
        if not math.isfinite(self.location_signal_lengthscale) or self.location_signal_lengthscale <= 0.0:
            raise ValueError("location_signal_lengthscale must be positive")
        self.location_signal_amplitude = float(self.location_signal_amplitude)
        if not math.isfinite(self.location_signal_amplitude) or self.location_signal_amplitude <= 0.0:
            raise ValueError("location_signal_amplitude must be positive")
        if self.location_strategy_rollout_scoring_support_mode not in {
            "union",
            "truth_plus_sampled",
            "truth_start_end",
            "fixed_common",
        }:
            raise ValueError(
                "location_strategy_rollout_scoring_support_mode must be one of: "
                "union, truth_plus_sampled, truth_start_end, fixed_common"
            )
        if self.location_strategy_rollout_scoring_support_size <= 0:
            raise ValueError("location_strategy_rollout_scoring_support_size must be positive")
        if self.location_strategy_rollout_score_mode not in {
            "start_final_entropy_drop",
            "future_step_support_sum",
        }:
            raise ValueError(
                "location_strategy_rollout_score_mode must be one of: "
                "start_final_entropy_drop, future_step_support_sum"
            )
        if (
            self.location_strategy_rollout_scoring_support_mode in {"truth_plus_sampled", "truth_start_end", "fixed_common"}
            and self.location_posterior_mode != "analytical_likelihood"
        ):
            raise ValueError(
                "location_strategy_rollout_scoring_support_mode "
                f"'{self.location_strategy_rollout_scoring_support_mode}' "
                "requires location_posterior_mode='analytical_likelihood'"
            )
        if (
            self.location_strategy_rollout_refresh_hypotheses_each_step
            and self.location_posterior_mode != "analytical_likelihood"
        ):
            raise ValueError(
                "location_strategy_rollout_refresh_hypotheses_each_step requires "
                "location_posterior_mode='analytical_likelihood'"
            )
        if not isinstance(self.location_strategy_rollout_final_refresh_enabled, bool):
            raise ValueError("location_strategy_rollout_final_refresh_enabled must be a boolean")
        if self.location_strategy_rollout_query_mode not in {"llm_strategy", "analytic_eig"}:
            raise ValueError(
                "location_strategy_rollout_query_mode must be one of: "
                "llm_strategy, analytic_eig"
            )
        if self.location_max_step_radius is not None:
            if isinstance(self.location_max_step_radius, bool):
                raise ValueError("location_max_step_radius must be a positive number or null")
            self.location_max_step_radius = float(self.location_max_step_radius)
            if not math.isfinite(self.location_max_step_radius) or self.location_max_step_radius <= 0.0:
                raise ValueError("location_max_step_radius must be a positive number or null")
        self.location_max_new_tokens = self.effective_max_model_len

    @property
    def effective_max_model_len(self) -> int:
        model_lengths = [
            spec.max_model_len
            for pair in self.model_pairs
            for spec in (pair.questioner, pair.answerer)
            if spec.max_model_len is not None
        ]
        if model_lengths:
            return max(model_lengths)
        return self.max_model_len

    def _project_environment_settings(self) -> None:
        """Project nested environment settings onto the current runtime fields.

        The framework-facing config shape is now nested, but the environment
        modules still consume strongly named attributes.  This projection keeps
        those modules simple while making YAML/config ownership environment
        local.
        """
        aliases = _environment_aliases(self.task)
        for key, value in self.environment.items():
            target = aliases.get(key, key)
            if hasattr(self, target):
                setattr(self, target, value)

    @property
    def location_strategy_num_candidates(self) -> int:
        return (self.location_strategy_num_retrieved + self.location_strategy_num_mutation
                + self.location_strategy_num_crossover + self.location_strategy_num_diverse)

    @property
    def base(self):
        from core.config import base_view

        return base_view(self)

    @property
    def animals_config(self):
        from core.config import animals_view

        return animals_view(self)

    @property
    def location_config(self):
        from core.config import location_view

        return location_view(self)


def _normalize_model_spec(raw_spec: object, side_name: str) -> ModelSpec:
    if not isinstance(raw_spec, dict):
        raise ValueError(f"{side_name} must be a mapping with at least a 'model' field")

    model_name = raw_spec.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"{side_name}.model must be a non-empty string")

    thinking = raw_spec.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise ValueError(f"{side_name}.thinking must be a boolean when provided")

    reasoning_effort = raw_spec.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError(f"{side_name}.reasoning_effort must be one of: low, medium, high")

    thinking_max_new_tokens = raw_spec.get("thinking_max_new_tokens")
    if thinking_max_new_tokens is not None and (
        not isinstance(thinking_max_new_tokens, int)
        or isinstance(thinking_max_new_tokens, bool)
        or thinking_max_new_tokens < 1
    ):
        raise ValueError(f"{side_name}.thinking_max_new_tokens must be a positive integer or null")

    thinking_final_max_new_tokens = raw_spec.get("thinking_final_max_new_tokens")
    if thinking_final_max_new_tokens is not None and (
        not isinstance(thinking_final_max_new_tokens, int)
        or isinstance(thinking_final_max_new_tokens, bool)
        or thinking_final_max_new_tokens < 1
    ):
        raise ValueError(f"{side_name}.thinking_final_max_new_tokens must be a positive integer or null")

    use_logprobs = raw_spec.get("use_logprobs", False)
    if not isinstance(use_logprobs, bool):
        raise ValueError(f"{side_name}.use_logprobs must be a boolean when provided")

    tensor_parallel_size = raw_spec.get("tensor_parallel_size")
    if tensor_parallel_size is not None and (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size < 1
    ):
        raise ValueError(f"{side_name}.tensor_parallel_size must be a positive integer or null")

    gpu_memory_utilization = raw_spec.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        if isinstance(gpu_memory_utilization, bool):
            raise ValueError(f"{side_name}.gpu_memory_utilization must be a positive number")
        try:
            gpu_memory_utilization = float(gpu_memory_utilization)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{side_name}.gpu_memory_utilization must be a positive number") from exc
        if not math.isfinite(gpu_memory_utilization) or gpu_memory_utilization <= 0.0:
            raise ValueError(f"{side_name}.gpu_memory_utilization must be a positive number")

    max_model_len = raw_spec.get("max_model_len")
    if max_model_len is not None and (
        not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 1
    ):
        raise ValueError(f"{side_name}.max_model_len must be a positive integer or null")

    cuda_visible_devices = raw_spec.get("cuda_visible_devices")
    if cuda_visible_devices is not None and not isinstance(cuda_visible_devices, str):
        raise ValueError(f"{side_name}.cuda_visible_devices must be a string or null")
    if isinstance(cuda_visible_devices, str) and not [
        device.strip()
        for device in cuda_visible_devices.split(",")
        if device.strip()
    ]:
        raise ValueError(f"{side_name}.cuda_visible_devices must list at least one device")

    vllm_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "cuda_visible_devices": cuda_visible_devices,
    }

    is_qwen = model_name.startswith("Qwen/")
    is_qwen25 = model_name.startswith("Qwen/Qwen2.5")
    is_gemma = model_name.startswith("google/gemma-4")
    is_harmony = model_name.startswith("openai/gpt-oss")
    if is_harmony:
        if thinking is not None:
            raise ValueError(f"{side_name}.thinking is not supported for {model_name}")
        if thinking_max_new_tokens is not None or thinking_final_max_new_tokens is not None:
            raise ValueError(f"{side_name}.thinking budgets are not supported for {model_name}")
        if use_logprobs:
            raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")
        return ModelSpec(
            model=model_name,
            reasoning_effort=reasoning_effort or "low",
            **vllm_kwargs,
        )

    if is_qwen or is_gemma:
        if reasoning_effort is not None:
            raise ValueError(f"{side_name}.reasoning_effort is not supported for {model_name}")
        if use_logprobs and not is_qwen25:
            raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")
        normalized_thinking = False if thinking is None else thinking
        if not normalized_thinking and (
            thinking_max_new_tokens is not None or thinking_final_max_new_tokens is not None
        ):
            raise ValueError(f"{side_name}.thinking budgets require thinking: true")
        return ModelSpec(
            model=model_name,
            thinking=normalized_thinking,
            thinking_max_new_tokens=(
                thinking_max_new_tokens if thinking_max_new_tokens is not None
                else 4096 if normalized_thinking else None
            ),
            thinking_final_max_new_tokens=(
                thinking_final_max_new_tokens if thinking_final_max_new_tokens is not None
                else 512 if normalized_thinking else None
            ),
            use_logprobs=use_logprobs,
            **vllm_kwargs,
        )

    if thinking is not None:
        raise ValueError(f"{side_name}.thinking is only supported for Qwen and Gemma 4 models")
    if reasoning_effort is not None:
        raise ValueError(f"{side_name}.reasoning_effort is only supported for gpt-oss models")
    if thinking_max_new_tokens is not None or thinking_final_max_new_tokens is not None:
        raise ValueError(f"{side_name}.thinking budgets are only supported for Qwen and Gemma 4 models")
    if use_logprobs:
        raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")

    return ModelSpec(model=model_name, **vllm_kwargs)


def _normalize_model_pair(raw_pair: object, index: int) -> ModelPair:
    if not isinstance(raw_pair, dict):
        raise ValueError(f"Each model_pairs entry must be a mapping, got {type(raw_pair).__name__}")

    if "questioner" not in raw_pair or "answerer" not in raw_pair:
        raise ValueError(f"model_pairs[{index}] must contain both 'questioner' and 'answerer'")

    return ModelPair(
        questioner=_normalize_model_spec(raw_pair["questioner"], f"model_pairs[{index}].questioner"),
        answerer=_normalize_model_spec(raw_pair["answerer"], f"model_pairs[{index}].answerer"),
    )


def _read_positive_int(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _read_nonneg_int(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _read_positive_float(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive number")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive number") from exc
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{key} must be a positive number")
    return coerced


def _read_probability(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number in [0.0, 1.0]")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number in [0.0, 1.0]") from exc
    if not math.isfinite(coerced) or not 0.0 <= coerced <= 1.0:
        raise ValueError(f"{key} must be a number in [0.0, 1.0]")
    return coerced


def _environment_aliases(task: str) -> dict[str, str]:
    """Map canonical nested environment keys to runtime attribute names."""
    common_animals = {
        "num_rounds": "animals_num_rounds",
        "strategy_num_retrieved": "animals_strategy_num_retrieved",
        "strategy_num_mutation": "animals_strategy_num_mutation",
        "strategy_num_crossover": "animals_strategy_num_crossover",
        "strategy_num_diverse": "animals_strategy_num_diverse",
        "strategy_num_rollouts": "animals_strategy_num_rollouts",
        "strategy_planning_depth": "animals_strategy_planning_depth",
        "strategy_belief_summary_top_k": "animals_strategy_belief_summary_top_k",
    }
    common_location = {
        "num_rounds": "location_num_rounds",
        "num_trials": "location_num_trials",
        "trial_batch_size": "location_trial_batch_size",
        "seed": "location_seed",
        "source_prior": "location_source_prior",
        "source_radius": "location_source_radius",
        "num_sources": "location_num_sources",
        "dim": "location_dim",
        "noise_sd": "location_noise_sd",
        "signal_model": "location_signal_model",
        "signal_lengthscale": "location_signal_lengthscale",
        "signal_amplitude": "location_signal_amplitude",
        "max_step_radius": "location_max_step_radius",
        "max_total_beliefs": "location_max_total_beliefs",
        "max_llm_prompt_beliefs": "location_max_llm_prompt_beliefs",
        "num_generated_hypotheses": "location_num_generated_hypotheses",
        "belief_support_refresh_enabled": "location_belief_support_refresh_enabled",
        "candidate_generation_mode": "location_candidate_generation_mode",
        "target_num_candidates": "location_target_num_candidates",
        "search_depth": "location_search_depth",
        "eig_quadrature_order": "location_eig_quadrature_order",
        "plot_trials": "location_plot_trials",
        "strategy_num_retrieved": "location_strategy_num_retrieved",
        "strategy_num_mutation": "location_strategy_num_mutation",
        "strategy_num_crossover": "location_strategy_num_crossover",
        "strategy_num_diverse": "location_strategy_num_diverse",
        "strategy_num_rollouts": "location_strategy_num_rollouts",
        "strategy_planning_depth": "location_strategy_planning_depth",
        "strategy_discount_factor": "location_strategy_discount_factor",
        "strategy_belief_summary_top_k": "location_strategy_belief_summary_top_k",
        "strategy_rollout_refresh_hypotheses_each_step": "location_strategy_rollout_refresh_hypotheses_each_step",
        "strategy_rollout_scoring_support_mode": "location_strategy_rollout_scoring_support_mode",
        "strategy_rollout_scoring_support_size": "location_strategy_rollout_scoring_support_size",
        "strategy_rollout_score_mode": "location_strategy_rollout_score_mode",
        "strategy_rollout_final_refresh_enabled": "location_strategy_rollout_final_refresh_enabled",
        "strategy_rollout_query_mode": "location_strategy_rollout_query_mode",
        "posterior_mode": "location_posterior_mode",
        "eig_bounds_enabled": "location_eig_bounds_enabled",
        "eig_bounds_inner_samples": "location_eig_bounds_inner_samples",
        "eig_bounds_seed": "location_eig_bounds_seed",
        "eig_bounds_chunk_size": "location_eig_bounds_chunk_size",
    }
    if task == "animals":
        return common_animals
    if task == "location_finding":
        return common_location
    return {}


def _flatten_environment_config(raw: dict) -> tuple[dict, dict[str, Any]]:
    task = raw.get("task", "animals")
    nested = raw.get("environment", {}) or {}
    if not isinstance(nested, dict):
        raise ValueError("environment must be a mapping when provided")
    flattened = dict(raw)
    aliases = _environment_aliases(str(task))
    for key, value in nested.items():
        flattened[aliases.get(key, key)] = value
    return flattened, dict(nested)


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw, environment = _flatten_environment_config(raw)
    task = raw.get("task", "animals")
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    model_pairs = [
        _normalize_model_pair(pair, index)
        for index, pair in enumerate(raw.get("model_pairs", []))
    ]
    belief_state_mode = raw.get("belief_state_mode", "uniform")
    if belief_state_mode not in {"uniform", "categorical"}:
        raise ValueError("belief_state_mode must be one of: uniform, categorical")
    belief_distribution_num_calls = raw.get("belief_distribution_num_calls", 1)
    if not isinstance(belief_distribution_num_calls, int) or isinstance(belief_distribution_num_calls, bool):
        raise ValueError("belief_distribution_num_calls must be an integer")
    if belief_distribution_num_calls < 1:
        raise ValueError("belief_distribution_num_calls must be at least 1")
    belief_distribution_permute_history = raw.get("belief_distribution_permute_history", False)
    if not isinstance(belief_distribution_permute_history, bool):
        raise ValueError("belief_distribution_permute_history must be a boolean")
    probability_parse_fallback_to_uniform = raw.get("probability_parse_fallback_to_uniform", True)
    if not isinstance(probability_parse_fallback_to_uniform, bool):
        raise ValueError("probability_parse_fallback_to_uniform must be a boolean")
    belief_prior_mode = raw.get("belief_prior_mode", "none")
    if belief_prior_mode not in {"none", "uniform", "exponential_rank"}:
        raise ValueError("belief_prior_mode must be one of: none, uniform, exponential_rank")
    belief_prior_exponential_rate = raw.get("belief_prior_exponential_rate", 0.0)
    if isinstance(belief_prior_exponential_rate, bool):
        raise ValueError("belief_prior_exponential_rate must be a non-negative number")
    try:
        belief_prior_exponential_rate = float(belief_prior_exponential_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("belief_prior_exponential_rate must be a non-negative number") from exc
    if not math.isfinite(belief_prior_exponential_rate) or belief_prior_exponential_rate < 0.0:
        raise ValueError("belief_prior_exponential_rate must be a non-negative number")
    belief_generation_enabled = raw.get("belief_generation_enabled", True)
    if not isinstance(belief_generation_enabled, bool):
        raise ValueError("belief_generation_enabled must be a boolean")
    belief_filtering_enabled = raw.get("belief_filtering_enabled", True)
    if not isinstance(belief_filtering_enabled, bool):
        raise ValueError("belief_filtering_enabled must be a boolean")
    belief_guess_threshold = raw.get("belief_guess_threshold", 0.99)
    if belief_guess_threshold is not None:
        if not isinstance(belief_guess_threshold, (int, float)) or isinstance(belief_guess_threshold, bool):
            raise ValueError("belief_guess_threshold must be a number in [0.0, 1.0] or null")
        belief_guess_threshold = float(belief_guess_threshold)
        if not math.isfinite(belief_guess_threshold) or not 0.0 <= belief_guess_threshold <= 1.0:
            raise ValueError("belief_guess_threshold must be a number in [0.0, 1.0] or null")
    answerer_sample_from_prior = raw.get("answerer_sample_from_prior", False)
    if not isinstance(answerer_sample_from_prior, bool):
        raise ValueError("answerer_sample_from_prior must be a boolean")
    answerer_prior_mode = raw.get("answerer_prior_mode", "inherit")
    if answerer_prior_mode not in {"inherit", "none", "uniform", "exponential_rank"}:
        raise ValueError("answerer_prior_mode must be one of: inherit, none, uniform, exponential_rank")
    answerer_prior_exponential_rate = raw.get("answerer_prior_exponential_rate")
    if answerer_prior_exponential_rate is not None:
        if isinstance(answerer_prior_exponential_rate, bool):
            raise ValueError("answerer_prior_exponential_rate must be a non-negative number or null")
        try:
            answerer_prior_exponential_rate = float(answerer_prior_exponential_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("answerer_prior_exponential_rate must be a non-negative number or null") from exc
        if not math.isfinite(answerer_prior_exponential_rate) or answerer_prior_exponential_rate < 0.0:
            raise ValueError("answerer_prior_exponential_rate must be a non-negative number or null")
    answerer_randomize_prior_order_per_trial = raw.get("answerer_randomize_prior_order_per_trial", False)
    if not isinstance(answerer_randomize_prior_order_per_trial, bool):
        raise ValueError("answerer_randomize_prior_order_per_trial must be a boolean")
    answerer_num_prior_trials = raw.get("answerer_num_prior_trials")
    if answerer_num_prior_trials is not None:
        if not isinstance(answerer_num_prior_trials, int) or isinstance(answerer_num_prior_trials, bool):
            raise ValueError("answerer_num_prior_trials must be a positive integer")
        if answerer_num_prior_trials < 1:
            raise ValueError("answerer_num_prior_trials must be a positive integer")
    answerer_prior_seed = raw.get("answerer_prior_seed")
    if answerer_prior_seed is not None and (
        not isinstance(answerer_prior_seed, int) or isinstance(answerer_prior_seed, bool)
    ):
        raise ValueError("answerer_prior_seed must be an integer or null")
    tensor_parallel_size = raw.get("tensor_parallel_size")
    if tensor_parallel_size is not None and (
        not isinstance(tensor_parallel_size, int) or isinstance(tensor_parallel_size, bool) or tensor_parallel_size < 1
    ):
        raise ValueError("tensor_parallel_size must be a positive integer or null")
    gpu_memory_utilization = raw.get("gpu_memory_utilization", 0.88)
    if isinstance(gpu_memory_utilization, bool):
        raise ValueError("gpu_memory_utilization must be a positive number")
    try:
        gpu_memory_utilization = float(gpu_memory_utilization)
    except (TypeError, ValueError) as exc:
        raise ValueError("gpu_memory_utilization must be a positive number") from exc
    if not math.isfinite(gpu_memory_utilization) or gpu_memory_utilization <= 0.0:
        raise ValueError("gpu_memory_utilization must be a positive number")
    max_model_len = raw.get("max_model_len", 4096)
    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 1:
        raise ValueError("max_model_len must be a positive integer")
    search_depth = _read_positive_int(raw, "search_depth", 1)
    forward_search_verbose = raw.get("forward_search_verbose", True)
    if not isinstance(forward_search_verbose, bool):
        raise ValueError("forward_search_verbose must be a boolean")

    animals_num_rounds = _read_positive_int(raw, "animals_num_rounds", 20)
    animals_strategy_num_retrieved = _read_positive_int(
        raw, "animals_strategy_num_retrieved", raw.get("location_strategy_num_retrieved", 2)
    )
    animals_strategy_num_mutation = _read_nonneg_int(
        raw, "animals_strategy_num_mutation", raw.get("location_strategy_num_mutation", 1)
    )
    animals_strategy_num_crossover = _read_nonneg_int(
        raw, "animals_strategy_num_crossover", raw.get("location_strategy_num_crossover", 1)
    )
    animals_strategy_num_diverse = _read_positive_int(
        raw, "animals_strategy_num_diverse", raw.get("location_strategy_num_diverse", 2)
    )
    animals_strategy_num_rollouts = _read_positive_int(
        raw, "animals_strategy_num_rollouts", raw.get("location_strategy_num_rollouts", 8)
    )
    animals_strategy_planning_depth = _read_positive_int(
        raw, "animals_strategy_planning_depth", raw.get("location_strategy_planning_depth", 8)
    )
    animals_strategy_belief_summary_top_k = _read_positive_int(
        raw, "animals_strategy_belief_summary_top_k", raw.get("location_strategy_belief_summary_top_k", 5)
    )
    location_num_rounds = _read_positive_int(raw, "location_num_rounds", 20)
    location_num_trials = _read_positive_int(raw, "location_num_trials", 1)
    location_trial_batch_size = _read_positive_int(raw, "location_trial_batch_size", 1)
    location_seed = raw.get("location_seed")
    if location_seed is not None and (
        not isinstance(location_seed, int) or isinstance(location_seed, bool)
    ):
        raise ValueError("location_seed must be an integer or null")
    location_source_prior = raw.get("location_source_prior", "normal")
    if location_source_prior not in {"normal", "branch_decoy"}:
        raise ValueError("location_source_prior must be one of: normal, branch_decoy")
    location_source_radius = _read_positive_float(raw, "location_source_radius", 1.0)
    location_num_sources = _read_positive_int(raw, "location_num_sources", 3)
    location_dim = _read_positive_int(raw, "location_dim", 2)
    location_noise_sd = _read_positive_float(raw, "location_noise_sd", 0.5)
    location_signal_model = raw.get("location_signal_model", "inverse_square")
    if location_signal_model not in {"inverse_square", "local_bump"}:
        raise ValueError("location_signal_model must be one of: inverse_square, local_bump")
    location_signal_lengthscale = _read_positive_float(raw, "location_signal_lengthscale", 0.75)
    location_signal_amplitude = _read_positive_float(raw, "location_signal_amplitude", 5.0)
    location_max_step_radius = raw.get("location_max_step_radius")
    if location_max_step_radius is not None:
        if isinstance(location_max_step_radius, bool):
            raise ValueError("location_max_step_radius must be a positive number or null")
        try:
            location_max_step_radius = float(location_max_step_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("location_max_step_radius must be a positive number or null") from exc
        if not math.isfinite(location_max_step_radius) or location_max_step_radius <= 0.0:
            raise ValueError("location_max_step_radius must be a positive number or null")
    location_max_total_beliefs = _read_positive_int(raw, "location_max_total_beliefs", 1000)
    location_max_llm_prompt_beliefs = _read_positive_int(
        raw,
        "location_max_llm_prompt_beliefs",
        raw.get("location_max_beliefs", 40),
    )
    location_num_generated_hypotheses = _read_positive_int(
        raw,
        "location_num_generated_hypotheses",
        location_max_llm_prompt_beliefs,
    )
    location_belief_support_refresh_enabled = raw.get(
        "location_belief_support_refresh_enabled",
        True,
    )
    if not isinstance(location_belief_support_refresh_enabled, bool):
        raise ValueError("location_belief_support_refresh_enabled must be a boolean")
    location_candidate_generation_mode = raw.get("location_candidate_generation_mode", "llm")
    if location_candidate_generation_mode not in {"llm", "support_grid"}:
        raise ValueError(
            "location_candidate_generation_mode must be one of: llm, support_grid"
        )
    location_target_num_candidates = _read_positive_int(raw, "location_target_num_candidates", 15)
    location_search_depth = raw.get("location_search_depth", 2)
    if not isinstance(location_search_depth, int) or isinstance(location_search_depth, bool):
        raise ValueError("location_search_depth must be an integer")
    if location_search_depth not in {1, 2}:
        raise ValueError("location_search_depth must be one of: 1, 2")
    location_eig_quadrature_order = _read_positive_int(raw, "location_eig_quadrature_order", 15)
    location_plot_trials = raw.get("location_plot_trials", False)
    if not isinstance(location_plot_trials, bool):
        raise ValueError("location_plot_trials must be a boolean")
    location_strategy_num_retrieved = _read_positive_int(raw, "location_strategy_num_retrieved", 2)
    location_strategy_num_mutation = _read_nonneg_int(raw, "location_strategy_num_mutation", 1)
    location_strategy_num_crossover = _read_nonneg_int(raw, "location_strategy_num_crossover", 1)
    location_strategy_num_diverse = _read_positive_int(raw, "location_strategy_num_diverse", 2)
    location_strategy_num_rollouts = _read_positive_int(raw, "location_strategy_num_rollouts", 8)
    location_strategy_planning_depth = _read_positive_int(raw, "location_strategy_planning_depth", 8)
    location_strategy_discount_factor = _read_probability(raw, "location_strategy_discount_factor", 1.0)
    location_strategy_belief_summary_top_k = _read_positive_int(raw, "location_strategy_belief_summary_top_k", 5)
    location_strategy_rollout_refresh_hypotheses_each_step = raw.get(
        "location_strategy_rollout_refresh_hypotheses_each_step",
        False,
    )
    if not isinstance(location_strategy_rollout_refresh_hypotheses_each_step, bool):
        raise ValueError("location_strategy_rollout_refresh_hypotheses_each_step must be a boolean")
    location_strategy_rollout_scoring_support_mode = raw.get(
        "location_strategy_rollout_scoring_support_mode",
        "union",
    )
    if location_strategy_rollout_scoring_support_mode not in {"union", "truth_plus_sampled", "truth_start_end", "fixed_common"}:
        raise ValueError(
            "location_strategy_rollout_scoring_support_mode must be one of: "
            "union, truth_plus_sampled, truth_start_end, fixed_common"
        )
    location_strategy_rollout_scoring_support_size = _read_positive_int(
        raw,
        "location_strategy_rollout_scoring_support_size",
        32,
    )
    location_strategy_rollout_score_mode = raw.get(
        "location_strategy_rollout_score_mode",
        "start_final_entropy_drop",
    )
    if location_strategy_rollout_score_mode not in {"start_final_entropy_drop", "future_step_support_sum"}:
        raise ValueError(
            "location_strategy_rollout_score_mode must be one of: "
            "start_final_entropy_drop, future_step_support_sum"
        )
    location_strategy_rollout_final_refresh_enabled = raw.get(
        "location_strategy_rollout_final_refresh_enabled",
        True,
    )
    if not isinstance(location_strategy_rollout_final_refresh_enabled, bool):
        raise ValueError("location_strategy_rollout_final_refresh_enabled must be a boolean")
    location_strategy_rollout_query_mode = raw.get(
        "location_strategy_rollout_query_mode",
        "llm_strategy",
    )
    if location_strategy_rollout_query_mode not in {"llm_strategy", "analytic_eig"}:
        raise ValueError(
            "location_strategy_rollout_query_mode must be one of: "
            "llm_strategy, analytic_eig"
        )
    location_posterior_mode = raw.get("location_posterior_mode", "analytical_likelihood")
    if location_posterior_mode not in {"analytical_likelihood", "llm_distribution"}:
        raise ValueError("location_posterior_mode must be one of: analytical_likelihood, llm_distribution")
    if (
        location_strategy_rollout_scoring_support_mode in {"truth_plus_sampled", "truth_start_end", "fixed_common"}
        and location_posterior_mode != "analytical_likelihood"
    ):
        raise ValueError(
            "location_strategy_rollout_scoring_support_mode "
            f"'{location_strategy_rollout_scoring_support_mode}' "
            "requires location_posterior_mode='analytical_likelihood'"
        )
    if (
        location_strategy_rollout_refresh_hypotheses_each_step
        and location_posterior_mode != "analytical_likelihood"
    ):
        raise ValueError(
            "location_strategy_rollout_refresh_hypotheses_each_step requires "
            "location_posterior_mode='analytical_likelihood'"
        )
    location_eig_bounds_enabled = raw.get("location_eig_bounds_enabled", False)
    if not isinstance(location_eig_bounds_enabled, bool):
        raise ValueError("location_eig_bounds_enabled must be a boolean")
    location_eig_bounds_inner_samples = _read_positive_int(
        raw,
        "location_eig_bounds_inner_samples",
        5000,
    )
    location_eig_bounds_seed = raw.get("location_eig_bounds_seed")
    if location_eig_bounds_seed is not None and (
        not isinstance(location_eig_bounds_seed, int) or isinstance(location_eig_bounds_seed, bool)
    ):
        raise ValueError("location_eig_bounds_seed must be an integer or null")
    location_eig_bounds_chunk_size = _read_positive_int(
        raw,
        "location_eig_bounds_chunk_size",
        8192,
    )
    method_names = raw.get("method_names", raw.get("extraction_methods", []))
    if task == "location_finding" and not method_names:
        method_names = ["EIG"]
    return Config(
        version = raw.get("version", 0),
        task = task,
        model_pairs = model_pairs,
        method_names = method_names,
        environment = environment,
        animals = raw.get("animals", []),
        batched_block_size = raw.get("batched_block_size", 50),
        generation_temperature_diverse = raw.get("generation_temperature_diverse", 1.0),
        generation_temperature_simple = raw.get("generation_temperature_simple", 0.7),
        answer_temperature = raw.get("answer_temperature", 0.7),
        search_depth = search_depth,
        forward_search_verbose = forward_search_verbose,
        target_num_questions = raw.get("target_num_questions", 15),
        num_mc_samples = _read_positive_int(raw, "num_mc_samples", 15),
        max_num_samples = raw.get("max_num_samples", 50),
        min_num_samples = raw.get("min_num_samples", 15),
        threshold_rejection_probability = raw.get("threshold_rejection_probability", 0.2),
        belief_state_mode = belief_state_mode,
        belief_probability_temperature = raw.get("belief_probability_temperature", 0.0),
        belief_distribution_num_calls = belief_distribution_num_calls,
        belief_distribution_permute_history = belief_distribution_permute_history,
        probability_parse_fallback_to_uniform = probability_parse_fallback_to_uniform,
        belief_prior_mode = belief_prior_mode,
        belief_prior_exponential_rate = belief_prior_exponential_rate,
        belief_generation_enabled = belief_generation_enabled,
        belief_filtering_enabled = belief_filtering_enabled,
        belief_guess_threshold = belief_guess_threshold,
        answerer_sample_from_prior = answerer_sample_from_prior,
        answerer_prior_mode = answerer_prior_mode,
        answerer_prior_exponential_rate = answerer_prior_exponential_rate,
        answerer_randomize_prior_order_per_trial = answerer_randomize_prior_order_per_trial,
        answerer_num_prior_trials = answerer_num_prior_trials,
        answerer_prior_seed = answerer_prior_seed,
        tensor_parallel_size = tensor_parallel_size,
        gpu_memory_utilization = gpu_memory_utilization,
        max_model_len = max_model_len,
        animals_num_rounds = animals_num_rounds,
        animals_strategy_num_retrieved = animals_strategy_num_retrieved,
        animals_strategy_num_mutation = animals_strategy_num_mutation,
        animals_strategy_num_crossover = animals_strategy_num_crossover,
        animals_strategy_num_diverse = animals_strategy_num_diverse,
        animals_strategy_num_rollouts = animals_strategy_num_rollouts,
        animals_strategy_planning_depth = animals_strategy_planning_depth,
        animals_strategy_belief_summary_top_k = animals_strategy_belief_summary_top_k,
        location_num_rounds = location_num_rounds,
        location_num_trials = location_num_trials,
        location_trial_batch_size = location_trial_batch_size,
        location_seed = location_seed,
        location_source_prior = location_source_prior,
        location_source_radius = location_source_radius,
        location_num_sources = location_num_sources,
        location_dim = location_dim,
        location_noise_sd = location_noise_sd,
        location_signal_model = location_signal_model,
        location_signal_lengthscale = location_signal_lengthscale,
        location_signal_amplitude = location_signal_amplitude,
        location_max_step_radius = location_max_step_radius,
        location_max_total_beliefs = location_max_total_beliefs,
        location_max_llm_prompt_beliefs = location_max_llm_prompt_beliefs,
        location_num_generated_hypotheses = location_num_generated_hypotheses,
        location_belief_support_refresh_enabled = location_belief_support_refresh_enabled,
        location_candidate_generation_mode = location_candidate_generation_mode,
        location_target_num_candidates = location_target_num_candidates,
        location_search_depth = location_search_depth,
        location_eig_quadrature_order = location_eig_quadrature_order,
        location_plot_trials = location_plot_trials,
        location_strategy_num_retrieved = location_strategy_num_retrieved,
        location_strategy_num_mutation = location_strategy_num_mutation,
        location_strategy_num_crossover = location_strategy_num_crossover,
        location_strategy_num_diverse = location_strategy_num_diverse,
        location_strategy_num_rollouts = location_strategy_num_rollouts,
        location_strategy_planning_depth = location_strategy_planning_depth,
        location_strategy_discount_factor = location_strategy_discount_factor,
        location_strategy_belief_summary_top_k = location_strategy_belief_summary_top_k,
        location_strategy_rollout_refresh_hypotheses_each_step = location_strategy_rollout_refresh_hypotheses_each_step,
        location_strategy_rollout_scoring_support_mode = location_strategy_rollout_scoring_support_mode,
        location_strategy_rollout_scoring_support_size = location_strategy_rollout_scoring_support_size,
        location_strategy_rollout_score_mode = location_strategy_rollout_score_mode,
        location_strategy_rollout_final_refresh_enabled = location_strategy_rollout_final_refresh_enabled,
        location_strategy_rollout_query_mode = location_strategy_rollout_query_mode,
        location_posterior_mode = location_posterior_mode,
        location_eig_bounds_enabled = location_eig_bounds_enabled,
        location_eig_bounds_inner_samples = location_eig_bounds_inner_samples,
        location_eig_bounds_seed = location_eig_bounds_seed,
        location_eig_bounds_chunk_size = location_eig_bounds_chunk_size,
        location_max_new_tokens = None,
    )


def build_models(
    model_pairs: list[ModelPair],
    build_model_adapter: Callable[[ModelSpec], "Model"],
    *,
    roles: Sequence[str] = ("questioner", "answerer"),
) -> dict[ModelSpec, "Model"]:
    model_specs: list[ModelSpec] = []
    seen_specs: set[ModelSpec] = set()
    role_names = tuple(dict.fromkeys(roles))
    for pair in model_pairs:
        for role in role_names:
            spec = getattr(pair, role)
            if spec not in seen_specs:
                model_specs.append(spec)
                seen_specs.add(spec)

    return {
        spec: build_model_adapter(spec)
        for spec in model_specs
    }


def _model_spec_stem(spec: ModelSpec) -> str:
    parts = [spec.model.replace("/", "_")]
    if spec.reasoning_effort is not None:
        parts.append(f"reasoning-{spec.reasoning_effort}")
    if spec.thinking is not None:
        parts.append(f"thinking-{'on' if spec.thinking else 'off'}")
    if spec.use_logprobs:
        parts.append("logprobs-on")
    return "__".join(parts)


def build_output_stem(
    run_id: str,
    method_name: str,
    questioner: ModelSpec,
    answerer: ModelSpec,
    version: int,
    belief_state_mode: BeliefStateMode = "uniform",
    search_depth: int = 1,
) -> str:
    return (
        f"{run_id}_{method_name}_Q:{_model_spec_stem(questioner)},"
        f"A:{_model_spec_stem(answerer)}_{belief_state_mode}_depth-{search_depth}_{version}_animals"
    )


def resolve_run_id() -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return slurm_job_id

    return datetime.now().strftime("%Y%m%dT%H%M%S")


def write_to_log(message: str, config: Config) -> None:
    if config.log_path is None:
        raise ValueError("config.log_path must be set before logging")

    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("a", encoding="utf-8") as file:
        file.write(message)


def print_and_log(message: str, config: Config) -> None:
    print(message)
    if config.log_path is not None:
        write_to_log(f"{message}\n", config)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def format_config_for_log(config: Config) -> str:
    return json.dumps(_json_ready(asdict(config)), indent=2, sort_keys=True)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_last_balanced_json_object(text: str) -> str | None:
    """Return the last top-level balanced JSON object in text (ignoring code fences)."""
    stripped = _strip_code_fences(text)
    # Collect start indices of all top-level '{' characters
    candidates: list[int] = []
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
        elif char == "{":
            if depth == 0:
                candidates.append(idx)
            depth += 1
        elif char == "}":
            depth -= 1
    # Try to parse from the last top-level '{' backwards
    for start_idx in reversed(candidates):
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start_idx, len(stripped)):
            char = stripped[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start_idx:idx + 1]
    return None


def _extract_first_balanced_json_object(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    start_idx: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(stripped):
        if start_idx is None:
            if char == "{":
                start_idx = idx
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start_idx:idx + 1]

    return None


def _repair_labeled_distribution_json_text(text: str) -> str:
    # Gemma sometimes emits doubled quotes before object keys, e.g. `"h2":0,""h3":1`.
    repaired = re.sub(r'(?<=[{,])\s*""([^"]+)":', r'"\1":', text)
    repaired = re.sub(r',\s*"+\s*([}\]])', r"\1", repaired)
    return re.sub(r",\s*([}\]])", r"\1", repaired)


def _distribution_payload_from_json(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    for key in ("weights", "probabilities", "posterior", "distribution"):
        nested_payload = payload.get(key)
        if nested_payload is not None:
            return nested_payload
    return payload


def _normalize_labeled_distribution_response(response_text: str, labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}

    normalized_text = _strip_code_fences(response_text)
    try:
        payload = json.loads(normalized_text)
    except (json.JSONDecodeError, TypeError) as exc:
        balanced_payload = _extract_last_balanced_json_object(response_text)
        candidate_payloads = [
            candidate
            for candidate in (balanced_payload, _repair_labeled_distribution_json_text(normalized_text))
            if candidate is not None
        ]
        for candidate_payload in candidate_payloads:
            try:
                payload = json.loads(candidate_payload)
                break
            except (json.JSONDecodeError, TypeError):
                repaired_payload = _repair_labeled_distribution_json_text(candidate_payload)
                try:
                    payload = json.loads(repaired_payload)
                    break
                except (json.JSONDecodeError, TypeError):
                    continue
        else:
            raise ValueError(f"Invalid probability JSON: {response_text!r}") from exc

    payload = _distribution_payload_from_json(payload)

    scores: list[float] = []
    if isinstance(payload, dict):
        for label in labels:
            raw_value = payload.get(label, 0.0)
            try:
                score = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Probability for {label!r} must be numeric: {raw_value!r}") from exc
            if not math.isfinite(score) or score < 0.0:
                raise ValueError(f"Probability for {label!r} must be finite and non-negative: {raw_value!r}")
            scores.append(score)
    elif isinstance(payload, list) and len(payload) == len(labels):
        for index, raw_value in enumerate(payload):
            try:
                score = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Probability at index {index} must be numeric: {raw_value!r}") from exc
            if not math.isfinite(score) or score < 0.0:
                raise ValueError(f"Probability at index {index} must be finite and non-negative: {raw_value!r}")
            scores.append(score)
    else:
        raise ValueError(f"Probability response must be a JSON object or {len(labels)}-element list: {response_text!r}")

    total = sum(scores)
    if total <= 0.0:
        raise ValueError(f"Probability response must contain a positive total weight: {response_text!r}")

    return {
        label: score / total
        for label, score in zip(labels, scores)
    }


def _normalize_probability_response(response_text: str, responses: list[str]) -> dict[str, float]:
    return _normalize_labeled_distribution_response(response_text, responses)


def _uniform_probability_response(responses: list[str]) -> dict[str, float]:
    if not responses:
        return {}

    probability = 1.0 / len(responses)
    return {
        response: probability
        for response in responses
    }


def _probability_results_from_messages(batch_messages: list[list[dict[str, str]]], responses: list[str], block_size: int,
                                       temperature: float,
                                       complete_messages_batched: Callable[..., list[str]],
                                       fallback_to_uniform: bool = False,
                                       max_new_tokens: int | None = None) -> list[dict[str, float]]:
    from environments.animals.prompts import is_answer_likelihood_messages

    # Validate that every message list is in answer-likelihood format, then shallow-copy
    # so downstream mutation is safe.
    for messages in batch_messages:
        if not is_answer_likelihood_messages(list(messages)):
            raise ValueError(
                "chat_probabilities_messages_batched requires dedicated answer likelihood messages. "
                "Build conversations with answer_likelihood_messages(...)."
            )
    probability_messages = [[dict(message) for message in messages] for messages in batch_messages]
    results: list[dict[str, float] | None] = [None] * len(probability_messages)
    pending_indices = list(range(len(probability_messages)))
    raw_completions: dict[int, str] = {}

    for _attempt in range(3):
        if not pending_indices:
            break

        completions = complete_messages_batched(
            [probability_messages[index] for index in pending_indices],
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )

        if len(completions) != len(pending_indices):
            raise ValueError(
                f"Expected {len(pending_indices)} probability completions, received {len(completions)}"
            )

        failed_indices: list[int] = []
        for index, completion in zip(pending_indices, completions):
            raw_completions[index] = completion
            try:
                results[index] = _normalize_probability_response(completion, responses)
            except ValueError:
                failed_indices.append(index)

        pending_indices = failed_indices

    if pending_indices:
        if fallback_to_uniform:
            for index in pending_indices:
                failed_completion = raw_completions.get(index, "")
                print(
                    f"Failed to parse probability JSON for index {index}, "
                    f"assigning uniform probabilities {failed_completion!r}"
                )
                results[index] = _uniform_probability_response(responses)
        else:
            failure_details = ", ".join(
                f"{index}: {raw_completions.get(index, '')!r}"
                for index in pending_indices
            )
            raise ValueError(
                "Failed to parse probability JSON after 3 attempts for "
                f"{len(pending_indices)} item(s): {failure_details}"
            )

    return [result for result in results if result is not None]


def _distribution_from_messages(messages: list[dict[str, str]], labels: list[str], temperature: float,
                                complete_message: Callable[..., list[str]],
                                num_calls: int = 1,
                                fallback_to_uniform: bool = False) -> dict[str, float]:
    distribution, _valid_count = _distribution_with_valid_count_from_messages(
        messages,
        labels,
        temperature,
        complete_message,
        num_calls=num_calls,
        fallback_to_uniform=fallback_to_uniform,
    )
    return distribution


def _distribution_with_valid_count_from_messages(messages: list[dict[str, str]], labels: list[str], temperature: float,
                                                 complete_message: Callable[..., list[str]],
                                                 num_calls: int = 1,
                                                 fallback_to_uniform: bool = False) -> tuple[dict[str, float], int]:
    if not labels:
        return {}, 0
    if num_calls < 1:
        raise ValueError("num_calls must be at least 1")

    completions = complete_message(messages=messages, temperature=temperature, num_responses=num_calls)
    if len(completions) != num_calls:
        raise ValueError(f"Expected {num_calls} distribution completions, received {len(completions)}")

    return _average_labeled_distributions_from_completions(
        completions,
        labels,
        fallback_to_uniform=fallback_to_uniform,
    )


def _distribution_with_valid_count_from_batched_messages(batch_messages: list[list[dict[str, str]]], labels: list[str],
                                                         temperature: float,
                                                         complete_messages_batched: Callable[..., list[str]],
                                                         block_size: int,
                                                         max_new_tokens: int = 8192,
                                                         fallback_to_uniform: bool = False) -> tuple[dict[str, float], int]:
    if not labels:
        return {}, 0
    if not batch_messages:
        raise ValueError("batch_messages must contain at least one prompt")

    completions = complete_messages_batched(
        batch_messages=batch_messages,
        temperature=temperature,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
    )
    if len(completions) != len(batch_messages):
        raise ValueError(
            f"Expected {len(batch_messages)} batched distribution completions, received {len(completions)}"
        )

    return _average_labeled_distributions_from_completions(
        completions,
        labels,
        fallback_to_uniform=fallback_to_uniform,
    )


def _average_labeled_distributions_from_completions(completions: list[str], labels: list[str],
                                                    fallback_to_uniform: bool = False,
                                                    fallback_distribution: dict[str, float] | None = None
                                                    ) -> tuple[dict[str, float], int]:
    completion_count = len(completions)
    valid_distributions: list[dict[str, float]] = []
    failed_completions: list[str] = []

    for completion in completions:
        try:
            valid_distributions.append(_normalize_labeled_distribution_response(completion, labels))
        except ValueError:
            failed_completions.append(completion)

    if valid_distributions:
        averaged_distribution = {
            label: sum(distribution[label] for distribution in valid_distributions) / len(valid_distributions)
            for label in labels
        }
        return averaged_distribution, len(valid_distributions)

    if fallback_distribution is not None:
        print(
            f"Failed to parse belief distribution JSON for all {completion_count} completion(s), "
            f"assigning fallback distribution from completions {failed_completions!r}"
        )
        return fallback_distribution, 0

    if fallback_to_uniform:
        print(
            f"Failed to parse belief distribution JSON for all {completion_count} completion(s), "
            f"assigning uniform distribution from completions {failed_completions!r}"
        )
        return _uniform_probability_response(labels), 0

    raise ValueError(
        f"Failed to parse belief distribution JSON for all {completion_count} completion(s): {failed_completions!r}"
    )

# prompts ask to generate collection of entities, one on each line --> convert the returned string to an array
def convert_string_to_array(response):
    return [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]


_BELIEF_MAX_LENGTH = 80
_BELIEF_EXPLANATION_PATTERN = re.compile(
    r"\b("
    r"because|however|therefore|based on|provided clues|previous answers|"
    r"contradiction|fit these geographic exclusions|logic of the previous answers|"
    r"there are no naturally occurring"
    r")\b",
    flags=re.IGNORECASE,
)
_BELIEF_REASONING_PAREN_PATTERN = re.compile(
    r"\((?:yes|no|incorrect|has|because|but|not)\b",
    flags=re.IGNORECASE,
)
_BELIEF_QUESTION_LIKE_PATTERN = re.compile(
    r"^(?:is|are|was|were|does|do|did|can|could|should|would|will)\b",
    flags=re.IGNORECASE,
)


def normalize_belief_label(raw_belief: str) -> str | None:
    cleaned_belief = re.sub(r"\s+", " ", raw_belief.strip())
    if not cleaned_belief:
        return None
    if any(char in cleaned_belief for char in "{}[];=&/\\"):
        return None
    if re.search(r"\bnew\s+\w+", cleaned_belief, flags=re.IGNORECASE):
        return None

    cleaned_belief = re.sub(r"\s*\([^)]*\)", "", cleaned_belief)
    cleaned_belief = re.sub(r"\s+", " ", cleaned_belief).strip()
    cleaned_belief = cleaned_belief.strip(" -:;,.")
    if not cleaned_belief:
        return None

    if len(cleaned_belief) > _BELIEF_MAX_LENGTH:
        return None
    if any(char in cleaned_belief for char in "{}[];=&/\\"):
        return None
    if "->" in cleaned_belief or "?" in cleaned_belief:
        return None
    if _BELIEF_QUESTION_LIKE_PATTERN.match(cleaned_belief):
        return None
    if _BELIEF_EXPLANATION_PATTERN.search(cleaned_belief):
        return None
    if _BELIEF_REASONING_PAREN_PATTERN.search(cleaned_belief):
        return None
    if cleaned_belief.count(",") >= 3:
        return None

    return cleaned_belief


def clean_generated_belief_labels(raw_beliefs: list[str]) -> list[str]:
    cleaned_beliefs: list[str] = []
    for raw_belief in raw_beliefs:
        cleaned_belief = normalize_belief_label(raw_belief)
        if cleaned_belief is not None:
            cleaned_beliefs.append(cleaned_belief)
    return cleaned_beliefs


def _binary_entropy(p_yes: float, p_no: float) -> float:
    p_yes_clipped = max(p_yes, 1e-12)
    p_no_clipped = max(p_no, 1e-12)
    return - (p_yes_clipped * np.log(p_yes_clipped) + p_no_clipped * np.log(p_no_clipped))


def _normalize_categorical_label(label: str) -> str | None:
    cleaned_label = label.strip()
    return cleaned_label or None


def build_uniform_prior(animals: list[str]) -> _BeliefState:
    return uniform_deduped(
        animals,
        key=lambda label: label.lower(),
        normalize=_normalize_categorical_label,
    )


def build_exponential_rank_prior(animals: list[str], rate: float) -> _BeliefState:
    if isinstance(rate, bool):
        raise ValueError("rate must be a non-negative finite number")
    try:
        numeric_rate = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate must be a non-negative finite number") from exc
    if not math.isfinite(numeric_rate) or numeric_rate < 0.0:
        raise ValueError("rate must be a non-negative finite number")

    deduped_animals = build_uniform_prior(animals).hypotheses
    if not deduped_animals:
        return _BeliefState()

    weights = [
        math.exp(-numeric_rate * index)
        for index in range(len(deduped_animals))
    ]
    return deduped_belief_state(
        deduped_animals,
        weights,
        key=lambda label: label.lower(),
        normalize=_normalize_categorical_label,
        fallback_to_uniform=False,
    )


def _prior_animals_for_config(config: Config, active_animals: list[str] | None) -> list[str]:
    if active_animals is not None:
        return active_animals
    if config.version < 0 or config.version >= len(config.animals):
        raise ValueError("config.version must select an animals entry before building a prior")
    return config.animals[config.version]


def _build_prior_from_mode(animals: list[str], mode: str, rate: float | None, field_name: str) -> _BeliefState | None:
    if mode == "none":
        return None
    if mode == "uniform":
        return build_uniform_prior(animals)
    if mode == "exponential_rank":
        return build_exponential_rank_prior(animals, 0.0 if rate is None else rate)
    raise ValueError(f"{field_name} must be one of: none, uniform, exponential_rank")


def get_questioner_prior(config: Config) -> _BeliefState | None:
    if config.belief_prior_mode == "none":
        return None
    prior_animals = _prior_animals_for_config(config, config.active_prior_animals)
    return _build_prior_from_mode(
        prior_animals,
        config.belief_prior_mode,
        config.belief_prior_exponential_rate,
        "belief_prior_mode",
    )


def get_answerer_prior(config: Config) -> _BeliefState | None:
    if config.answerer_prior_mode == "inherit":
        if config.belief_prior_mode == "none":
            return None
        prior_animals = _prior_animals_for_config(config, config.active_answerer_prior_animals)
        return _build_prior_from_mode(
            prior_animals,
            config.belief_prior_mode,
            config.belief_prior_exponential_rate,
            "belief_prior_mode",
        )
    if config.answerer_prior_mode == "none":
        return None
    prior_animals = _prior_animals_for_config(config, config.active_answerer_prior_animals)
    return _build_prior_from_mode(
        prior_animals,
        config.answerer_prior_mode,
        config.answerer_prior_exponential_rate,
        "answerer_prior_mode",
    )


def get_configured_prior(config: Config) -> _BeliefState | None:
    return get_questioner_prior(config)


def sort_belief_state_descending(belief_state: _BeliefState) -> _BeliefState:
    return belief_state.sorted_descending()


def format_belief_state(belief_state: _BeliefState, top_n: int | None = None) -> str:
    if len(belief_state.hypotheses) == 0:
        return "[]"

    entries = list(zip(belief_state.hypotheses, belief_state.probabilities))
    if top_n is not None:
        entries = sorted(entries, key=lambda entry: entry[1], reverse=True)[:top_n]

    formatted_entries = [
        f"{belief} ({probability:.3f})"
        for belief, probability in entries
    ]
    return "[" + ", ".join(formatted_entries) + "]"


def format_categorical_belief_summary(belief_state: _BeliefState, top_n: int | None = None) -> str:
    if top_n is None:
        return f"{len(belief_state.hypotheses)} belief(s): {format_belief_state(belief_state)}"

    top_count = min(top_n, len(belief_state.hypotheses))
    return f"{len(belief_state.hypotheses)} belief(s): {format_belief_state(belief_state, top_n=top_count)}"


def is_uniform_belief_state(belief_state: _BeliefState, tolerance: float = 1e-9) -> bool:
    if len(belief_state.hypotheses) <= 1:
        return True

    uniform_probability = 1.0 / len(belief_state.hypotheses)
    return all(
        math.isclose(probability, uniform_probability, rel_tol=tolerance, abs_tol=tolerance)
        for probability in belief_state.probabilities
    )


# reverses a messages array so that the final question comes first
def reverse_history(history_questioner: list[dict[str,str]]) -> list[dict[str,str]]:
    blocks = split_history_into_qa_blocks(history_questioner)
    return [x for b in blocks[::-1] for x in b]


def split_history_into_qa_blocks(history_questioner: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    return [history_questioner[i:i + 2] for i in range(0, len(history_questioner), 2)]


def sample_permuted_history_messages(history_questioner: list[dict[str, str]], num_samples: int) -> list[list[dict[str, str]]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    blocks = split_history_into_qa_blocks(history_questioner)
    if len(blocks) <= 1:
        return [[dict(message) for message in history_questioner] for _ in range(num_samples)]

    permuted_histories: list[list[dict[str, str]]] = []
    for _ in range(num_samples):
        permutation = np.random.permutation(len(blocks))
        permuted_histories.append([
            dict(message)
            for block_index in permutation
            for message in blocks[block_index]
        ])

    return permuted_histories


def get_question_answered(question: str, goal_object: str, answerer: Model, answer_temperature: float) -> str:
    from environments.animals.prompts import answer_question_yesnocorrect_system_prompt

    user_question = {"role": "user", "content": f"{question}"}
    messages = [answer_question_yesnocorrect_system_prompt(entity=goal_object), user_question]
    return answerer.chat_complete(messages=messages, temperature=answer_temperature)[0]


def is_guess_correct_via_answerer(guess: str, goal_object: str, answerer: Model, answer_temperature: float) -> bool:
    return get_question_answered(
        f"Is it {guess}?",
        goal_object,
        answerer,
        answer_temperature,
    ) == "Correct!"


def generate_original_beliefs(questioner: Model, config: Config) -> list[str]:
    from environments.animals.prompts import generate_original_animals_system_prompt

    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    user_question = {"role": "user", "content": f"Let\'s start the game of 20 questions. Generate a diverse "
                                                f"set of animals, at least {min_num_samples}."}
    messages = [generate_original_animals_system_prompt(max_num_samples), user_question]
    new_beliefs = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    return convert_string_to_array(new_beliefs)
