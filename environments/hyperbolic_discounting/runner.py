"""Hyperbolic temporal discounting environment (continuous observation surrogate)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from helpers import (
    Config,
    _average_labeled_distributions_from_completions,
    print_and_log,
    write_to_log,
)

if TYPE_CHECKING:
    from model import Model


@dataclass(frozen=True)
class HyperbolicParams:
    k: float
    alpha: float

    def __post_init__(self) -> None:
        if self.k <= 0.0 or self.alpha <= 0.0:
            raise ValueError("k and alpha must be positive")
        if not math.isfinite(self.k) or not math.isfinite(self.alpha):
            raise ValueError("k and alpha must be finite")


@dataclass(frozen=True)
class HyperbolicDesign:
    immediate_reward: float
    delayed_reward: float
    days: int

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError("days must be a positive integer")


@dataclass(frozen=True)
class HyperbolicObservation:
    design: HyperbolicDesign
    value: float


@dataclass(frozen=True)
class HyperbolicBeliefState:
    hypotheses: list[HyperbolicParams] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.hypotheses) != len(self.probabilities):
            raise ValueError("HyperbolicBeliefState hypotheses and probabilities must have the same length")


@dataclass(frozen=True)
class HyperbolicFindingMetrics:
    parameter_rmse: list[float]
    k_rmse: list[float]
    top_probability: list[float]
    selected_eig: list[float]
    implied_choice_accuracy: list[float] = field(default_factory=list)


@dataclass
class _HyperbolicTrialState:
    trial_idx: int
    env: HyperbolicDiscountingEnv
    observations: list[HyperbolicObservation]
    rng: np.random.Generator
    belief_state: HyperbolicBeliefState | None = None
    final_estimate: HyperbolicParams | None = None
    final_rmse: float = float("inf")


class HyperbolicDiscountingEnv:
    def __init__(
        self,
        *,
        noise_sd: float,
        true_params: HyperbolicParams | None = None,
        rng: np.random.Generator | None = None,
        k_mean: float = 0.0,
        k_std: float = 1.0,
        alpha_scale: float = 1.0,
    ) -> None:
        self.noise_sd = noise_sd
        self.k_mean = k_mean
        self.k_std = k_std
        self.alpha_scale = alpha_scale
        self.rng = rng or np.random.default_rng()
        self.observed_data: list[HyperbolicObservation] = []
        self.true_params = HyperbolicParams(k=1.0, alpha=1.0)
        self.reset(true_params=true_params)

    def reset(self, true_params: HyperbolicParams | None = None) -> None:
        self.observed_data = []
        if true_params is None:
            log_k = float(self.rng.normal(self.k_mean, self.k_std))
            k = math.exp(log_k)
            alpha = abs(float(self.rng.normal(0.0, self.alpha_scale)))
            if alpha <= 0.0:
                alpha = self.alpha_scale
            self.true_params = HyperbolicParams(k=k, alpha=alpha)
        else:
            self.true_params = true_params

    def run_experiment(self, design: HyperbolicDesign) -> HyperbolicObservation:
        mean = latent_mean(design, self.true_params)
        value = float(self.rng.normal(mean, self.noise_sd))
        result = HyperbolicObservation(design=design, value=round(value, 4))
        self.observed_data.append(result)
        return result


def normalize_design(
    immediate_reward: object,
    delayed_reward: object,
    days: object,
) -> HyperbolicDesign:
    try:
        ir = float(immediate_reward)
        dr = float(delayed_reward)
        day_count = int(days)
    except (TypeError, ValueError) as exc:
        raise ValueError("design must contain numeric iR, dR, and integer days") from exc
    if not math.isfinite(ir) or not math.isfinite(dr):
        raise ValueError("reward values must be finite")
    return HyperbolicDesign(immediate_reward=ir, delayed_reward=dr, days=day_count)


def latent_mean(design: HyperbolicDesign, params: HyperbolicParams) -> float:
    v0 = design.immediate_reward
    v1 = design.delayed_reward / (1.0 + params.k * design.days)
    return (v1 - v0) / params.alpha


def implied_choice_probability(design: HyperbolicDesign, params: HyperbolicParams) -> float:
    z = latent_mean(design, params)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def implied_prefers_delayed(params: HyperbolicParams, design: HyperbolicDesign) -> bool:
    """True when the probit surrogate prefers the delayed reward over immediate."""
    return implied_choice_probability(design, params) >= 0.5


def eval_holdout_designs(config: Config) -> list[HyperbolicDesign]:
    """Holdout designs for BoxingGym-style implied binary-choice accuracy."""
    if config.htd_eval_holdout_designs:
        from environments.hyperbolic_discounting.parsing import parse_hyperbolic_design

        return [parse_hyperbolic_design(item) for item in config.htd_eval_holdout_designs]
    ir_lo, ir_hi = config.htd_ir_bounds
    dr_lo, dr_hi = config.htd_dr_bounds
    days_lo, days_hi = config.htd_days_bounds
    return [
        normalize_design(ir_lo, dr_hi, days_lo),
        normalize_design(ir_hi, dr_lo, days_hi),
        normalize_design((ir_lo + ir_hi) / 2.0, (dr_lo + dr_hi) / 2.0, (days_lo + days_hi) // 2),
        normalize_design(ir_hi, dr_hi, days_lo),
    ]


def implied_choice_accuracy(
    estimate: HyperbolicParams,
    truth: HyperbolicParams,
    designs: list[HyperbolicDesign],
) -> float:
    if not designs:
        return 0.0
    matches = sum(
        1
        for design in designs
        if implied_prefers_delayed(estimate, design) == implied_prefers_delayed(truth, design)
    )
    return float(matches) / float(len(designs))


def _log_normal_pdf(value: float, mean: float, sd: float) -> float:
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def log_prior(params: HyperbolicParams, config: Config) -> float:
    log_k = math.log(params.k)
    log_k_prior = -0.5 * ((log_k - config.htd_k_mean) / config.htd_k_std) ** 2
    log_k_prior -= math.log(config.htd_k_std) + 0.5 * math.log(2.0 * math.pi)
    alpha_prior = (
        math.log(2.0)
        - math.log(math.pi)
        - math.log(config.htd_alpha_scale ** 2)
        - (params.alpha ** 2) / (2.0 * config.htd_alpha_scale ** 2)
    )
    return float(log_k_prior + alpha_prior)


def _dedupe_hypotheses(hypotheses: list[HyperbolicParams]) -> list[HyperbolicParams]:
    deduped: list[HyperbolicParams] = []
    seen: set[tuple[float, float]] = set()
    for hypothesis in hypotheses:
        key = (round(hypothesis.k, 8), round(hypothesis.alpha, 8))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hypothesis)
    return deduped


def sort_hyperbolic_belief_state(belief_state: HyperbolicBeliefState) -> HyperbolicBeliefState:
    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return HyperbolicBeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability) for _hypothesis, probability in ordered],
    )


def prune_hyperbolic_beliefs(
    belief_state: HyperbolicBeliefState,
    max_beliefs: int,
) -> HyperbolicBeliefState:
    if len(belief_state.hypotheses) <= max_beliefs:
        return belief_state
    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )[:max_beliefs]
    total = sum(probability for _hypothesis, probability in ordered)
    if total <= 0.0:
        uniform = 1.0 / len(ordered)
        return HyperbolicBeliefState(
            [hypothesis for hypothesis, _probability in ordered],
            [uniform] * len(ordered),
        )
    return HyperbolicBeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability / total) for _hypothesis, probability in ordered],
    )


def build_hyperbolic_belief_state_unpruned(
    hypotheses: list[HyperbolicParams],
    observations: list[HyperbolicObservation],
    config: Config,
) -> HyperbolicBeliefState:
    hypotheses = _dedupe_hypotheses(hypotheses)
    if not hypotheses:
        return HyperbolicBeliefState([], [])

    log_scores = np.asarray([log_prior(hypothesis, config) for hypothesis in hypotheses], dtype=float)
    if observations:
        sd = config.htd_noise_sd
        for observation in observations:
            means = np.asarray(
                [latent_mean(observation.design, hypothesis) for hypothesis in hypotheses],
                dtype=float,
            )
            z = (observation.value - means) / sd
            log_scores += -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)

    normalizer = _logsumexp(log_scores)
    probabilities = np.exp(log_scores - normalizer).tolist()
    return sort_hyperbolic_belief_state(HyperbolicBeliefState(hypotheses, probabilities))


def build_hyperbolic_belief_state(
    hypotheses: list[HyperbolicParams],
    observations: list[HyperbolicObservation],
    config: Config,
) -> HyperbolicBeliefState:
    state = build_hyperbolic_belief_state_unpruned(hypotheses, observations, config)
    return prune_hyperbolic_beliefs(state, max_beliefs=config.htd_max_total_beliefs)


def _hyperbolic_posterior_labels(count: int) -> list[str]:
    return [f"H{index + 1}" for index in range(count)]


def _hyperbolic_posterior_context_probabilities(
    hypotheses: list[HyperbolicParams],
    context_state: HyperbolicBeliefState | None,
) -> list[float]:
    if context_state is None or not context_state.hypotheses:
        uniform = 1.0 / len(hypotheses)
        return [uniform] * len(hypotheses)
    lookup = {
        (round(hypothesis.k, 8), round(hypothesis.alpha, 8)): probability
        for hypothesis, probability in zip(context_state.hypotheses, context_state.probabilities)
    }
    raw = [
        lookup.get((round(hypothesis.k, 8), round(hypothesis.alpha, 8)), 0.0)
        for hypothesis in hypotheses
    ]
    total = sum(raw)
    if total <= 0.0:
        uniform = 1.0 / len(hypotheses)
        return [uniform] * len(hypotheses)
    return [float(value / total) for value in raw]


def build_hyperbolic_posteriors_many(
    questioner: "Model | None",
    hypotheses_many: list[list[HyperbolicParams]],
    observations_many: list[list[HyperbolicObservation]],
    config: Config,
    *,
    context_states: list[HyperbolicBeliefState | None] | None = None,
    label: str = "hyperbolic posterior scoring",
    prune: bool = True,
) -> list[HyperbolicBeliefState]:
    if len(hypotheses_many) != len(observations_many):
        raise ValueError("hypotheses_many and observations_many must have the same length")
    if context_states is None:
        context_states = [None] * len(hypotheses_many)
    if len(context_states) != len(hypotheses_many):
        raise ValueError("context_states and hypotheses_many must have the same length")

    if config.htd_posterior_mode == "analytical_likelihood":
        states = [
            build_hyperbolic_belief_state_unpruned(hypotheses, observations, config)
            for hypotheses, observations in zip(hypotheses_many, observations_many)
        ]
        return [
            prune_hyperbolic_beliefs(state, max_beliefs=config.htd_max_total_beliefs) if prune else state
            for state in states
        ]
    if config.htd_posterior_mode != "llm_distribution":
        raise ValueError("htd_posterior_mode must be one of: analytical_likelihood, llm_distribution")
    if questioner is None:
        raise ValueError("htd_posterior_mode='llm_distribution' requires a questioner model")

    from environments.hyperbolic_discounting.prompts import hyperbolic_posterior_distribution_messages

    deduped = [_dedupe_hypotheses(list(hypotheses)) for hypotheses in hypotheses_many]
    batch_messages: list[list[dict[str, str]]] = []
    branch_labels: list[list[str]] = []
    branch_prompt_counts: list[int] = []

    for hypotheses, observations, context_state in zip(deduped, observations_many, context_states):
        labels = _hyperbolic_posterior_labels(len(hypotheses))
        branch_labels.append(labels)
        if not hypotheses:
            branch_prompt_counts.append(0)
            continue
        context_probabilities = _hyperbolic_posterior_context_probabilities(hypotheses, context_state)
        histories = [list(observations) for _ in range(config.belief_distribution_num_calls)]
        prompts = [
            hyperbolic_posterior_distribution_messages(history, hypotheses, context_probabilities, config)
            for history in histories
        ]
        branch_prompt_counts.append(len(prompts))
        batch_messages.extend(prompts)

    completions: list[str] = []
    if batch_messages:
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=batch_messages,
                temperature=config.belief_probability_temperature,
                block_size=config.batched_block_size,
                max_new_tokens=max(512, min(2048, 32 * max((len(labels) for labels in branch_labels), default=0) + 128)),
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.belief_probability_temperature)[0]
                for messages in batch_messages
            ]

    scored_states: list[HyperbolicBeliefState] = []
    completion_offset = 0
    for branch_idx, hypotheses in enumerate(deduped):
        labels = branch_labels[branch_idx]
        prompt_count = branch_prompt_counts[branch_idx]
        branch_completions = completions[completion_offset:completion_offset + prompt_count]
        completion_offset += prompt_count
        if not hypotheses:
            scored_states.append(HyperbolicBeliefState([], []))
            continue
        context_probabilities = _hyperbolic_posterior_context_probabilities(
            hypotheses,
            context_states[branch_idx],
        )
        context_distribution = dict(zip(labels, context_probabilities))
        distribution, _valid = _average_labeled_distributions_from_completions(
            branch_completions,
            labels,
            fallback_to_uniform=config.probability_parse_fallback_to_uniform,
            fallback_distribution=context_distribution,
        )
        scored_state = sort_hyperbolic_belief_state(
            HyperbolicBeliefState(hypotheses, [distribution[label] for label in labels])
        )
        scored_states.append(
            prune_hyperbolic_beliefs(scored_state, max_beliefs=config.htd_max_total_beliefs) if prune else scored_state
        )
    _log_hyperbolic(f"{label}: scored LLM posterior across {len(scored_states)} branch(es)", config)
    return scored_states


def build_hyperbolic_posterior(
    questioner: "Model | None",
    hypotheses: list[HyperbolicParams],
    observations: list[HyperbolicObservation],
    config: Config,
    *,
    context_state: HyperbolicBeliefState | None = None,
    label: str = "hyperbolic posterior scoring",
    prune: bool = True,
) -> HyperbolicBeliefState:
    return build_hyperbolic_posteriors_many(
        questioner,
        [hypotheses],
        [observations],
        config,
        context_states=[context_state],
        label=label,
        prune=prune,
    )[0]


def _seed_hyperbolic_hypotheses(config: Config, rng: np.random.Generator) -> list[HyperbolicParams]:
    count = max(3, min(config.htd_max_llm_prompt_beliefs, 12))
    hypotheses: list[HyperbolicParams] = []
    for _ in range(count):
        log_k = float(rng.normal(config.htd_k_mean, config.htd_k_std))
        k = math.exp(log_k)
        alpha = abs(float(rng.normal(0.0, config.htd_alpha_scale)))
        if alpha <= 0.0:
            alpha = config.htd_alpha_scale
        hypotheses.append(HyperbolicParams(k=k, alpha=alpha))
    return _dedupe_hypotheses(hypotheses)


def prompt_hyperbolic_belief_state(
    belief_state: HyperbolicBeliefState,
    config: Config,
) -> HyperbolicBeliefState:
    return prune_hyperbolic_beliefs(belief_state, max_beliefs=config.htd_max_llm_prompt_beliefs)


def _summarize_belief_state(belief_state: HyperbolicBeliefState) -> str:
    if not belief_state.hypotheses:
        return "empty"
    top = belief_state.hypotheses[0]
    top_probability = belief_state.probabilities[0] if belief_state.probabilities else 0.0
    return f"top=(k={top.k:g}, alpha={top.alpha:g}, p={top_probability:.4f}), n={len(belief_state.hypotheses)}"


def _format_params(params: HyperbolicParams) -> str:
    return f"(k={params.k:g}, alpha={params.alpha:g})"


def _generate_hyperbolic_hypotheses_many(
    questioner: "Model",
    observations_many: list[list[HyperbolicObservation]],
    belief_states: list[HyperbolicBeliefState | None],
    config: Config,
    *,
    label: str = "batched belief generation",
) -> list[list[HyperbolicParams]]:
    from environments.hyperbolic_discounting.parsing import parse_hyperbolic_hypotheses
    from environments.hyperbolic_discounting.prompts import belief_generation_messages

    if len(observations_many) != len(belief_states):
        raise ValueError("observations_many and belief_states must have the same length")
    if not observations_many:
        return []

    batch_messages = [
        belief_generation_messages(observations, belief_state, config)
        for observations, belief_state in zip(observations_many, belief_states)
    ]
    results: list[list[HyperbolicParams] | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[index] for index in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} hypothesis completions, received {len(completions)}")
        still_pending: list[int] = []
        for index, completion in zip(pending, completions):
            try:
                results[index] = parse_hyperbolic_hypotheses(completion)
            except ValueError as exc:
                _log_hyperbolic(f"{label}: item {index} parse failed ({exc})", config)
                still_pending.append(index)
        pending = still_pending

    seed_rng = np.random.default_rng(config.htd_seed)
    hypotheses_many: list[list[HyperbolicParams]] = []
    for index, raw in enumerate(results):
        if raw:
            hypotheses_many.append(raw[: config.htd_num_generated_hypotheses])
        else:
            hypotheses_many.append(_seed_hyperbolic_hypotheses(config, seed_rng))
    _log_hyperbolic(f"{label}: parsed hypotheses for {len(hypotheses_many)} trial(s)", config)
    return hypotheses_many


def generate_hyperbolic_candidates_many(
    questioner: "Model",
    belief_states: list[HyperbolicBeliefState],
    observations_many: list[list[HyperbolicObservation]],
    config: Config,
) -> list[list[HyperbolicDesign]]:
    from environments.hyperbolic_discounting.parsing import parse_candidate_designs
    from environments.hyperbolic_discounting.prompts import candidate_generation_messages

    if len(belief_states) != len(observations_many):
        raise ValueError("belief_states and observations_many must have the same length")
    if not belief_states:
        return []

    bounds = (
        tuple(config.htd_ir_bounds),
        tuple(config.htd_dr_bounds),
        tuple(config.htd_days_bounds),
    )
    batch_messages = [
        candidate_generation_messages(belief_state, observations, config)
        for belief_state, observations in zip(belief_states, observations_many)
    ]
    results: list[list[HyperbolicDesign] | None] = [None] * len(belief_states)
    pending = list(range(len(belief_states)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[index] for index in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} candidate completions, received {len(completions)}")
        still_pending: list[int] = []
        for index, completion in zip(pending, completions):
            try:
                results[index] = parse_candidate_designs(
                    completion,
                    ir_bounds=bounds[0],
                    dr_bounds=bounds[1],
                    days_bounds=bounds[2],
                )
            except ValueError as exc:
                _log_hyperbolic(f"candidate generation item {index} failed ({exc})", config)
                still_pending.append(index)
        pending = still_pending

    return [
        (raw if raw is not None else [])[: config.htd_target_num_candidates]
        for raw in results
    ]


def choose_design_naive_many(
    questioner: "Model",
    observations_many: list[list[HyperbolicObservation]],
    config: Config,
) -> list[HyperbolicDesign | None]:
    from environments.hyperbolic_discounting.parsing import parse_candidate_designs
    from environments.hyperbolic_discounting.prompts import naive_design_messages

    if not observations_many:
        return []
    bounds = (
        tuple(config.htd_ir_bounds),
        tuple(config.htd_dr_bounds),
        tuple(config.htd_days_bounds),
    )
    batch_messages = [naive_design_messages(observations, config) for observations in observations_many]
    results: list[HyperbolicDesign | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[index] for index in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} naive design completions, received {len(completions)}")
        still_pending: list[int] = []
        for index, completion in zip(pending, completions):
            try:
                designs = parse_candidate_designs(
                    completion,
                    ir_bounds=bounds[0],
                    dr_bounds=bounds[1],
                    days_bounds=bounds[2],
                )
                results[index] = designs[0] if designs else None
            except ValueError:
                still_pending.append(index)
        pending = still_pending
    return results


def estimate_params_naive_many(
    questioner: "Model",
    observations_many: list[list[HyperbolicObservation]],
    config: Config,
) -> list[HyperbolicParams]:
    from environments.hyperbolic_discounting.parsing import parse_hyperbolic_hypotheses
    from environments.hyperbolic_discounting.prompts import naive_estimate_messages

    if not observations_many:
        return []
    batch_messages = [naive_estimate_messages(observations, config) for observations in observations_many]
    if callable(getattr(questioner, "chat_complete_messages_batched", None)):
        completions = questioner.chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=config.generation_temperature_simple,
            block_size=config.batched_block_size,
        )
    else:
        completions = [
            questioner.chat_complete(messages, temperature=config.generation_temperature_simple)[0]
            for messages in batch_messages
        ]
    seed_rng = np.random.default_rng(config.htd_seed)
    estimates: list[HyperbolicParams] = []
    for completion in completions:
        hypotheses = parse_hyperbolic_hypotheses(completion)
        estimates.append(
            hypotheses[0] if hypotheses else _seed_hyperbolic_hypotheses(config, seed_rng)[0]
        )
    return estimates


def choose_strategy_designs_many(
    questioner: "Model",
    belief_states: list[HyperbolicBeliefState],
    observations_many: list[list[HyperbolicObservation]],
    config: Config,
) -> list[tuple[HyperbolicDesign | None, float]]:
    return [
        choose_strategy_design(questioner, belief_state, observations, config)
        for belief_state, observations in zip(belief_states, observations_many)
    ]


def generate_hyperbolic_hypotheses(
    questioner: "Model",
    observations: list[HyperbolicObservation],
    belief_state: HyperbolicBeliefState | None,
    config: Config,
    *,
    label: str = "belief generation",
    rng: np.random.Generator | None = None,
) -> list[HyperbolicParams]:
    from environments.hyperbolic_discounting.parsing import parse_hyperbolic_hypotheses
    from environments.hyperbolic_discounting.prompts import belief_generation_messages

    _log_hyperbolic(
        f"{label}: requesting hyperbolic hypotheses (observations={len(observations)})",
        config,
    )
    messages = belief_generation_messages(observations, belief_state, config)
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            hypotheses = parse_hyperbolic_hypotheses(completion)
            if hypotheses:
                return hypotheses[: config.htd_num_generated_hypotheses]
        except ValueError as exc:
            _log_hyperbolic(f"{label}: parse failed attempt {attempt + 1}/3 ({exc})", config)
    seed_rng = rng or np.random.default_rng(config.htd_seed)
    return _seed_hyperbolic_hypotheses(config, seed_rng)


def generate_hyperbolic_candidates(
    questioner: "Model",
    belief_state: HyperbolicBeliefState,
    observations: list[HyperbolicObservation],
    config: Config,
) -> list[HyperbolicDesign]:
    from environments.hyperbolic_discounting.parsing import parse_candidate_designs
    from environments.hyperbolic_discounting.prompts import candidate_generation_messages

    messages = candidate_generation_messages(belief_state, observations, config)
    bounds = (
        tuple(config.htd_ir_bounds),
        tuple(config.htd_dr_bounds),
        tuple(config.htd_days_bounds),
    )
    candidates: list[HyperbolicDesign] = []
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            candidates = parse_candidate_designs(
                completion,
                ir_bounds=bounds[0],
                dr_bounds=bounds[1],
                days_bounds=bounds[2],
            )
            break
        except ValueError as exc:
            _log_hyperbolic(f"candidate generation attempt {attempt + 1}/3 failed ({exc})", config)
    return candidates[: config.htd_target_num_candidates]


def choose_design_naive(
    questioner: "Model",
    observations: list[HyperbolicObservation],
    config: Config,
) -> HyperbolicDesign | None:
    from environments.hyperbolic_discounting.parsing import parse_candidate_designs
    from environments.hyperbolic_discounting.prompts import naive_design_messages

    messages = naive_design_messages(observations, config)
    bounds = (
        tuple(config.htd_ir_bounds),
        tuple(config.htd_dr_bounds),
        tuple(config.htd_days_bounds),
    )
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            designs = parse_candidate_designs(
                completion,
                ir_bounds=bounds[0],
                dr_bounds=bounds[1],
                days_bounds=bounds[2],
            )
            if designs:
                return designs[0]
        except ValueError:
            continue
    return None


def estimate_params_naive(
    questioner: "Model",
    observations: list[HyperbolicObservation],
    config: Config,
) -> HyperbolicParams:
    from environments.hyperbolic_discounting.parsing import parse_hyperbolic_hypotheses
    from environments.hyperbolic_discounting.prompts import naive_estimate_messages

    completion = questioner.chat_complete(
        naive_estimate_messages(observations, config),
        temperature=config.generation_temperature_simple,
    )[0]
    hypotheses = parse_hyperbolic_hypotheses(completion)
    if not hypotheses:
        rng = np.random.default_rng(config.htd_seed)
        return _seed_hyperbolic_hypotheses(config, rng)[0]
    return hypotheses[0]


def parameter_rmse(estimate: HyperbolicParams, truth: HyperbolicParams) -> float:
    return float(math.sqrt((estimate.k - truth.k) ** 2 + (estimate.alpha - truth.alpha) ** 2))


def k_rmse(estimate: HyperbolicParams, truth: HyperbolicParams) -> float:
    return abs(estimate.k - truth.k)


def _hyperbolic_effective_sample_size(belief_state: HyperbolicBeliefState) -> float:
    if not belief_state.probabilities:
        return 0.0
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    denominator = float(np.sum(probabilities ** 2))
    if denominator <= 0.0:
        return 0.0
    return 1.0 / denominator


def sample_hyperbolic_eig_belief_state(
    belief_state: HyperbolicBeliefState,
    config: Config,
    rng: np.random.Generator,
) -> tuple[HyperbolicBeliefState, bool]:
    if len(belief_state.hypotheses) <= config.num_mc_samples:
        return belief_state, False
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_indices = rng.choice(
        len(belief_state.hypotheses),
        size=config.num_mc_samples,
        replace=True,
        p=probabilities,
    )
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    sample_collapsed = len(unique_indices) < min(2, len(belief_state.hypotheses))
    sampled_hypotheses = [belief_state.hypotheses[int(index)] for index in unique_indices]
    sampled_probabilities = [float(count / np.sum(counts)) for count in counts]
    return sort_hyperbolic_belief_state(
        HyperbolicBeliefState(sampled_hypotheses, sampled_probabilities)
    ), sample_collapsed


def _posterior_after_observation(
    belief_state: HyperbolicBeliefState,
    design: HyperbolicDesign,
    value: float,
    noise_sd: float,
    config: Config,
) -> HyperbolicBeliefState:
    if not belief_state.hypotheses:
        return belief_state
    log_scores = []
    for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        mean = latent_mean(design, hypothesis)
        log_scores.append(
            math.log(max(probability, 1e-300))
            + _log_normal_pdf(value, mean, noise_sd)
            + log_prior(hypothesis, config)
        )
    normalizer = _logsumexp(np.asarray(log_scores, dtype=float))
    return sort_hyperbolic_belief_state(
        HyperbolicBeliefState(
            list(belief_state.hypotheses),
            [math.exp(log_score - normalizer) for log_score in log_scores],
        )
    )


def _quadrature_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(float), (weights.astype(float) / math.sqrt(math.pi))


def _normal_logpdf_array(values: np.ndarray, means: np.ndarray, noise_sd: float) -> np.ndarray:
    z = (values - means) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp_array(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True)), axis=axis)


def _expected_information_gain_from_means(
    probabilities: np.ndarray,
    means: np.ndarray,
    noise_sd: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    if len(means) <= 1:
        return 0.0
    probabilities = np.asarray(probabilities, dtype=float)
    means = np.asarray(means, dtype=float)
    y_values = means[:, None] + math.sqrt(2.0) * noise_sd * nodes[None, :]
    component_log_likelihoods = _normal_logpdf_array(y_values, means[:, None], noise_sd)
    all_log_likelihoods = _normal_logpdf_array(y_values[:, :, None], means[None, None, :], noise_sd)
    mixture_log_likelihoods = _logsumexp_array(
        all_log_likelihoods + np.log(np.maximum(probabilities, 1e-300))[None, None, :],
        axis=2,
    )
    value = np.sum(probabilities[:, None] * weights[None, :] * (component_log_likelihoods - mixture_log_likelihoods))
    return max(0.0, float(value))


def expected_information_gain(
    belief_state: HyperbolicBeliefState,
    design: HyperbolicDesign,
    noise_sd: float,
    quadrature_order: int,
) -> float:
    if len(belief_state.hypotheses) <= 1:
        return 0.0
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    means = np.asarray(
        [latent_mean(design, hypothesis) for hypothesis in belief_state.hypotheses],
        dtype=float,
    )
    nodes, weights = _quadrature_nodes(quadrature_order)
    return _expected_information_gain_from_means(probabilities, means, noise_sd, nodes, weights)


def score_candidate_designs(
    belief_state: HyperbolicBeliefState,
    candidates: list[HyperbolicDesign],
    config: Config,
    questioner: "Model | None" = None,
    observations: list[HyperbolicObservation] | None = None,
) -> list[float]:
    from environments.hyperbolic_discounting.env import HyperbolicBEDEnvironment, _belief_state_from_hyperbolic
    from methods.continuous_eig import score_continuous_forward_search

    if not candidates:
        return []
    if (
        config.htd_search_depth == 2
        and config.htd_posterior_mode == "llm_distribution"
        and (questioner is None or observations is None)
    ):
        raise ValueError("depth-2 LLM EIG scoring requires questioner and observations")
    environment = HyperbolicBEDEnvironment(config=config)
    history = [
        (observation.design, observation)
        for observation in (observations or [])
    ]
    return score_continuous_forward_search(
        _belief_state_from_hyperbolic(belief_state),
        candidates,
        environment,
        questioner,
        history,
        config,
        noise_sd=config.htd_noise_sd,
        quadrature_order=config.htd_eig_quadrature_order,
        search_depth=config.htd_search_depth,
    )


def _merge_hypotheses(
    belief_state: HyperbolicBeliefState | None,
    new_hypotheses: list[HyperbolicParams],
) -> list[HyperbolicParams]:
    existing = [] if belief_state is None else list(belief_state.hypotheses)
    return _dedupe_hypotheses(existing + list(new_hypotheses))


def _top_parameter_rmse(belief_state: HyperbolicBeliefState, truth: HyperbolicParams) -> float:
    if not belief_state.hypotheses:
        return float("inf")
    top = belief_state.hypotheses[0]
    return parameter_rmse(top, truth)


def choose_strategy_design(
    questioner: "Model",
    belief_state: HyperbolicBeliefState,
    observations: list[HyperbolicObservation],
    config: Config,
) -> tuple[HyperbolicDesign | None, float]:
    candidates = generate_hyperbolic_candidates(questioner, belief_state, observations, config)
    if not candidates:
        return None, 0.0
    scores = score_candidate_designs(belief_state, candidates, config, questioner, observations)
    best_idx = int(np.argmax(scores))
    return candidates[best_idx], float(scores[best_idx])


def _hyperbolic_trial_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.htd_seed is None:
        return np.random.default_rng(fallback_rng.integers(0, np.iinfo(np.uint32).max))
    seed_sequence = np.random.SeedSequence([config.htd_seed, trial_idx, 0])
    return np.random.default_rng(seed_sequence)


def _hyperbolic_planning_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.htd_seed is None:
        return np.random.default_rng(fallback_rng.integers(0, np.iinfo(np.uint32).max))
    seed_sequence = np.random.SeedSequence([config.htd_seed, trial_idx, 1])
    return np.random.default_rng(seed_sequence)


def _make_hyperbolic_trial_state(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> _HyperbolicTrialState:
    env_rng = _hyperbolic_trial_rng(config, trial_idx, fallback_rng)
    planning_rng = _hyperbolic_planning_rng(config, trial_idx, fallback_rng)
    env = HyperbolicDiscountingEnv(
        noise_sd=config.htd_noise_sd,
        rng=env_rng,
        k_mean=config.htd_k_mean,
        k_std=config.htd_k_std,
        alpha_scale=config.htd_alpha_scale,
    )
    return _HyperbolicTrialState(
        trial_idx=trial_idx,
        env=env,
        observations=[],
        rng=planning_rng,
    )


def _write_to_log_if_configured(message: str, config: Config) -> None:
    if config.log_path is not None:
        write_to_log(message, config)


def run_hyperbolic_finding(
    questioner: "Model",
    config: Config,
    rng: np.random.Generator | None = None,
    output_dir: Path | None = None,
    method_name: str = "EIG",
) -> HyperbolicFindingMetrics:
    """Run hyperbolic temporal discounting through :mod:`core.experiment`."""
    del rng
    if method_name not in {"EIG", "StrategyEIG", "StrategyEIG+root", "Naive", "naive", "naive+belief"}:
        raise ValueError(
            "Hyperbolic discounting supports method_name='EIG', 'StrategyEIG', "
            "'StrategyEIG+root', 'Naive', or 'naive+belief'"
        )
    if config.htd_trial_batch_size > 1:
        from environments.hyperbolic_discounting.batched_trials import run_hyperbolic_trials_batched

        return run_hyperbolic_trials_batched(
            questioner,
            config,
            output_dir=output_dir,
            method_name=method_name,
        )
    from core.experiment import run_from_config

    _run_result, summary = run_from_config(
        config,
        questioner,
        answerer=None,
        method_name=method_name,
        output_dir=output_dir,
    )
    return HyperbolicFindingMetrics(
        parameter_rmse=list(summary.metrics.get("parameter_rmse", [])),
        k_rmse=list(summary.metrics.get("k_rmse", [])),
        top_probability=list(summary.metrics.get("top_probability", [])),
        selected_eig=list(summary.metrics.get("selected_eig", [])),
        implied_choice_accuracy=list(summary.metrics.get("implied_choice_accuracy", [])),
    )


def _log_hyperbolic(message: str, config: Config) -> None:
    print_and_log(f"[hyperbolic_discounting] {message}", config)
