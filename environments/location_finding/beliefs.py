from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from core import BeliefState
from helpers import Config, _average_labeled_distributions_from_completions
from .formatting import _location_posterior_labels, _log_location, _summarize_belief_state
from .parsing import parse_source_hypotheses
from .physics import _hypothesis_log_prior, _log_normal_pdf, _logsumexp, signal_intensity_for_hypothesis
from .prompts import _belief_generation_messages, _location_posterior_context_probabilities, _location_posterior_distribution_messages, _permuted_location_observation_histories
from .types import LocationObservation, SourceConfig, _dedupe_source_configs

if TYPE_CHECKING:
    from model import Model


def build_location_belief_state(
    hypotheses: list[SourceConfig],
    observations: list[LocationObservation],
    config: Config,
) -> BeliefState:
    state = build_location_belief_state_unpruned(hypotheses, observations, config)
    return prune_location_beliefs(state, max_beliefs=config.location_max_total_beliefs)


def build_location_belief_state_unpruned(
    hypotheses: list[SourceConfig],
    observations: list[LocationObservation],
    config: Config,
) -> BeliefState:
    hypotheses = _dedupe_source_configs(hypotheses)
    if not hypotheses:
        return BeliefState([], [])

    # theta: (H, S, D) — all hypotheses stacked into a single array
    theta = np.asarray(hypotheses, dtype=float)
    H, S, D = theta.shape

    # Log prior: -0.5 * ||theta||^2 - 0.5 * S*D * log(2π), shape (H,)
    log_scores = (
        -0.5 * np.sum(theta.reshape(H, -1) ** 2, axis=1)
        - 0.5 * S * D * math.log(2.0 * math.pi)
    )

    if observations:
        queries = np.asarray([obs.query for obs in observations], dtype=float)  # (O, D)
        values = np.asarray([obs.value for obs in observations], dtype=float)   # (O,)
        sd = config.location_noise_sd
        b, m, alpha = 0.1, 1e-4, 1.0

        # Pairwise squared distances between each hypothesis-source and each query.
        # theta[:, np.newaxis, :, :] → (H, 1, S, D)
        # queries[np.newaxis, :, np.newaxis, :] → (1, O, 1, D)
        # distances_sq → (H, O, S)
        distances_sq = np.sum(
            (theta[:, np.newaxis, :, :] - queries[np.newaxis, :, np.newaxis, :]) ** 2,
            axis=3,
        )

        # Signal means for every (hypothesis, observation) pair: (H, O)
        means = b + np.sum(alpha / (m + distances_sq), axis=2)

        # Log-likelihood under Gaussian noise: (H, O), then summed over observations → (H,)
        z = (values[np.newaxis, :] - means) / sd
        log_scores += np.sum(-0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi), axis=1)

    normalizer = _logsumexp(log_scores)
    probabilities = np.exp(log_scores - normalizer).tolist()
    state = BeliefState(hypotheses, probabilities)
    return sort_location_belief_state(state)


def sort_location_belief_state(belief_state: BeliefState) -> BeliefState:
    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return BeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability) for _hypothesis, probability in ordered],
    )


def prune_location_beliefs(
    belief_state: BeliefState,
    max_beliefs: int,
) -> BeliefState:
    if len(belief_state.hypotheses) <= max_beliefs:
        return belief_state

    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )[:max_beliefs]
    total = sum(probability for _hypothesis, probability in ordered)
    if total <= 0.0:
        probability = 1.0 / len(ordered)
        return BeliefState([hypothesis for hypothesis, _probability in ordered], [probability] * len(ordered))
    return BeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability / total) for _hypothesis, probability in ordered],
    )


def build_location_posteriors_many(
    questioner: "Model" | None,
    hypotheses_many: list[list[SourceConfig]],
    observations_many: list[list[LocationObservation]],
    config: Config,
    *,
    context_states: list[BeliefState | None] | None = None,
    label: str = "location posterior scoring",
    prune: bool = True,
    rng: np.random.Generator | None = None,
) -> list[BeliefState]:
    if len(hypotheses_many) != len(observations_many):
        raise ValueError("hypotheses_many and observations_many must have the same length")
    if context_states is None:
        context_states = [None] * len(hypotheses_many)
    if len(context_states) != len(hypotheses_many):
        raise ValueError("context_states and hypotheses_many must have the same length")

    if config.location_posterior_mode == "analytical_likelihood":
        states = [
            build_location_belief_state_unpruned(hypotheses, observations, config)
            for hypotheses, observations in zip(hypotheses_many, observations_many)
        ]
        return [
            prune_location_beliefs(state, max_beliefs=config.location_max_total_beliefs)
            if prune
            else state
            for state in states
        ]
    if config.location_posterior_mode != "llm_distribution":
        raise ValueError("location_posterior_mode must be one of: analytical_likelihood, llm_distribution")
    if questioner is None:
        raise ValueError("location_posterior_mode='llm_distribution' requires a questioner model")

    deduped_hypotheses_many = [_dedupe_source_configs(list(hypotheses)) for hypotheses in hypotheses_many]
    batch_messages: list[list[dict[str, str]]] = []
    branch_prompt_counts: list[int] = []
    branch_labels: list[list[str]] = []
    active_branch_indices: list[int] = []
    for branch_idx, (hypotheses, observations, context_state) in enumerate(
        zip(deduped_hypotheses_many, observations_many, context_states)
    ):
        labels = _location_posterior_labels(len(hypotheses))
        branch_labels.append(labels)
        if not hypotheses:
            branch_prompt_counts.append(0)
            continue
        active_branch_indices.append(branch_idx)
        context_probabilities = _location_posterior_context_probabilities(hypotheses, context_state)
        if config.belief_distribution_permute_history:
            histories = _permuted_location_observation_histories(
                observations,
                config.belief_distribution_num_calls,
                rng,
            )
        else:
            histories = [list(observations) for _call_idx in range(config.belief_distribution_num_calls)]
        prompts = [
            _location_posterior_distribution_messages(history, hypotheses, context_probabilities, config)
            for history in histories
        ]
        branch_prompt_counts.append(len(prompts))
        batch_messages.extend(prompts)

    completions: list[str] = []
    if batch_messages:
        posterior_max_new_tokens = max(512, min(2048, 32 * max((len(labels) for labels in branch_labels), default=0) + 128))
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=batch_messages,
                temperature=config.belief_probability_temperature,
                block_size=config.batched_block_size,
                max_new_tokens=posterior_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.belief_probability_temperature)[0]
                for messages in batch_messages
            ]
        if len(completions) != len(batch_messages):
            raise ValueError(
                f"Expected {len(batch_messages)} location posterior completions, received {len(completions)}"
            )

    scored_states: list[BeliefState] = []
    completion_offset = 0
    valid_total = 0
    prompt_total = 0
    for branch_idx, hypotheses in enumerate(deduped_hypotheses_many):
        labels = branch_labels[branch_idx]
        prompt_count = branch_prompt_counts[branch_idx]
        branch_completions = completions[completion_offset:completion_offset + prompt_count]
        completion_offset += prompt_count
        prompt_total += prompt_count
        if not hypotheses:
            scored_states.append(BeliefState([], []))
            continue
        context_probabilities = _location_posterior_context_probabilities(
            hypotheses,
            context_states[branch_idx],
        )
        context_distribution = {
            label: probability
            for label, probability in zip(labels, context_probabilities)
        }
        distribution, valid_count = _average_labeled_distributions_from_completions(
            branch_completions,
            labels,
            fallback_to_uniform=config.probability_parse_fallback_to_uniform,
            fallback_distribution=context_distribution,
        )
        valid_total += valid_count
        scored_state = sort_location_belief_state(
            BeliefState(
                hypotheses,
                [distribution[label] for label in labels],
            )
        )
        scored_states.append(
            prune_location_beliefs(scored_state, max_beliefs=config.location_max_total_beliefs)
            if prune
            else scored_state
        )

    detail = f"{valid_total}/{prompt_total} valid"
    if config.belief_distribution_permute_history:
        detail = f"permuted-history, {detail}"
    _log_location(
        f"{label}: scored LLM posterior distribution ({detail}) across "
        f"{len(active_branch_indices)}/{len(hypotheses_many)} nonempty support(s)",
        config,
    )
    return scored_states


def build_location_posterior(
    questioner: "Model" | None,
    hypotheses: list[SourceConfig],
    observations: list[LocationObservation],
    config: Config,
    *,
    context_state: BeliefState | None = None,
    label: str = "location posterior scoring",
    prune: bool = True,
) -> BeliefState:
    return build_location_posteriors_many(
        questioner,
        [hypotheses],
        [observations],
        config,
        context_states=[context_state],
        label=label,
        prune=prune,
    )[0]


def prompt_location_belief_state(
    belief_state: BeliefState,
    config: Config,
) -> BeliefState:
    return prune_location_beliefs(belief_state, max_beliefs=config.location_max_llm_prompt_beliefs)


def _location_effective_sample_size(belief_state: BeliefState) -> float:
    if not belief_state.probabilities:
        return 0.0
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    denominator = float(np.sum(probabilities ** 2))
    if denominator <= 0.0:
        return 0.0
    return 1.0 / denominator


def sample_location_eig_belief_state(
    belief_state: BeliefState,
    config: Config,
    rng: np.random.Generator,
) -> tuple[BeliefState, bool]:
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
    sampled_state = sort_location_belief_state(BeliefState(sampled_hypotheses, sampled_probabilities))
    return sampled_state, sample_collapsed


def _posterior_after_observation(
    belief_state: BeliefState,
    query: Location,
    value: float,
    noise_sd: float,
) -> BeliefState:
    if not belief_state.hypotheses:
        return belief_state
    log_scores = []
    for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        mean = signal_intensity_for_hypothesis(hypothesis, query)
        log_scores.append(math.log(max(probability, 1e-300)) + _log_normal_pdf(value, mean, noise_sd))
    normalizer = _logsumexp(log_scores)
    return sort_location_belief_state(
        BeliefState(
            list(belief_state.hypotheses),
            [math.exp(log_score - normalizer) for log_score in log_scores],
        )
    )


def _merge_hypotheses(previous: BeliefState, generated: list[SourceConfig]) -> list[SourceConfig]:
    return _dedupe_source_configs(list(previous.hypotheses) + list(generated))
