from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core import BeliefState
from helpers import Config
from .beliefs import build_location_posterior
from .formatting import _format_source_array, _log_location, _summarize_candidates
from .parsing import parse_best_source_estimate_from_completion, parse_candidate_locations, parse_single_location_from_completion, parse_source_hypotheses
from .prompts import _belief_generation_messages, _candidate_generation_messages, _naive_location_messages, _naive_source_estimate_messages, _naive_source_estimate_repair_messages
from .types import Location, LocationObservation, SourceConfig

if TYPE_CHECKING:
    from model import Model


def _location_key(location: Location) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in location)


def _is_repeated_location(location: Location, observations: list[LocationObservation]) -> bool:
    key = _location_key(location)
    return any(_location_key(observation.query) == key for observation in observations)


def _completion_excerpt(completion: str, max_chars: int = 800) -> str:
    cleaned = completion.replace("\n", "\\n")
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return f"{cleaned[:half]} ... {cleaned[-half:]}"


def generate_location_hypotheses(
    questioner: "Model",
    observations: list[LocationObservation],
    belief_state: BeliefState | None,
    config: Config,
    *,
    label: str = "belief generation",
) -> list[SourceConfig]:
    previous_count = 0 if belief_state is None else len(belief_state.hypotheses)
    _log_location(
        f"{label}: requesting source hypotheses "
        f"(observations={len(observations)}, previous_beliefs={previous_count}, "
        f"max_generate={config.location_num_generated_hypotheses}, "
        f"max_context={config.location_max_llm_prompt_beliefs})",
        config,
    )
    messages = _belief_generation_messages(observations, belief_state, config)
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            hypotheses = parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            _log_location(f"{label}: parsed {len(hypotheses)} valid unique source configuration(s)", config)
            return hypotheses
        except ValueError as exc:
            _log_location(
                f"{label}: attempt {attempt + 1}/3 could not parse source hypotheses ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
    return []


def _generate_location_hypotheses_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    belief_states: list[BeliefState | None],
    config: Config,
    *,
    label: str = "batched belief generation",
) -> list[list[SourceConfig]]:
    if len(observations_many) != len(belief_states):
        raise ValueError("observations_many and belief_states must have the same length")
    if not observations_many:
        return []

    _log_location(
        f"{label}: requesting {len(observations_many)} hypothetical source-support refresh(es) "
        f"(block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [
        _belief_generation_messages(observations, belief_state, config)
        for observations, belief_state in zip(observations_many, belief_states)
    ]
    results: list[list[SourceConfig] | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
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
        for idx, completion in zip(pending, completions):
            try:
                results[idx] = parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            except ValueError as exc:
                _log_location(
                    f"{label}: attempt {attempt + 1}/3 item {idx} could not parse hypotheses ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(idx)
        pending = still_pending

    hypotheses_many = [result if result is not None else [] for result in results]
    counts = [len(hypotheses) for hypotheses in hypotheses_many]
    nonempty_count = sum(1 for count in counts if count > 0)
    total_count = sum(counts)
    _log_location(
        f"{label}: parsed {total_count} generated hypothesis/hypotheses across "
        f"{nonempty_count}/{len(hypotheses_many)} nonempty refresh(es)",
        config,
    )
    return hypotheses_many


def generate_location_candidates(
    questioner: "Model",
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[Location]:
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"candidate generation: requesting {config.location_target_num_candidates} location(s) "
        f"(observations={len(observations)}, beliefs={len(belief_state.hypotheses)}, "
        f"bounds=[{bounds[0]}, {bounds[1]}])",
        config,
    )
    messages = _candidate_generation_messages(belief_state, observations, config)
    candidates: list[Location] = []
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            candidates = parse_candidate_locations(completion, config.location_dim, bounds)
            break
        except ValueError as exc:
            _log_location(
                f"candidate generation: attempt {attempt + 1}/3 could not parse candidates ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
    selected = candidates[:config.location_target_num_candidates]
    _log_location(
        f"candidate generation: parsed={len(candidates)}, returned={len(selected)}, "
        f"locations={_summarize_candidates(selected)}",
        config,
    )
    return selected


def generate_location_candidates_many(
    questioner: "Model",
    belief_states: list[BeliefState],
    observations_many: list[list[LocationObservation]],
    config: Config,
) -> list[list[Location]]:
    if len(belief_states) != len(observations_many):
        raise ValueError("belief_states and observations_many must have the same length")
    if not belief_states:
        return []
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"candidate generation: requesting candidates for {len(belief_states)} trial(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [
        _candidate_generation_messages(belief_state, observations, config)
        for belief_state, observations in zip(belief_states, observations_many)
    ]
    results: list[list[Location] | None] = [None] * len(belief_states)
    pending = list(range(len(belief_states)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} candidate completions, received {len(completions)}")
        still_pending: list[int] = []
        for idx, completion in zip(pending, completions):
            try:
                results[idx] = parse_candidate_locations(completion, config.location_dim, bounds)
            except ValueError as exc:
                _log_location(
                    f"candidate generation: attempt {attempt + 1}/3 item {idx} could not parse candidates ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(idx)
        pending = still_pending

    candidates_many: list[list[Location]] = []
    for idx, raw in enumerate(results):
        candidates = raw if raw is not None else []
        selected = candidates[:config.location_target_num_candidates]
        _log_location(
            f"candidate generation: item {idx} parsed={len(candidates)}, returned={len(selected)}, "
            f"locations={_summarize_candidates(selected)}",
            config,
        )
        candidates_many.append(selected)
    return candidates_many


def choose_location_naive(
    questioner: "Model",
    observations: list[LocationObservation],
    config: Config,
    belief_state: BeliefState | None = None,
) -> Location | None:
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"naive query generation: requesting one location "
        f"(observations={len(observations)}, bounds=[{bounds[0]}, {bounds[1]}])",
        config,
    )
    messages = _naive_location_messages(observations, config, belief_state=belief_state)
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            location = parse_single_location_from_completion(completion, config.location_dim, bounds)
            _log_location(f"Naive selection: chose direct LLM query {list(location)}", config)
            return location
        except ValueError as exc:
            _log_location(
                f"naive query generation: attempt {attempt + 1}/3 failed ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
            _log_location(f"naive query generation raw completion excerpt: {_completion_excerpt(completion)}", config)
    return None


def estimate_sources_naive(
    questioner: "Model",
    observations: list[LocationObservation],
    config: Config,
) -> SourceConfig:
    _log_location(
        f"naive source estimate: requesting one final source configuration "
        f"(observations={len(observations)})",
        config,
    )
    completion = questioner.chat_complete(
        _naive_source_estimate_messages(observations, config),
        temperature=config.generation_temperature_simple,
    )[0]
    try:
        estimate = parse_best_source_estimate_from_completion(
            completion,
            config.location_num_sources,
            config.location_dim,
        )
    except ValueError as exc:
        _log_location(f"naive source estimate: could not parse estimate ({exc}); retrying JSON repair", config)
        _log_location(f"naive source estimate raw completion excerpt: {_completion_excerpt(completion)}", config)
        repair_completion = questioner.chat_complete(
            _naive_source_estimate_repair_messages(completion, observations, config),
            temperature=0.0,
        )[0]
        try:
            estimate = parse_best_source_estimate_from_completion(
                repair_completion,
                config.location_num_sources,
                config.location_dim,
            )
        except ValueError as repair_exc:
            _log_location(
                f"naive source estimate: repair failed ({repair_exc}); using center fallback",
                config,
            )
            estimate = tuple(
                tuple(0.0 for _coord_idx in range(config.location_dim))
                for _source_idx in range(config.location_num_sources)
            )
    _log_location(f"Naive estimate: chose sources {_format_source_array(np.asarray(estimate, dtype=float))}", config)
    return estimate


def choose_locations_naive_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    config: Config,
    belief_states: list[BeliefState | None] | None = None,
) -> list[Location | None]:
    if not observations_many:
        return []
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"naive query generation: requesting {len(observations_many)} location(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    if belief_states is None:
        belief_states = [None] * len(observations_many)
    if len(belief_states) != len(observations_many):
        raise ValueError("belief_states must match observations_many length")
    batch_messages = [
        _naive_location_messages(observations, config, belief_state=belief_state)
        for observations, belief_state in zip(observations_many, belief_states)
    ]
    results: list[Location | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} naive query completions, received {len(completions)}")
        still_pending: list[int] = []
        for idx, completion in zip(pending, completions):
            try:
                location = parse_single_location_from_completion(completion, config.location_dim, bounds)
                _log_location(f"Naive selection: chose direct LLM query {list(location)}", config)
                results[idx] = location
            except ValueError as exc:
                _log_location(
                    f"naive query generation: attempt {attempt + 1}/3 item {idx} failed ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                _log_location(f"naive query generation raw completion excerpt: {_completion_excerpt(completion)}", config)
                still_pending.append(idx)
        pending = still_pending

    return results


def estimate_sources_naive_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    config: Config,
) -> list[SourceConfig]:
    if not observations_many:
        return []
    _log_location(
        f"naive source estimate: requesting {len(observations_many)} source configuration(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [_naive_source_estimate_messages(observations, config) for observations in observations_many]
    if callable(getattr(questioner, "chat_complete_messages_batched", None)):
        completions = questioner.chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=config.generation_temperature_simple,
            block_size=config.batched_block_size,
            max_new_tokens=config.location_max_new_tokens,
        )
    else:
        completions = [
            questioner.chat_complete(messages, temperature=config.generation_temperature_simple)[0]
            for messages in batch_messages
        ]
    if len(completions) != len(batch_messages):
        raise ValueError(f"Expected {len(batch_messages)} naive estimate completions, received {len(completions)}")

    estimates: list[SourceConfig | None] = []
    failed_indices: list[int] = []
    for completion in completions:
        try:
            estimate = parse_best_source_estimate_from_completion(
                completion,
                config.location_num_sources,
                config.location_dim,
            )
        except ValueError as exc:
            _log_location(f"naive source estimate: could not parse estimate ({exc}); will retry JSON repair", config)
            _log_location(f"naive source estimate raw completion excerpt: {_completion_excerpt(completion)}", config)
            estimate = None
            failed_indices.append(len(estimates))
        estimates.append(estimate)

    if failed_indices:
        repair_messages = [
            _naive_source_estimate_repair_messages(
                completions[idx],
                observations_many[idx],
                config,
            )
            for idx in failed_indices
        ]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            repair_completions = questioner.chat_complete_messages_batched(
                batch_messages=repair_messages,
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            repair_completions = [
                questioner.chat_complete(messages, temperature=0.0)[0]
                for messages in repair_messages
            ]
        if len(repair_completions) != len(repair_messages):
            raise ValueError(f"Expected {len(repair_messages)} naive repair completions, received {len(repair_completions)}")
        for idx, repair_completion in zip(failed_indices, repair_completions):
            try:
                estimates[idx] = parse_best_source_estimate_from_completion(
                    repair_completion,
                    config.location_num_sources,
                    config.location_dim,
                )
            except ValueError as repair_exc:
                _log_location(
                    f"naive source estimate: repair failed ({repair_exc}); using center fallback",
                    config,
                )
                estimates[idx] = tuple(
                    tuple(0.0 for _coord_idx in range(config.location_dim))
                    for _source_idx in range(config.location_num_sources)
                )

    final_estimates: list[SourceConfig] = []
    for estimate in estimates:
        assert estimate is not None, "estimate should be set by the repair loop"
        _log_location(f"Naive estimate: chose sources {_format_source_array(np.asarray(estimate, dtype=float))}", config)
        final_estimates.append(estimate)
    return final_estimates
