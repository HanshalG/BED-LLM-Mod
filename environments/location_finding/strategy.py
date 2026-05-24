from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from helpers import Config
from .beliefs import build_location_posteriors_many, prompt_location_belief_state, prune_location_beliefs
from .formatting import _format_location, _log_location
from .generation import _generate_location_hypotheses_many, _is_repeated_location
from .parsing import _clean_strategy_text, _strategy_key, parse_location_strategies, parse_location_strategy_roots, parse_strategy_location
from .physics import signal_intensity_for_hypothesis
from .prompts import _strategy_crossover_messages, _strategy_diverse_messages, _strategy_location_messages, _strategy_mutation_messages, _strategy_root_crossover_messages, _strategy_root_diverse_messages, _strategy_root_mutation_messages
from .types import Location, LocationBeliefState, LocationObservation, LocationStrategyCandidate, LocationStrategyEntry, LocationStrategyEvaluation, LocationStrategyLibrary, SourceConfig, _StrategyEvaluationRequest, _StrategyLocationRequest, _StrategyRollout, _dedupe_source_configs

if TYPE_CHECKING:
    from model import Model


def _extend_unique_strategies(target: list[str], candidates: list[str], seen: set[str], limit: int) -> None:
    for strategy in candidates:
        cleaned = _clean_strategy_text(strategy)
        if cleaned is None:
            continue
        key = _strategy_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        target.append(cleaned)
        if len(target) >= limit:
            return


def _extend_unique_strategy_candidates(
    target: list[LocationStrategyCandidate],
    candidates: list[LocationStrategyCandidate],
    seen: set[str],
    limit: int,
    observations: list[LocationObservation],
) -> None:
    for candidate in candidates:
        cleaned = _clean_strategy_text(candidate.strategy)
        if cleaned is None or candidate.root_query is None:
            continue
        if _is_repeated_location(candidate.root_query, observations):
            continue
        key = _strategy_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        target.append(LocationStrategyCandidate(strategy=cleaned, root_query=candidate.root_query))
        if len(target) >= limit:
            return


def _strategy_phase_single(
    questioner: "Model",
    messages: list[dict[str, str]],
    strategies: list[str],
    seen: set[str],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    """Run one strategy-generation phase with up to 3 attempts, extending strategies in place."""
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            _extend_unique_strategies(strategies, parse_location_strategies(completion), seen, target_count)
            return
        except ValueError as exc:
            _log_location(
                f"strategy proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )


def _strategy_phase_batched(
    questioner: "Model",
    batch_messages: list[list[dict[str, str]]],
    pending_indices: list[int],
    prepared: list[dict[str, object]],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    """Run one batched strategy-generation phase (3 attempts, pending-indices retry)."""
    for attempt in range(3):
        if not pending_indices:
            break
        msgs = [batch_messages[i] for i in pending_indices]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=msgs,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(m, temperature=config.generation_temperature_diverse)[0]
                for m in msgs
            ]
        still_pending: list[int] = []
        for request_idx, completion in zip(pending_indices, completions):
            item = prepared[request_idx]
            try:
                _extend_unique_strategies(
                    item["strategies"],  # type: ignore[arg-type]
                    parse_location_strategies(completion),
                    item["seen"],  # type: ignore[arg-type]
                    target_count,
                )
            except ValueError as exc:
                _log_location(
                    f"strategy proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending_indices[:] = still_pending


def generate_location_strategies(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
) -> list[str]:
    target_count = config.location_strategy_num_candidates
    retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
    strategies: list[str] = []
    seen: set[str] = set()
    _log_location(
        f"strategy proposal: retrieved={len(retrieved_entries)}, mutation={config.location_strategy_num_mutation}, "
        f"crossover={config.location_strategy_num_crossover}, diverse={config.location_strategy_num_diverse}, "
        f"library_size={len(library)}",
        config,
    )

    # Phase R: retrieved (no LLM call)
    _extend_unique_strategies(strategies, [e.strategy for e in retrieved_entries], seen, target_count)

    # Phase M: mutation (falls back to diverse if library empty)
    if config.location_strategy_num_mutation > 0:
        msgs = (
            _strategy_mutation_messages(retrieved_entries, belief_state, observations, config,
                                        config.location_strategy_num_mutation)
            if retrieved_entries
            else _strategy_diverse_messages(belief_state, observations, config,
                                            config.location_strategy_num_mutation)
        )
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "mutation")

    # Phase C: crossover (falls back to diverse if library empty)
    if config.location_strategy_num_crossover > 0:
        msgs = (
            _strategy_crossover_messages(retrieved_entries, belief_state, observations, config,
                                         config.location_strategy_num_crossover)
            if retrieved_entries
            else _strategy_diverse_messages(belief_state, observations, config,
                                            config.location_strategy_num_crossover)
        )
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "crossover")

    # Phase D: diverse — fills any remaining slots (including unfilled M/C)
    diverse_needed = target_count - len(strategies)
    if diverse_needed > 0:
        msgs = _strategy_diverse_messages(belief_state, observations, config, diverse_needed)
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "diverse")

    selected = strategies[:target_count]
    _log_location(f"strategy proposal: using {len(selected)} strategy/strategies", config)
    return selected


def generate_location_strategies_many(
    questioner: "Model",
    requests: list[tuple[LocationBeliefState, list[LocationObservation], LocationStrategyLibrary]],
    config: Config,
) -> list[list[str]]:
    if not requests:
        return []
    target_count = config.location_strategy_num_candidates
    prepared: list[dict[str, object]] = []
    for request_idx, (belief_state, observations, library) in enumerate(requests):
        retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
        strategies: list[str] = []
        seen: set[str] = set()
        _log_location(
            f"strategy proposal: retrieved={len(retrieved_entries)}, mutation={config.location_strategy_num_mutation}, "
            f"crossover={config.location_strategy_num_crossover}, diverse={config.location_strategy_num_diverse}, "
            f"library_size={len(library)}",
            config,
        )
        _extend_unique_strategies(strategies, [e.strategy for e in retrieved_entries], seen, target_count)
        prepared.append({
            "strategies": strategies,
            "seen": seen,
            "retrieved_entries": retrieved_entries,
            "observations": observations,
            "belief_state": belief_state,
        })

    # Phase M: mutation
    if config.location_strategy_num_mutation > 0:
        mutation_messages = [
            (
                _strategy_mutation_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_phase_batched(questioner, mutation_messages, pending, prepared, target_count, config, "mutation")

    # Phase C: crossover
    if config.location_strategy_num_crossover > 0:
        crossover_messages = [
            (
                _strategy_crossover_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_phase_batched(questioner, crossover_messages, pending, prepared, target_count, config, "crossover")

    # Phase D: diverse — fills any remaining slots
    pending = [i for i in range(len(prepared)) if len(prepared[i]["strategies"]) < target_count]  # type: ignore[arg-type]
    if pending:
        diverse_messages: list[Any] = [None] * len(prepared)
        for i in pending:
            diverse_messages[i] = _strategy_diverse_messages(
                prepared[i]["belief_state"],  # type: ignore[arg-type]
                prepared[i]["observations"],  # type: ignore[arg-type]
                config,
                target_count - len(prepared[i]["strategies"]),  # type: ignore[arg-type]
            )
        _strategy_phase_batched(questioner, diverse_messages, pending, prepared, target_count, config, "diverse")

    results: list[list[str]] = []
    for item in prepared:
        selected = (item["strategies"])[:target_count]  # type: ignore[index]
        _log_location(f"strategy proposal: using {len(selected)} strategy/strategies", config)
        results.append(selected)
    return results


def _strategy_root_phase_single(
    questioner: "Model",
    messages: list[dict[str, str]],
    candidates: list[LocationStrategyCandidate],
    seen: set[str],
    target_count: int,
    observations: list[LocationObservation],
    config: Config,
    phase_name: str,
) -> None:
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            parsed = parse_location_strategy_roots(
                completion, config.location_dim, tuple(config.location_query_bounds)
            )
            _extend_unique_strategy_candidates(candidates, parsed, seen, target_count, observations)
            return
        except ValueError as exc:
            _log_location(
                f"strategy+root proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )


def _strategy_root_phase_batched(
    questioner: "Model",
    batch_messages: list[list[dict[str, str]]],
    pending_indices: list[int],
    prepared: list[dict[str, object]],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    for attempt in range(3):
        if not pending_indices:
            break
        msgs = [batch_messages[i] for i in pending_indices]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=msgs,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(m, temperature=config.generation_temperature_diverse)[0]
                for m in msgs
            ]
        still_pending: list[int] = []
        for request_idx, completion in zip(pending_indices, completions):
            item = prepared[request_idx]
            item_observations: list[LocationObservation] = item["observations"]  # type: ignore[assignment]
            try:
                parsed = parse_location_strategy_roots(
                    completion, config.location_dim, tuple(config.location_query_bounds)
                )
                _extend_unique_strategy_candidates(
                    item["candidates"],  # type: ignore[arg-type]
                    parsed,
                    item["seen"],  # type: ignore[arg-type]
                    target_count,
                    item_observations,
                )
            except ValueError as exc:
                _log_location(
                    f"strategy+root proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending_indices[:] = still_pending


def generate_location_strategy_roots(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
) -> list[LocationStrategyCandidate]:
    target_count = config.location_strategy_num_candidates
    retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
    candidates: list[LocationStrategyCandidate] = []
    seen: set[str] = set()
    retrieved_candidates = [
        LocationStrategyCandidate(entry.strategy, entry.root_query)
        for entry in retrieved_entries
        if entry.root_query is not None
    ]
    _log_location(
        f"strategy+root proposal: retrieved={len(retrieved_candidates)}, "
        f"mutation={config.location_strategy_num_mutation}, crossover={config.location_strategy_num_crossover}, "
        f"diverse={config.location_strategy_num_diverse}, library_size={len(library)}",
        config,
    )

    # Phase R: retrieved (no LLM call)
    _extend_unique_strategy_candidates(candidates, retrieved_candidates, seen, target_count, observations)

    # Phase M: mutation (falls back to diverse if library empty)
    if config.location_strategy_num_mutation > 0:
        msgs = (
            _strategy_root_mutation_messages(retrieved_entries, belief_state, observations, config,
                                             config.location_strategy_num_mutation)
            if retrieved_entries
            else _strategy_root_diverse_messages(belief_state, observations, config,
                                                 config.location_strategy_num_mutation)
        )
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "mutation")

    # Phase C: crossover (falls back to diverse if library empty)
    if config.location_strategy_num_crossover > 0:
        msgs = (
            _strategy_root_crossover_messages(retrieved_entries, belief_state, observations, config,
                                              config.location_strategy_num_crossover)
            if retrieved_entries
            else _strategy_root_diverse_messages(belief_state, observations, config,
                                                 config.location_strategy_num_crossover)
        )
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "crossover")

    # Phase D: diverse — fills remaining slots
    diverse_needed = target_count - len(candidates)
    if diverse_needed > 0:
        msgs = _strategy_root_diverse_messages(belief_state, observations, config, diverse_needed)
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "diverse")

    selected = candidates[:target_count]
    _log_location(
        "strategy+root proposal: using "
        + "; ".join(f"{_format_location(candidate.root_query)} :: {candidate.strategy[:80]}"
                    for candidate in selected if candidate.root_query is not None),
        config,
    )
    return selected


def generate_location_strategy_roots_many(
    questioner: "Model",
    requests: list[tuple[LocationBeliefState, list[LocationObservation], LocationStrategyLibrary]],
    config: Config,
) -> list[list[LocationStrategyCandidate]]:
    if not requests:
        return []
    target_count = config.location_strategy_num_candidates
    prepared: list[dict[str, object]] = []
    for request_idx, (belief_state, observations, library) in enumerate(requests):
        retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
        candidates: list[LocationStrategyCandidate] = []
        seen: set[str] = set()
        retrieved_candidates = [
            LocationStrategyCandidate(entry.strategy, entry.root_query)
            for entry in retrieved_entries
            if entry.root_query is not None
        ]
        _log_location(
            f"strategy+root proposal: retrieved={len(retrieved_candidates)}, "
            f"mutation={config.location_strategy_num_mutation}, crossover={config.location_strategy_num_crossover}, "
            f"diverse={config.location_strategy_num_diverse}, library_size={len(library)}",
            config,
        )
        _extend_unique_strategy_candidates(candidates, retrieved_candidates, seen, target_count, observations)
        prepared.append({
            "candidates": candidates,
            "seen": seen,
            "retrieved_entries": retrieved_entries,
            "observations": observations,
            "belief_state": belief_state,
        })

    # Phase M: mutation
    if config.location_strategy_num_mutation > 0:
        mutation_messages = [
            (
                _strategy_root_mutation_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_root_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_root_phase_batched(questioner, mutation_messages, pending, prepared, target_count, config, "mutation")

    # Phase C: crossover
    if config.location_strategy_num_crossover > 0:
        crossover_messages = [
            (
                _strategy_root_crossover_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_root_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_root_phase_batched(questioner, crossover_messages, pending, prepared, target_count, config, "crossover")

    # Phase D: diverse — fills remaining slots
    pending = [i for i in range(len(prepared)) if len(prepared[i]["candidates"]) < target_count]  # type: ignore[arg-type]
    if pending:
        diverse_messages: list[Any] = [None] * len(prepared)
        for i in pending:
            diverse_messages[i] = _strategy_root_diverse_messages(
                prepared[i]["belief_state"],  # type: ignore[arg-type]
                prepared[i]["observations"],  # type: ignore[arg-type]
                config,
                target_count - len(prepared[i]["candidates"]),  # type: ignore[arg-type]
            )
        _strategy_root_phase_batched(questioner, diverse_messages, pending, prepared, target_count, config, "diverse")

    results: list[list[LocationStrategyCandidate]] = []
    for item in prepared:
        selected = (item["candidates"])[:target_count]  # type: ignore[index]
        _log_location(
            "strategy+root proposal: using "
            + "; ".join(f"{_format_location(candidate.root_query)} :: {candidate.strategy[:80]}"
                        for candidate in selected if candidate.root_query is not None),
            config,
        )
        results.append(selected)
    return results


def generate_strategy_locations_many(
    questioner: "Model",
    requests: list[_StrategyLocationRequest],
    config: Config,
) -> list[Location | None]:
    if not requests:
        return []

    bounds = tuple(config.location_query_bounds)
    batch_messages = [
        _strategy_location_messages(request.strategy, request.belief_state, request.observations, config)
        for request in requests
    ]
    results: list[Location | None] = [None] * len(requests)
    pending = list(range(len(requests)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_simple,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_simple)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} strategy-location completions, received {len(completions)}")
        still_pending: list[int] = []
        for request_idx, completion in zip(pending, completions):
            request = requests[request_idx]
            try:
                location = parse_strategy_location(completion, config.location_dim, bounds)
                if _is_repeated_location(location, request.observations):
                    raise ValueError("Location repeats a previous query")
                results[request_idx] = location
            except ValueError as exc:
                _log_location(
                    f"strategy location: attempt {attempt + 1}/3 failed ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending = still_pending

    return results


def generate_strategy_location(
    questioner: "Model",
    strategy: str,
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> Location | None:
    return generate_strategy_locations_many(
        questioner,
        [_StrategyLocationRequest(strategy, belief_state, list(observations))],
        config,
    )[0]


def _sample_source_hypothesis(
    belief_state: LocationBeliefState,
    rng: np.random.Generator,
) -> tuple[SourceConfig, float]:
    if not belief_state.hypotheses:
        raise ValueError("Cannot sample a source hypothesis from an empty belief state")
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_index = int(rng.choice(len(belief_state.hypotheses), p=probabilities))
    return belief_state.hypotheses[sampled_index], float(probabilities[sampled_index])


def _hypothesis_probability(
    belief_state: LocationBeliefState,
    hypothesis: SourceConfig,
    probability_floor: float = 1e-300,
) -> float:
    for candidate, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        if candidate == hypothesis:
            return max(float(probability), probability_floor)
    return probability_floor


def _full_rollout_observations(
    real_observations: list[LocationObservation],
    rollout: _StrategyRollout,
) -> list[LocationObservation]:
    return list(real_observations) + list(rollout.simulated_observations)


def _root_query_fingerprint(root_queries: list[Location | None]) -> str:
    formatted_queries = [
        _format_location(query)
        for query in root_queries
        if query is not None
    ]
    if not formatted_queries:
        return ""
    return max(set(formatted_queries), key=formatted_queries.count)


def _location_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _align_belief_state_to_support(
    belief_state: LocationBeliefState,
    support: list[SourceConfig],
) -> LocationBeliefState:
    probability_lookup = {
        hypothesis: float(probability)
        for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities)
    }
    probabilities = [max(probability_lookup.get(hypothesis, 0.0), 0.0) for hypothesis in support]
    total = sum(probabilities)
    if total <= 0.0 and support:
        probabilities = [1.0 / len(support)] * len(support)
    elif total > 0.0:
        probabilities = [float(probability / total) for probability in probabilities]
    return LocationBeliefState(list(support), probabilities)


def _rollout_entropy_reduction_score(
    rollout: _StrategyRollout,
    real_observations: list[LocationObservation],
    config: Config,
) -> float:
    support = _dedupe_source_configs(
        list(rollout.particle_support)
        + list(rollout.final_generated_hypotheses)
        + list(rollout.belief_state.hypotheses)
        + ([] if rollout.final_scoring_belief_state is None else list(rollout.final_scoring_belief_state.hypotheses))
        + [rollout.truth]
    )
    if len(support) <= 1:
        return 0.0
    start_state = _align_belief_state_to_support(rollout.start_belief_state, support)
    final_state = _align_belief_state_to_support(rollout.final_scoring_belief_state or rollout.belief_state, support)
    if config.location_strategy_discount_factor >= 1.0:
        return _location_entropy(start_state.probabilities) - _location_entropy(final_state.probabilities)
    if not rollout.simulated_observations:
        return _location_entropy(start_state.probabilities) - _location_entropy(final_state.probabilities)

    gamma = config.location_strategy_discount_factor
    score = 0.0
    previous_state = start_state
    full_history = list(real_observations)
    for step_idx, simulated_observation in enumerate(rollout.simulated_observations):
        full_history.append(simulated_observation)
        if step_idx == len(rollout.simulated_observations) - 1:
            next_state = final_state
        else:
            next_state = build_location_belief_state_unpruned(support, full_history, config)
        score += (gamma ** step_idx) * (
            _location_entropy(previous_state.probabilities) - _location_entropy(next_state.probabilities)
        )
        previous_state = next_state
    return score


def evaluate_location_strategies_by_rollout(
    questioner: "Model",
    strategies: list[str],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    rng: np.random.Generator,
    root_queries: list[Location | None] | None = None,
) -> list[LocationStrategyEvaluation]:
    if not strategies:
        return []
    if root_queries is None:
        root_queries = [None] * len(strategies)
    if len(root_queries) != len(strategies):
        raise ValueError("root_queries must have the same length as strategies")
    if not belief_state.hypotheses:
        return [
            LocationStrategyEvaluation(
                strategy,
                0.0,
                0.0,
                "" if root_query is None else _format_location(root_query),
                [0.0] * config.location_strategy_num_rollouts,
                root_query=root_query,
            )
            for strategy, root_query in zip(strategies, root_queries)
        ]

    rollouts: list[_StrategyRollout] = []
    for strategy_index, strategy in enumerate(strategies):
        for _rollout_idx in range(config.location_strategy_num_rollouts):
            truth, start_probability = _sample_source_hypothesis(belief_state, rng)
            rollouts.append(
                _StrategyRollout(
                    request_index=0,
                    strategy_index=strategy_index,
                    strategy=strategy,
                    truth=truth,
                    start_probability=start_probability,
                    start_belief_state=belief_state,
                    belief_state=belief_state,
                    particle_support=list(belief_state.hypotheses),
                    root_query=root_queries[strategy_index],
                )
            )

    for depth_idx in range(config.location_strategy_planning_depth):
        active_indices = [
            rollout_idx
            for rollout_idx, rollout in enumerate(rollouts)
            if rollout.belief_state.hypotheses
        ]
        if not active_indices:
            break

        fixed_root_indices = [
            rollout_idx
            for rollout_idx in active_indices
            if depth_idx == 0 and rollouts[rollout_idx].root_query is not None
        ]
        generated_indices = [rollout_idx for rollout_idx in active_indices if rollout_idx not in fixed_root_indices]
        locations_by_index: dict[int, Location] = {
            rollout_idx: rollouts[rollout_idx].root_query  # type: ignore[dict-item]
            for rollout_idx in fixed_root_indices
        }
        if generated_indices:
            location_requests = [
                _StrategyLocationRequest(
                    strategy=rollouts[rollout_idx].strategy,
                    belief_state=rollouts[rollout_idx].belief_state,
                    observations=_full_rollout_observations(observations, rollouts[rollout_idx]),
                )
                for rollout_idx in generated_indices
            ]
            locations = generate_strategy_locations_many(questioner, location_requests, config)
            locations_by_index.update(zip(generated_indices, locations))

        # Phase 1: add simulated observations for all rollouts that got a valid location
        stepped_indices: list[int] = []
        for rollout_idx in active_indices:
            location = locations_by_index[rollout_idx]
            rollout = rollouts[rollout_idx]
            if location is None:
                _log_location(
                    f"strategy rollout: depth {depth_idx + 1} location unavailable after retries; skipping rollout step",
                    config,
                )
                continue
            if depth_idx == 0 and rollout.root_query is None:
                rollout.root_query = location
            mean = signal_intensity_for_hypothesis(rollout.truth, location)
            observed_value = float(round(rng.normal(mean, config.location_noise_sd), 2))
            rollout.simulated_observations.append(
                LocationObservation(query=location, value=observed_value)
            )
            stepped_indices.append(rollout_idx)

        # Phase 2: batch-update belief states (respects location_posterior_mode)
        if stepped_indices:
            updated_states = build_location_posteriors_many(
                questioner,
                [rollouts[i].particle_support for i in stepped_indices],
                [_full_rollout_observations(observations, rollouts[i]) for i in stepped_indices],
                config,
                context_states=[rollouts[i].belief_state for i in stepped_indices],
                label=f"strategy rollout depth {depth_idx + 1} belief update",
                prune=False,
            )
            for rollout_idx, updated_state in zip(stepped_indices, updated_states):
                rollouts[rollout_idx].belief_state = updated_state
                rollouts[rollout_idx].simulated_belief_states.append(updated_state)

    final_refresh_indices = [
        rollout_idx
        for rollout_idx, rollout in enumerate(rollouts)
        if rollout.belief_state.hypotheses
    ]
    if final_refresh_indices:
        final_histories = [
            _full_rollout_observations(observations, rollouts[rollout_idx])
            for rollout_idx in final_refresh_indices
        ]
        final_generated_many = _generate_location_hypotheses_many(
            questioner,
            final_histories,
            [
                prompt_location_belief_state(rollouts[rollout_idx].belief_state, config)
                for rollout_idx in final_refresh_indices
            ],
            config,
            label="strategy rollout final belief refresh",
        )
        for rollout_idx, branch_history, generated_hypotheses in zip(
            final_refresh_indices,
            final_histories,
            final_generated_many,
        ):
            rollout = rollouts[rollout_idx]
            rollout.final_generated_hypotheses = generated_hypotheses
        final_supports = [
            _dedupe_source_configs(
                list(rollouts[rollout_idx].particle_support)
                + list(rollouts[rollout_idx].final_generated_hypotheses)
            )
            for rollout_idx in final_refresh_indices
        ]
        final_scoring_states = build_location_posteriors_many(
            questioner,
            final_supports,
            final_histories,
            config,
            context_states=[rollouts[rollout_idx].belief_state for rollout_idx in final_refresh_indices],
            label="strategy rollout final posterior scoring",
            prune=False,
        )
        for rollout_idx, final_scoring_state in zip(final_refresh_indices, final_scoring_states):
            rollouts[rollout_idx].final_scoring_belief_state = final_scoring_state
            rollouts[rollout_idx].belief_state = prune_location_beliefs(
                final_scoring_state,
                max_beliefs=config.location_max_total_beliefs,
            )

    scores_by_strategy: list[list[float]] = [[] for _strategy in strategies]
    root_queries_by_strategy: list[list[Location | None]] = [[] for _strategy in strategies]
    for rollout in rollouts:
        score = _rollout_entropy_reduction_score(rollout, observations, config)
        scores_by_strategy[rollout.strategy_index].append(float(score))
        root_queries_by_strategy[rollout.strategy_index].append(rollout.root_query)

    evaluations: list[LocationStrategyEvaluation] = []
    for strategy, scores, strategy_root_queries, fixed_root_query in zip(
        strategies,
        scores_by_strategy,
        root_queries_by_strategy,
        root_queries,
    ):
        if scores:
            mean_score = float(np.mean(scores))
            score_variance = float(np.var(scores))
        else:
            mean_score = 0.0
            score_variance = 0.0
        evaluations.append(
            LocationStrategyEvaluation(
                strategy=strategy,
                mean_score=mean_score,
                score_variance=score_variance,
                root_query_fingerprint=(
                    _format_location(fixed_root_query)
                    if fixed_root_query is not None
                    else _root_query_fingerprint(strategy_root_queries)
                ),
                rollout_scores=scores,
                root_query=fixed_root_query,
            )
        )
    return evaluations


def evaluate_location_strategies_by_rollout_many(
    questioner: "Model",
    requests: list[_StrategyEvaluationRequest],
    config: Config,
) -> list[list[LocationStrategyEvaluation]]:
    if not requests:
        return []

    evaluations_by_request: list[list[LocationStrategyEvaluation] | None] = [None] * len(requests)
    rollouts: list[_StrategyRollout] = []
    root_queries_by_request: list[list[Location | None]] = []
    for request_idx, request in enumerate(requests):
        root_queries = (
            [None] * len(request.strategies)
            if request.root_queries is None
            else list(request.root_queries)
        )
        if len(root_queries) != len(request.strategies):
            raise ValueError("root_queries must have the same length as strategies")
        root_queries_by_request.append(root_queries)
        if not request.strategies:
            evaluations_by_request[request_idx] = []
            continue
        if not request.belief_state.hypotheses:
            evaluations_by_request[request_idx] = [
                LocationStrategyEvaluation(
                    strategy,
                    0.0,
                    0.0,
                    "" if root_query is None else _format_location(root_query),
                    [0.0] * config.location_strategy_num_rollouts,
                    root_query=root_query,
                )
                for strategy, root_query in zip(request.strategies, root_queries)
            ]
            continue

        for strategy_index, strategy in enumerate(request.strategies):
            for _rollout_idx in range(config.location_strategy_num_rollouts):
                truth, start_probability = _sample_source_hypothesis(request.belief_state, request.rng)
                rollouts.append(
                    _StrategyRollout(
                        request_index=request_idx,
                        strategy_index=strategy_index,
                        strategy=strategy,
                        truth=truth,
                        start_probability=start_probability,
                        start_belief_state=request.belief_state,
                        belief_state=request.belief_state,
                        particle_support=list(request.belief_state.hypotheses),
                        root_query=root_queries[strategy_index],
                    )
                )

    for depth_idx in range(config.location_strategy_planning_depth):
        active_indices = [
            rollout_idx
            for rollout_idx, rollout in enumerate(rollouts)
            if rollout.belief_state.hypotheses
        ]
        if not active_indices:
            break

        fixed_root_indices = [
            rollout_idx
            for rollout_idx in active_indices
            if depth_idx == 0 and rollouts[rollout_idx].root_query is not None
        ]
        generated_indices = [rollout_idx for rollout_idx in active_indices if rollout_idx not in fixed_root_indices]
        locations_by_index: dict[int, Location] = {
            rollout_idx: rollouts[rollout_idx].root_query  # type: ignore[dict-item]
            for rollout_idx in fixed_root_indices
        }
        if generated_indices:
            location_requests = [
                _StrategyLocationRequest(
                    strategy=rollouts[rollout_idx].strategy,
                    belief_state=rollouts[rollout_idx].belief_state,
                    observations=_full_rollout_observations(
                        requests[rollouts[rollout_idx].request_index].observations,
                        rollouts[rollout_idx],
                    ),
                )
                for rollout_idx in generated_indices
            ]
            locations = generate_strategy_locations_many(questioner, location_requests, config)
            locations_by_index.update(zip(generated_indices, locations))

        # Phase 1: add simulated observations for all rollouts that got a valid location
        stepped_indices_many: list[int] = []
        for rollout_idx in active_indices:
            location = locations_by_index[rollout_idx]
            rollout = rollouts[rollout_idx]
            request = requests[rollout.request_index]
            if location is None:
                _log_location(
                    f"strategy rollout: depth {depth_idx + 1} location unavailable after retries; skipping rollout step",
                    config,
                )
                continue
            if depth_idx == 0 and rollout.root_query is None:
                rollout.root_query = location
            mean = signal_intensity_for_hypothesis(rollout.truth, location)
            observed_value = float(round(request.rng.normal(mean, config.location_noise_sd), 2))
            rollout.simulated_observations.append(
                LocationObservation(query=location, value=observed_value)
            )
            stepped_indices_many.append(rollout_idx)

        # Phase 2: batch-update belief states (respects location_posterior_mode)
        if stepped_indices_many:
            updated_states_many = build_location_posteriors_many(
                questioner,
                [rollouts[i].particle_support for i in stepped_indices_many],
                [
                    _full_rollout_observations(requests[rollouts[i].request_index].observations, rollouts[i])
                    for i in stepped_indices_many
                ],
                config,
                context_states=[rollouts[i].belief_state for i in stepped_indices_many],
                label=f"strategy rollout depth {depth_idx + 1} belief update",
                prune=False,
            )
            for rollout_idx, updated_state in zip(stepped_indices_many, updated_states_many):
                rollouts[rollout_idx].belief_state = updated_state
                rollouts[rollout_idx].simulated_belief_states.append(updated_state)

    final_refresh_indices = [
        rollout_idx
        for rollout_idx, rollout in enumerate(rollouts)
        if rollout.belief_state.hypotheses
    ]
    if final_refresh_indices:
        final_histories = [
            _full_rollout_observations(
                requests[rollouts[rollout_idx].request_index].observations,
                rollouts[rollout_idx],
            )
            for rollout_idx in final_refresh_indices
        ]
        final_generated_many = _generate_location_hypotheses_many(
            questioner,
            final_histories,
            [
                prompt_location_belief_state(rollouts[rollout_idx].belief_state, config)
                for rollout_idx in final_refresh_indices
            ],
            config,
            label="strategy rollout final belief refresh",
        )
        for rollout_idx, generated_hypotheses in zip(final_refresh_indices, final_generated_many):
            rollouts[rollout_idx].final_generated_hypotheses = generated_hypotheses
        final_supports = [
            _dedupe_source_configs(
                list(rollouts[rollout_idx].particle_support)
                + list(rollouts[rollout_idx].final_generated_hypotheses)
            )
            for rollout_idx in final_refresh_indices
        ]
        final_scoring_states = build_location_posteriors_many(
            questioner,
            final_supports,
            final_histories,
            config,
            context_states=[rollouts[rollout_idx].belief_state for rollout_idx in final_refresh_indices],
            label="strategy rollout final posterior scoring",
            prune=False,
        )
        for rollout_idx, final_scoring_state in zip(final_refresh_indices, final_scoring_states):
            rollouts[rollout_idx].final_scoring_belief_state = final_scoring_state
            rollouts[rollout_idx].belief_state = prune_location_beliefs(
                final_scoring_state,
                max_beliefs=config.location_max_total_beliefs,
            )

    scores_by_request: list[list[list[float]]] = [
        [[] for _strategy in request.strategies]
        for request in requests
    ]
    roots_by_request: list[list[list[Location | None]]] = [
        [[] for _strategy in request.strategies]
        for request in requests
    ]
    for rollout in rollouts:
        score = _rollout_entropy_reduction_score(
            rollout,
            requests[rollout.request_index].observations,
            config,
        )
        scores_by_request[rollout.request_index][rollout.strategy_index].append(float(score))
        roots_by_request[rollout.request_index][rollout.strategy_index].append(rollout.root_query)

    for request_idx, request in enumerate(requests):
        if evaluations_by_request[request_idx] is not None:
            continue
        evaluations: list[LocationStrategyEvaluation] = []
        root_queries = root_queries_by_request[request_idx]
        for strategy, scores, strategy_root_queries, fixed_root_query in zip(
            request.strategies,
            scores_by_request[request_idx],
            roots_by_request[request_idx],
            root_queries,
        ):
            if scores:
                mean_score = float(np.mean(scores))
                score_variance = float(np.var(scores))
            else:
                mean_score = 0.0
                score_variance = 0.0
            evaluations.append(
                LocationStrategyEvaluation(
                    strategy=strategy,
                    mean_score=mean_score,
                    score_variance=score_variance,
                    root_query_fingerprint=(
                        _format_location(fixed_root_query)
                        if fixed_root_query is not None
                        else _root_query_fingerprint(strategy_root_queries)
                    ),
                    rollout_scores=scores,
                    root_query=fixed_root_query,
                )
            )
        evaluations_by_request[request_idx] = evaluations
    return [evaluations or [] for evaluations in evaluations_by_request]


def _strategy_entries_from_evaluations(
    evaluations: list[LocationStrategyEvaluation],
    round_index: int,
) -> list[LocationStrategyEntry]:
    return [
        LocationStrategyEntry(
            strategy=evaluation.strategy,
            mean_score=evaluation.mean_score,
            score_variance=evaluation.score_variance,
            root_query_fingerprint=evaluation.root_query_fingerprint,
            round_index=round_index,
            root_query=evaluation.root_query,
        )
        for evaluation in evaluations
    ]


def choose_location_with_strategy_rollouts(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
    rng: np.random.Generator,
    round_index: int,
    *,
    fixed_root: bool = False,
) -> tuple[Location | None, float, LocationStrategyEvaluation | None]:
    if fixed_root:
        strategy_candidates = generate_location_strategy_roots(questioner, belief_state, observations, library, config)
        strategies = [candidate.strategy for candidate in strategy_candidates]
        root_queries = [candidate.root_query for candidate in strategy_candidates]
    else:
        strategies = generate_location_strategies(questioner, belief_state, observations, library, config)
        root_queries = None
    evaluations = evaluate_location_strategies_by_rollout(
        questioner,
        strategies,
        belief_state,
        observations,
        config,
        rng,
        root_queries=root_queries,
    )
    library.replace_entries(_strategy_entries_from_evaluations(evaluations, round_index))
    if not evaluations:
        return None, 0.0, None

    best_evaluation = max(evaluations, key=lambda evaluation: evaluation.mean_score)
    _log_location(
        "strategy rollout: best strategy "
        f"score={best_evaluation.mean_score:.6f}, variance={best_evaluation.score_variance:.6f}, "
        f"root={best_evaluation.root_query_fingerprint!r}",
        config,
    )
    if fixed_root and best_evaluation.root_query is not None:
        _log_location(f"strategy+root: asking fixed root query {_format_location(best_evaluation.root_query)}", config)
        location: Location | None = best_evaluation.root_query
    else:
        location = generate_strategy_location(
            questioner,
            best_evaluation.strategy,
            belief_state,
            observations,
            config,
        )
    return location, best_evaluation.mean_score, best_evaluation


def choose_locations_with_strategy_rollouts_many(
    questioner: "Model",
    states: list[_LocationTrialState],
    config: Config,
    round_index: int,
    *,
    fixed_root: bool = False,
) -> list[tuple[Location | None, float, LocationStrategyEvaluation | None]]:
    if not states:
        return []
    for state in states:
        if state.belief_state is None:
            raise ValueError("Strategy trial state is missing a belief state")
        if state.strategy_library is None:
            raise ValueError("Strategy trial state is missing a strategy library")

    strategy_requests = [
        (state.belief_state, state.observations, state.strategy_library)  # type: ignore[arg-type]
        for state in states
    ]
    if fixed_root:
        strategy_candidates_many = generate_location_strategy_roots_many(questioner, strategy_requests, config)
        strategies_many = [[candidate.strategy for candidate in candidates] for candidates in strategy_candidates_many]
        root_queries_many = [[candidate.root_query for candidate in candidates] for candidates in strategy_candidates_many]
    else:
        strategies_many = generate_location_strategies_many(questioner, strategy_requests, config)
        root_queries_many = [None for _state in states]

    evaluation_requests = [
        _StrategyEvaluationRequest(
            strategies=strategies,
            belief_state=state.belief_state,  # type: ignore[arg-type]
            observations=list(state.observations),
            rng=state.rng,
            root_queries=root_queries,
        )
        for state, strategies, root_queries in zip(states, strategies_many, root_queries_many)
    ]
    evaluations_many = evaluate_location_strategies_by_rollout_many(questioner, evaluation_requests, config)
    for state, evaluations in zip(states, evaluations_many):
        state.strategy_library.replace_entries(_strategy_entries_from_evaluations(evaluations, round_index))  # type: ignore[union-attr]

    results: list[tuple[Location | None, float, LocationStrategyEvaluation | None]] = []
    selected_location_requests: list[_StrategyLocationRequest] = []
    selected_location_indices: list[int] = []
    for idx, (state, evaluations) in enumerate(zip(states, evaluations_many)):
        if not evaluations:
            results.append((None, 0.0, None))
            continue

        best_evaluation = max(evaluations, key=lambda evaluation: evaluation.mean_score)
        _log_location(
            "strategy rollout: best strategy "
            f"score={best_evaluation.mean_score:.6f}, variance={best_evaluation.score_variance:.6f}, "
            f"root={best_evaluation.root_query_fingerprint!r}",
            config,
        )
        if fixed_root and best_evaluation.root_query is not None:
            _log_location(f"strategy+root: asking fixed root query {_format_location(best_evaluation.root_query)}", config)
            results.append((best_evaluation.root_query, best_evaluation.mean_score, best_evaluation))
        else:
            results.append((None, best_evaluation.mean_score, best_evaluation))
            selected_location_indices.append(idx)
            selected_location_requests.append(
                _StrategyLocationRequest(
                    best_evaluation.strategy,
                    state.belief_state,  # type: ignore[arg-type]
                    list(state.observations),
                )
            )

    if selected_location_requests:
        selected_locations = generate_strategy_locations_many(questioner, selected_location_requests, config)
        for result_idx, location in zip(selected_location_indices, selected_locations):
            _old_location, score, evaluation = results[result_idx]
            results[result_idx] = (location, score, evaluation)

    return results
