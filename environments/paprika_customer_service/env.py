"""Core environment adapter for Paprika customer-service troubleshooting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from methods.categorical_eig import CategoricalEIG, FullTwoStepCategoricalEIG

from .data import load_paprika_tasks
from .parsing import parse_distribution, parse_json_object, parse_string_list
from .prompts import (
    candidate_messages,
    customer_messages,
    hypothesis_messages,
    judge_messages,
    likelihood_messages,
    mapping_messages,
    refinement_messages,
    filtering_messages,
)
from .types import PaprikaAction, PaprikaObservation, PaprikaTask


def _complete(model: Any, messages: list[dict[str, str]], temperature: float) -> str:
    responses = model.chat_complete(messages, temperature=temperature, num_responses=1)
    if not responses:
        raise ValueError("Model returned no response")
    return responses[0]


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if clean and clean.casefold() not in seen:
            result.append(clean)
            seen.add(clean.casefold())
    return result


def _dedupe_indices(values: Sequence[Any], upper_bound: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < upper_bound and index not in seen:
            result.append(index)
            seen.add(index)
    return result


def _normalized_action_kind(query: str, proposed_kind: str) -> str:
    del proposed_kind
    normalized = query.strip().casefold()
    diagnostic_prefixes = (
        "is ", "are ", "does ", "do ", "did ", "has ", "have ", "can you ",
        "could you ", "would you ", "check ", "please check ", "inspect ",
        "please inspect ", "verify ", "please verify ", "listen ", "please listen ",
    )
    if normalized.startswith(diagnostic_prefixes) or normalized.endswith("?"):
        return "diagnostic"
    solution_verbs = (
        "replace", "recalibrate", "calibrate", "reset", "restart", "reinstall",
        "update", "enable", "disable", "reconnect", "repair", "refill", "clear",
        "clean", "adjust", "increase", "decrease", "remove", "straighten", "close",
        "open", "tighten",
    )
    words = normalized.replace("/", " ").split()
    is_solution = any(verb in words for verb in solution_verbs) or any(
        word.startswith(("recalibrat", "calibrat")) for word in words
    )
    return "solution" if is_solution else "diagnostic"


def _is_uncertainty_outcome(value: str) -> bool:
    normalized = value.casefold()
    markers = (
        "not checked", "cannot determine", "can't determine", "do not know",
        "don't know", "not sure", "unable to check", "cannot check",
    )
    return any(marker in normalized for marker in markers)


def _reply_explicitly_uncertain(reply: str) -> bool:
    normalized = reply.casefold()
    markers = (
        "did not check", "didn't check", "have not checked", "haven't checked",
        "do not know", "don't know", "not sure", "cannot tell", "can't tell",
        "cannot determine", "can't determine", "unable to check", "cannot check",
        "can't check", "unable to perform", "cannot perform", "can't perform",
    )
    return any(marker in normalized for marker in markers)


class PaprikaCustomerServiceEnvironment(
    Environment[PaprikaTask, str, PaprikaAction, PaprikaObservation]
):
    """Paprika adapter preserving the released scenario, solution, and success rule."""

    def __init__(self, config: Any, answerer: Any) -> None:
        self.config = config
        self.answerer = answerer
        self.questioner: Any | None = None
        self.tasks: list[PaprikaTask] = []
        self._active_task: PaprikaTask | None = None
        self._active_batch_scenarios: tuple[str, ...] = ()
        self._scenario_by_support: dict[tuple[str, ...], str] = {}
        self._likelihood_cache: dict[tuple[str, PaprikaAction], tuple[float, ...]] = {}
        self._shared_cache_hits = 0
        self._shared_cache_misses = 0
        self._structured_parse_retries = 0
        self._structured_parse_failures = 0

    @property
    def name(self) -> str:
        return "paprika_customer_service"

    def configure_for_run(self, config: Any) -> "PaprikaCustomerServiceEnvironment":
        path = getattr(config, "paprika_data_path", None)
        if not path:
            raise ValueError(
                "paprika_data_path must point to Paprika's customer_service.json; "
                "run scripts/fetch_paprika.py first"
            )
        self.tasks = load_paprika_tasks(
            path,
            split=getattr(config, "paprika_split", "eval"),
            verify_official_hash=bool(getattr(config, "paprika_verify_official_hash", True)),
        )
        offset = int(getattr(config, "paprika_task_offset", 0))
        count = int(getattr(config, "paprika_num_trials", 5))
        self.tasks = self.tasks[offset : offset + count]
        if len(self.tasks) != count:
            raise ValueError(f"Requested {count} Paprika tasks at offset {offset}, found {len(self.tasks)}")
        return self

    def validate_config(self, config: Any) -> None:
        for name in ("paprika_num_trials", "paprika_num_rounds", "paprika_num_hypotheses", "paprika_num_candidates"):
            value = getattr(config, name, None)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if getattr(config, "paprika_split", "eval") not in {"train", "eval"}:
            raise ValueError("paprika_split must be 'train' or 'eval'")

    def set_questioner(self, model: Any) -> None:
        self.questioner = model

    def prepare_action_batch(self, hidden_states: Sequence[PaprikaTask]) -> None:
        self._active_batch_scenarios = tuple(task.scenario for task in hidden_states)

    def _questioner(self) -> Any:
        if self.questioner is None:
            raise RuntimeError("Paprika environment has no attached questioner")
        return self.questioner

    def _cached_complete(
        self,
        model: Any,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        namespace: str,
    ) -> str:
        if not bool(getattr(self.config, "paprika_shared_call_cache_enabled", True)):
            self._shared_cache_misses += 1
            return _complete(model, messages, temperature)
        cache = getattr(model, "_paprika_shared_call_cache", None)
        if cache is None:
            cache = {}
            setattr(model, "_paprika_shared_call_cache", cache)
        key = (
            namespace,
            float(temperature),
            json.dumps(messages, sort_keys=True, ensure_ascii=True),
        )
        if key in cache:
            self._shared_cache_hits += 1
            return cache[key]
        response = _complete(model, messages, temperature)
        cache[key] = response
        self._shared_cache_misses += 1
        return response

    def _cached_complete_many(
        self,
        model: Any,
        batch_messages: Sequence[list[dict[str, str]]],
        temperature: float,
        *,
        namespace: str,
    ) -> list[str]:
        if not batch_messages:
            return []
        enabled = bool(getattr(self.config, "paprika_shared_call_cache_enabled", True))
        cache = getattr(model, "_paprika_shared_call_cache", None)
        if cache is None:
            cache = {}
            setattr(model, "_paprika_shared_call_cache", cache)
        keys = [
            (
                namespace,
                float(temperature),
                json.dumps(messages, sort_keys=True, ensure_ascii=True),
            )
            for messages in batch_messages
        ]
        results: list[str | None] = [None] * len(keys)
        missing_positions: list[int] = []
        missing_messages: list[list[dict[str, str]]] = []
        for index, (key, messages) in enumerate(zip(keys, batch_messages)):
            if enabled and key in cache:
                results[index] = cache[key]
                self._shared_cache_hits += 1
            else:
                missing_positions.append(index)
                missing_messages.append(messages)
        if missing_messages:
            responses = model.chat_complete_messages_batched(
                missing_messages,
                temperature=temperature,
                block_size=int(getattr(self.config, "batched_block_size", 50)),
                max_new_tokens=getattr(self.config, "location_max_new_tokens", None),
            )
            if len(responses) != len(missing_messages):
                raise ValueError("Batched Paprika completion returned the wrong response count")
            for position, response in zip(missing_positions, responses):
                results[position] = response
                if enabled:
                    cache[keys[position]] = response
                self._shared_cache_misses += 1
        return [str(result) for result in results]

    @staticmethod
    def _repair_messages(
        messages: list[dict[str, str]], response: str, error: ValueError
    ) -> list[dict[str, str]]:
        return list(messages) + [
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "That response could not be parsed: "
                    f"{error}. Return only corrected strict JSON matching the requested schema."
                ),
            },
        ]

    def _complete_parsed(
        self,
        model: Any,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        namespace: str,
        parser: Any,
    ) -> Any:
        response = self._cached_complete(
            model, messages, temperature, namespace=namespace
        )
        current_messages = messages
        maximum = int(getattr(self.config, "paprika_structured_max_retries", 2))
        for attempt in range(maximum + 1):
            try:
                return parser(response)
            except ValueError as exc:
                if attempt >= maximum:
                    self._structured_parse_failures += 1
                    raise
                self._structured_parse_retries += 1
                current_messages = self._repair_messages(current_messages, response, exc)
                response = self._cached_complete(
                    model,
                    current_messages,
                    temperature,
                    namespace=f"{namespace}:retry:{attempt + 1}",
                )
        raise AssertionError("unreachable")

    def _parse_many_with_retries(
        self,
        model: Any,
        messages: Sequence[list[dict[str, str]]],
        responses: Sequence[str],
        temperature: float,
        *,
        namespace: str,
        parsers: Sequence[Any],
    ) -> list[Any]:
        results: list[Any | None] = [None] * len(responses)
        current_messages = [list(item) for item in messages]
        current_responses = list(responses)
        active = list(range(len(responses)))
        maximum = int(getattr(self.config, "paprika_structured_max_retries", 2))
        for attempt in range(maximum + 1):
            failed: list[tuple[int, ValueError]] = []
            for index in active:
                try:
                    results[index] = parsers[index](current_responses[index])
                except ValueError as exc:
                    failed.append((index, exc))
            if not failed:
                return results
            if attempt >= maximum:
                self._structured_parse_failures += len(failed)
                raise failed[0][1]
            self._structured_parse_retries += len(failed)
            retry_indices = [index for index, _error in failed]
            retry_messages = []
            for index, error in failed:
                current_messages[index] = self._repair_messages(
                    current_messages[index], current_responses[index], error
                )
                retry_messages.append(current_messages[index])
            retry_responses = self._cached_complete_many(
                model,
                retry_messages,
                temperature,
                namespace=f"{namespace}:retry:{attempt + 1}",
            )
            for index, response in zip(retry_indices, retry_responses):
                current_responses[index] = response
            active = retry_indices
        raise AssertionError("unreachable")

    def trial_count(self, config: Any) -> int:
        return int(getattr(config, "paprika_num_trials", 5))

    def round_count(self, config: Any) -> int:
        return int(getattr(config, "paprika_num_rounds", 20))

    def run_seed(self, config: Any) -> int | None:
        return getattr(config, "paprika_seed", None)

    def sample_hidden_state(self, rng: np.random.Generator) -> PaprikaTask:
        task = self.tasks[int(rng.integers(0, len(self.tasks)))]
        self._active_task = task
        return task

    def sample_hidden_state_for_trial(self, trial_index: int, rng: np.random.Generator) -> PaprikaTask:
        del rng
        task = self.tasks[trial_index]
        self._active_task = task
        return task

    def sample_hidden_states_for_trials(self, trial_indices: Sequence[int], rng: np.random.Generator) -> list[PaprikaTask]:
        del rng
        return [self.tasks[index] for index in trial_indices]

    def log_prior(self, hypothesis: str) -> float:
        return 0.0

    def log_likelihood(self, hypothesis: str, action: PaprikaAction, observation: PaprikaObservation) -> float:
        if not observation.mapped_cleanly or observation.mapped_outcome is None:
            return 0.0
        probabilities = self._likelihood_for(hypothesis, action)
        index = action.outcomes.index(observation.mapped_outcome)
        return math.log(max(probabilities[index], 1e-12))

    def _likelihood_for(self, hypothesis: str, action: PaprikaAction) -> tuple[float, ...]:
        key = (hypothesis, action)
        if key not in self._likelihood_cache:
            self.outcome_likelihoods([hypothesis], action)
        return self._likelihood_cache[key]

    def outcome_likelihoods(self, hypotheses: Sequence[str], action: PaprikaAction) -> np.ndarray:
        return self.outcome_likelihoods_many([(hypotheses, action)])[0]

    def outcome_likelihoods_many(
        self,
        requests: Sequence[tuple[Sequence[str], PaprikaAction]],
    ) -> list[np.ndarray]:
        missing_keys: list[tuple[str, PaprikaAction]] = []
        missing_messages: list[list[dict[str, str]]] = []
        seen_missing: set[tuple[str, PaprikaAction]] = set()
        for hypotheses, action in requests:
            for hypothesis in hypotheses:
                key = (hypothesis, action)
                if key not in self._likelihood_cache and key not in seen_missing:
                    seen_missing.add(key)
                    missing_keys.append(key)
                    missing_messages.append(likelihood_messages(hypothesis, action))
        if missing_messages:
            temperature = float(
                getattr(self.config, "generation_temperature_simple", 0.0)
            )
            responses = self._cached_complete_many(
                self._questioner(),
                missing_messages,
                temperature,
                namespace="questioner:likelihood",
            )
            parsed = self._parse_many_with_retries(
                self._questioner(),
                missing_messages,
                responses,
                temperature,
                namespace="questioner:likelihood",
                parsers=[
                    (lambda text, outcomes=action.outcomes: parse_distribution(text, outcomes))
                    for _hypothesis, action in missing_keys
                ],
            )
            for key, distribution in zip(missing_keys, parsed):
                self._likelihood_cache[key] = distribution
        return [
            np.asarray([self._likelihood_cache[(hypothesis, action)] for hypothesis in hypotheses])
            for hypotheses, action in requests
        ]

    def log_likelihood_many(self, hypotheses: Sequence[str], action: PaprikaAction, observation: PaprikaObservation) -> np.ndarray:
        if not observation.mapped_cleanly or observation.mapped_outcome is None:
            return np.zeros(len(hypotheses), dtype=float)
        outcome_index = action.outcomes.index(observation.mapped_outcome)
        matrix = self.outcome_likelihoods(hypotheses, action)
        return np.log(np.maximum(matrix[:, outcome_index], 1e-12))

    def _initial_for_task(self, task: PaprikaTask, model: Any, config: Any) -> BeliefState[str]:
        count = int(getattr(config, "paprika_num_hypotheses", 12))
        messages = hypothesis_messages(task.scenario, count)
        hypotheses = self._complete_parsed(
            model,
            messages,
            float(getattr(config, "generation_temperature_diverse", 1.0)),
            namespace="questioner:hypotheses",
            parser=lambda text: parse_string_list(
                text, "hypotheses", minimum=count, maximum=count
            ),
        )
        state = BeliefState.uniform(hypotheses)
        self._scenario_by_support[state.hypotheses] = task.scenario
        return state

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[str]:
        self.set_questioner(model)
        if self._active_task is None:
            raise RuntimeError("A Paprika task must be selected before belief initialization")
        return self._initial_for_task(self._active_task, model, config)

    def initial_belief_states(self, trial_indices: Sequence[int], model: Any, config: Any) -> list[BeliefState[str]]:
        self.set_questioner(model)
        return [self._initial_for_task(self.tasks[index], model, config) for index in trial_indices]

    def update_belief_state(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], model: Any, config: Any) -> BeliefState[str]:
        if (
            not history
            or not history[-1][1].mapped_cleanly
            or history[-1][1].mapped_outcome is None
        ):
            return belief_state
        scenario = history[0][0].scenario
        hypotheses = list(belief_state.hypotheses)
        if bool(getattr(config, "paprika_belief_refresh_enabled", True)):
            count = int(getattr(config, "paprika_num_refresh_hypotheses", 6))
            messages = refinement_messages(scenario, hypotheses, history, count)
            refined = self._complete_parsed(
                model,
                messages,
                float(getattr(config, "generation_temperature_diverse", 1.0)),
                namespace="questioner:hypothesis_refinement",
                parser=lambda text: parse_string_list(
                    text,
                    "refined_hypotheses",
                    minimum=count,
                    maximum=count,
                ),
            )
            hypotheses = _dedupe(hypotheses + refined)
            filter_messages = filtering_messages(scenario, hypotheses, history)
            filter_response = self._complete_parsed(
                model,
                filter_messages,
                float(getattr(config, "generation_temperature_simple", 0.0)),
                namespace="questioner:hypothesis_filter",
                parser=parse_json_object,
            )
            indices = filter_response.get("keep_indices")
            if not isinstance(indices, list):
                raise ValueError("Hypothesis filter must return keep_indices")
            kept_indices = _dedupe_indices(indices, len(hypotheses))
            if not kept_indices:
                raise ValueError("Hypothesis filter rejected every candidate")
            hypotheses = [hypotheses[index] for index in kept_indices]

        log_weights = np.full(len(hypotheses), -math.log(len(hypotheses)), dtype=float)
        for action, observation in history:
            if observation.mapped_cleanly and observation.mapped_outcome is not None:
                log_weights += self.log_likelihood_many(hypotheses, action, observation)
        updated = BeliefState.from_log_scores(hypotheses, log_weights)
        maximum = int(getattr(config, "paprika_max_hypotheses", 24))
        if len(updated.hypotheses) > maximum:
            order = np.argsort(-np.asarray(updated.probabilities))[:maximum]
            updated = BeliefState(
                hypotheses=tuple(updated.hypotheses[int(index)] for index in order),
                probabilities=tuple(updated.probabilities[int(index)] for index in order),
            )
        self._scenario_by_support[updated.hypotheses] = scenario
        return updated

    def _parse_candidates(self, text: str, scenario: str, history: Sequence[tuple[PaprikaAction, PaprikaObservation]], expected: int) -> list[PaprikaAction]:
        raw = parse_json_object(text).get("candidates")
        if not isinstance(raw, list):
            raise ValueError("JSON field 'candidates' must be a list")
        transcript = tuple((action.query, observation.reply) for action, observation in history)
        actions: list[PaprikaAction] = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not isinstance(item.get("outcomes"), list):
                continue
            try:
                actions.append(
                    PaprikaAction(
                        item["query"],
                        tuple(item["outcomes"]),
                        scenario,
                        transcript,
                        kind=_normalized_action_kind(item["query"], item.get("kind", "")),
                    )
                )
            except (TypeError, ValueError):
                continue
        unique: dict[str, PaprikaAction] = {action.query.casefold(): action for action in actions}
        result = list(unique.values())
        if len(result) != expected:
            raise ValueError(f"Expected {expected} valid Paprika candidates, parsed {len(result)}")
        return result

    def generate_candidate_actions(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], model: Any, config: Any) -> list[PaprikaAction]:
        scenario = history[0][0].scenario if history else self._scenario_by_support.get(belief_state.hypotheses)
        if not scenario:
            raise RuntimeError("Could not associate Paprika belief support with a scenario")
        count = int(getattr(config, "paprika_num_candidates", 5))
        messages = candidate_messages(scenario, belief_state.hypotheses, history, count)
        return self._complete_parsed(
            model,
            messages,
            float(getattr(config, "generation_temperature_diverse", 1.0)),
            namespace="questioner:candidates",
            parser=lambda text: self._parse_candidates(text, scenario, history, count),
        )

    def generate_candidate_actions_many(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[PaprikaAction, PaprikaObservation]]],
        model: Any,
        config: Any,
    ) -> list[list[PaprikaAction]]:
        if len(belief_states) != len(histories):
            raise ValueError("belief_states and histories must have the same length")
        count = int(getattr(config, "paprika_num_candidates", 5))
        scenarios: list[str] = []
        messages: list[list[dict[str, str]]] = []
        for belief_state, history in zip(belief_states, histories):
            scenario = history[0][0].scenario if history else self._scenario_by_support.get(belief_state.hypotheses)
            if not scenario:
                raise RuntimeError("Could not associate Paprika belief support with a scenario")
            scenarios.append(scenario)
            messages.append(candidate_messages(scenario, belief_state.hypotheses, history, count))
        responses = self._cached_complete_many(
            model,
            messages,
            float(getattr(config, "generation_temperature_diverse", 1.0)),
            namespace="questioner:candidates",
        )
        return self._parse_many_with_retries(
            model,
            messages,
            responses,
            float(getattr(config, "generation_temperature_diverse", 1.0)),
            namespace="questioner:candidates",
            parsers=[
                (lambda text, scenario=scenario, history=history: self._parse_candidates(text, scenario, history, count))
                for scenario, history in zip(scenarios, histories)
            ],
        )

    def observe(self, action: PaprikaAction, hidden_state: PaprikaTask, rng: np.random.Generator) -> PaprikaObservation:
        del rng
        reply = self._cached_complete(self.answerer, customer_messages(action, hidden_state.solution), float(getattr(self.config, "answer_temperature", 0.7)), namespace="answerer:customer").strip()
        customer_goal = reply.casefold() == "goal reached"
        goal = customer_goal
        if not customer_goal and action.kind == "solution":
            judge = self._cached_complete(self._questioner(), judge_messages(hidden_state.scenario, hidden_state.solution, action.query), 0.0, namespace="questioner:success_judge")
            goal = "<VALID>" in judge and "<NOTVALID>" not in judge
        if customer_goal:
            return PaprikaObservation(reply=reply, mapped_outcome=None, mapped_cleanly=True, goal_reached=True)
        map_messages = mapping_messages(reply, action.outcomes)
        mapping = self._complete_parsed(
            self._questioner(),
            map_messages,
            0.0,
            namespace="questioner:outcome_mapper",
            parser=parse_json_object,
        )
        selected = mapping.get("outcome")
        clean = mapping.get("clean") is True and isinstance(selected, str)
        canonical = next((outcome for outcome in action.outcomes if clean and outcome.casefold() == selected.strip().casefold()), None)
        if canonical is not None and _is_uncertainty_outcome(canonical) and not _reply_explicitly_uncertain(reply):
            alternatives = tuple(outcome for outcome in action.outcomes if not _is_uncertainty_outcome(outcome))
            repaired = self._complete_parsed(
                self._questioner(),
                mapping_messages(reply, alternatives, uncertainty_forbidden=True),
                0.0,
                namespace="questioner:outcome_mapper_repair",
                parser=parse_json_object,
            )
            selected = repaired.get("outcome")
            clean = repaired.get("clean") is True and isinstance(selected, str)
            canonical = next(
                (
                    outcome
                    for outcome in alternatives
                    if clean and outcome.casefold() == selected.strip().casefold()
                ),
                None,
            )
        return PaprikaObservation(reply=reply, mapped_outcome=canonical, mapped_cleanly=canonical is not None, goal_reached=goal)

    def round_metrics(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], hidden_state: PaprikaTask) -> dict[str, float]:
        latest = history[-1][1]
        solved = any(observation.goal_reached for _action, observation in history)
        clean_count = sum(observation.mapped_cleanly for _action, observation in history)
        exact_mass = sum(probability for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities) if hypothesis.casefold() == hidden_state.solution.casefold())
        return {"resolved": float(solved), "turns_used": float(len(history)), "answer_set_coverage": clean_count / len(history), "latest_answer_mapped_cleanly": float(latest.mapped_cleanly), "true_solution_exact_mass": float(exact_mass), "shared_call_cache_hits": float(self._shared_cache_hits), "shared_call_cache_misses": float(self._shared_cache_misses), "structured_parse_retries": float(self._structured_parse_retries), "structured_parse_failures": float(self._structured_parse_failures)}

    def early_stop(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], hidden_state: PaprikaTask, latest_observation: PaprikaObservation) -> bool:
        del belief_state, history, hidden_state
        return latest_observation.goal_reached

    def early_stop_without_belief_state(self) -> bool:
        return True

    def build_eig_method(self, config: Any) -> CategoricalEIG:
        del config
        return CategoricalEIG()

    def build_full_two_step_eig_method(self, config: Any) -> FullTwoStepCategoricalEIG:
        del config
        return FullTwoStepCategoricalEIG()

    def branch_observation(self, action: PaprikaAction, outcome_index: int) -> PaprikaObservation:
        outcome = action.outcomes[outcome_index]
        return PaprikaObservation(
            reply=outcome,
            mapped_outcome=outcome,
            mapped_cleanly=True,
            goal_reached=False,
        )

    def generate_naive_action(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], model: Any, config: Any, *, method_name: str | None = None) -> PaprikaAction:
        del method_name
        scenario = history[0][0].scenario if history else self._scenario_by_support.get(belief_state.hypotheses)
        if not scenario and self._active_task is not None:
            scenario = self._active_task.scenario
        if not scenario:
            raise RuntimeError("Could not determine Paprika scenario")
        messages = candidate_messages(scenario, (), history, 1)
        return self._complete_parsed(
            model,
            messages,
            float(getattr(config, "generation_temperature_simple", 0.7)),
            namespace="questioner:naive_action",
            parser=lambda text: self._parse_candidates(text, scenario, history, 1)[0],
        )

    def generate_naive_actions_many(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[PaprikaAction, PaprikaObservation]]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[PaprikaAction]:
        del belief_states, method_name
        if len(histories) != len(self._active_batch_scenarios):
            raise ValueError("Paprika naive batch context does not match active histories")
        scenarios = [
            history[0][0].scenario if history else scenario
            for history, scenario in zip(histories, self._active_batch_scenarios)
        ]
        messages = [
            candidate_messages(scenario, (), history, 1)
            for scenario, history in zip(scenarios, histories)
        ]
        temperature = float(getattr(config, "generation_temperature_simple", 0.7))
        responses = self._cached_complete_many(
            model,
            messages,
            temperature,
            namespace="questioner:naive_action",
        )
        return self._parse_many_with_retries(
            model,
            messages,
            responses,
            temperature,
            namespace="questioner:naive_action",
            parsers=[
                (
                    lambda text, scenario=scenario, history=history: self._parse_candidates(
                        text, scenario, history, 1
                    )[0]
                )
                for scenario, history in zip(scenarios, histories)
            ],
        )

    def naive_requires_belief_state(self, method_name: str | None = None) -> bool:
        del method_name
        return False

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[PaprikaAction, PaprikaObservation]],
        hidden_state: PaprikaTask,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        del belief_state, hidden_state, model, config, method_name
        latest = history[-1][1]
        return {
            "resolved": float(any(observation.goal_reached for _action, observation in history)),
            "turns_used": float(len(history)),
            "answer_set_coverage": sum(
                observation.mapped_cleanly for _action, observation in history
            )
            / len(history),
            "latest_answer_mapped_cleanly": float(latest.mapped_cleanly),
            "shared_call_cache_hits": float(self._shared_cache_hits),
            "shared_call_cache_misses": float(self._shared_cache_misses),
            "structured_parse_retries": float(self._structured_parse_retries),
            "structured_parse_failures": float(self._structured_parse_failures),
        }

    def save_artifacts(self, run_result: Any, output_dir: Path, config: Any) -> dict[str, Path]:
        del config
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "paprika_smoke.json"
        records = []
        for trial in run_result.trials:
            records.append({"task_id": trial.hidden_state.task_id, "scenario": trial.hidden_state.scenario, "solution": trial.hidden_state.solution, "turns": [{"query": round_result.chosen.action.query, "kind": round_result.chosen.action.kind, "outcomes": list(round_result.chosen.action.outcomes), "reply": round_result.observation.reply, "mapped_outcome": round_result.observation.mapped_outcome, "mapped_cleanly": round_result.observation.mapped_cleanly, "goal_reached": round_result.observation.goal_reached} for round_result in trial.rounds], "final_metrics": trial.final_metrics})
        path.write_text(json.dumps(records, indent=2) + "\n")
        return {"paprika_smoke": path}
