"""Finite-target Bayesian experimental design for the MediQ benchmark."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from core import BeliefState, Environment
from methods.categorical_eig import CategoricalEIG, FullTwoStepCategoricalEIG

from .data import (
    MEDIQ_COMMIT,
    MEDIQ_ICRAFT_MD_SHA256,
    MEDIQ_IMEDQA_DEV_SHA256,
    MEDIQ_REPOSITORY,
    load_mediq_tasks_with_report,
)
from .parsing import (
    parse_candidate_set_validation,
    parse_candidate_validation,
    parse_distribution,
    parse_fact_selection,
    parse_json_object,
    parse_mapping,
    parse_relevance,
)
from .prompts import (
    ANSWERABLE_RECORD_OUTCOME,
    UNANSWERABLE_RECORD_OUTCOME,
    candidate_messages,
    candidate_set_validation_messages,
    candidate_validation_messages,
    data_estimation_outcome_messages,
    data_estimation_posterior_messages,
    factored_likelihood_messages,
    likelihood_messages,
    mapping_messages,
    patient_fact_messages,
    posterior_messages,
    prior_messages,
    record_availability_messages,
    relevance_messages,
    repair_messages,
)
from .types import MediQAction, MediQObservation, MediQTask


UNAVAILABLE_OUTCOME = "Information unavailable / not in record"
YES_OUTCOME = "Yes"
NO_OUTCOME = "No"
PATIENT_CANNOT_ANSWER = (
    "The patient cannot answer this question from the supplied record."
)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _project_joint_to_marginals(
    matrix: np.ndarray,
    row_marginals: np.ndarray,
    column_marginals: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    rows = np.asarray(row_marginals, dtype=float)
    columns = np.asarray(column_marginals, dtype=float)
    if values.shape != (len(rows), len(columns)):
        raise ValueError("joint matrix shape does not match requested marginals")
    if np.any(values <= 0.0) or np.any(rows <= 0.0) or np.any(columns <= 0.0):
        raise ValueError("joint projection requires strictly positive values")
    if not math.isclose(
        float(np.sum(rows)),
        float(np.sum(columns)),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("joint row and column marginals must have equal mass")

    for _iteration in range(max_iterations):
        values *= (rows / np.sum(values, axis=1))[:, None]
        values *= (columns / np.sum(values, axis=0))[None, :]
        residual = max(
            float(np.max(np.abs(np.sum(values, axis=1) - rows))),
            float(np.max(np.abs(np.sum(values, axis=0) - columns))),
        )
        if residual <= tolerance:
            return values
    raise RuntimeError("joint marginal projection did not converge")


def _is_unavailable(value: str) -> bool:
    normalized = value.casefold()
    return any(
        marker in normalized
        for marker in (
            "unavailable",
            "not in record",
            "cannot answer",
            "unknown",
            "not provided",
            "not recorded",
        )
    )


def _ensure_unavailable(values: Sequence[Any]) -> tuple[str, ...]:
    outcomes: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        if clean and not _is_unavailable(clean) and clean.casefold() not in seen:
            outcomes.append(clean)
            seen.add(clean.casefold())
    outcomes = outcomes[:4]
    outcomes.append(UNAVAILABLE_OUTCOME)
    return tuple(outcomes)


def _is_compound_query(query: str) -> bool:
    return re.search(r"\b(?:and|or)\b", query.casefold()) is not None


def _is_binary_query(query: str) -> bool:
    return (
        re.match(
            r"^(?:is|are|was|were|has|have|had|do|does|did|can|could|would|will)\b",
            query.strip(),
            flags=re.IGNORECASE,
        )
        is not None
    )


_QUERY_STOPWORDS = {
    "a",
    "an",
    "any",
    "are",
    "at",
    "can",
    "child",
    "could",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "experience",
    "experienced",
    "experiences",
    "for",
    "from",
    "had",
    "has",
    "have",
    "history",
    "in",
    "is",
    "of",
    "on",
    "patient",
    "recent",
    "report",
    "reported",
    "reports",
    "the",
    "there",
    "to",
    "was",
    "were",
    "will",
    "with",
    "would",
}


def _query_content_tokens(query: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", query.casefold())
    return {
        token[:-1] if len(token) > 4 and token.endswith("s") else token
        for token in tokens
        if token not in _QUERY_STOPWORDS
    }


def _queries_semantically_equivalent(left: str, right: str) -> bool:
    left_tokens = _query_content_tokens(left)
    right_tokens = _query_content_tokens(right)
    if not left_tokens or not right_tokens:
        return left.strip().casefold() == right.strip().casefold()
    if left_tokens == right_tokens:
        return True
    overlap = len(left_tokens & right_tokens)
    return overlap >= 2 and overlap / len(left_tokens | right_tokens) >= 0.8


def _query_contract_error(task: MediQTask, query: str) -> str | None:
    if _is_compound_query(query):
        return "contains 'and' or 'or'; ask one variable only"
    if not _is_binary_query(query):
        return "is not a yes/no predicate; use a binary question"
    normalized = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
    for _label, option in task.options:
        option_normalized = " ".join(re.findall(r"[a-z0-9]+", option.casefold()))
        if len(option_normalized) >= 4 and option_normalized in normalized:
            return f"contains or directly asks about answer option {option!r}"
    if re.search(
        r"\b(?:treated|treatment|given|administered|prescribed|prescription|"
        r"received|therapy|medication|drug|antibiotic|managed|management|"
        r"ordered|performed|obtained|diagnosed|diagnosis)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "asks about a diagnosis or management decision rather than patient evidence"
    if re.search(
        r"\b(?:hemodynamically stable|clinically stable|critically ill|toxic appearing)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "asks for a derived clinical judgment rather than an explicit observation"
    return None


class MediQEnvironment(Environment[MediQTask, str, MediQAction, MediQObservation]):
    """MediQ with beliefs over the released multiple-choice answer labels."""

    def __init__(self, config: Any, answerer: Any) -> None:
        self.config = config
        self.answerer = answerer
        self.questioner: Any | None = None
        self.tasks: list[MediQTask] = []
        self._raw_row_count = 0
        self._excluded_source_ids: tuple[str, ...] = ()
        self._active_task: MediQTask | None = None
        self._active_batch_tasks: tuple[MediQTask, ...] = ()
        self._task_by_belief_identity: dict[int, MediQTask] = {}
        self._likelihood_cache: dict[tuple[str, MediQAction], tuple[float, ...]] = {}
        self._record_availability_cache: dict[MediQAction, tuple[float, float]] = {}
        self._binary_likelihood_cache: dict[
            tuple[str, MediQAction], tuple[float, float]
        ] = {}
        self._data_estimation_marginal_cache: dict[
            MediQAction, tuple[float, ...]
        ] = {}
        self._data_estimation_posterior_cache: dict[
            tuple[str, MediQAction], tuple[float, ...]
        ] = {}
        self._data_estimation_projection_residuals: dict[MediQAction, float] = {}
        self._shared_cache_hits = 0
        self._shared_cache_misses = 0
        self._structured_parse_retries = 0
        self._structured_parse_failures = 0
        self._candidate_validation_checks = 0
        self._candidate_validation_retries = 0
        self._candidate_validation_failures = 0
        self._candidate_validation_cache: dict[MediQAction, tuple[bool, str]] = {}
        self._candidate_set_validation_checks = 0
        self._candidate_set_validation_rejections = 0
        self._candidate_set_validation_cache: dict[
            tuple[MediQAction, ...], tuple[bool, str]
        ] = {}
        self._patient_observations = 0
        self._patient_relevance_checks = 0
        self._patient_raw_irrelevant = 0
        self._patient_relevance_repairs = 0
        self._patient_relevance_failures = 0

    @property
    def name(self) -> str:
        return "mediq"

    def validate_config(self, config: Any) -> None:
        for name in (
            "mediq_num_trials",
            "mediq_num_rounds",
            "mediq_trial_batch_size",
            "mediq_num_candidates",
            "mediq_max_patient_facts",
        ):
            value = getattr(config, name, None)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if getattr(config, "mediq_dataset", "imedqa") not in {
            "imedqa",
            "icraft_md",
        }:
            raise ValueError("mediq_dataset must be one of: imedqa, icraft_md")
        if getattr(config, "mediq_likelihood_mode", "joint_option") not in {
            "joint_option",
            "factored_record",
            "data_estimation",
        }:
            raise ValueError(
                "mediq_likelihood_mode must be one of: joint_option, "
                "factored_record, data_estimation"
            )
        if not isinstance(getattr(config, "mediq_verify_official_hash", True), bool):
            raise ValueError("mediq_verify_official_hash must be a boolean")
        if not isinstance(getattr(config, "mediq_skip_unusable_tasks", True), bool):
            raise ValueError("mediq_skip_unusable_tasks must be a boolean")
        if not isinstance(getattr(config, "mediq_shared_call_cache_enabled", True), bool):
            raise ValueError("mediq_shared_call_cache_enabled must be a boolean")
        retries = getattr(config, "mediq_structured_max_retries", 2)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("mediq_structured_max_retries must be a non-negative integer")
        offset = getattr(config, "mediq_task_offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("mediq_task_offset must be a non-negative integer")
        floor = getattr(config, "mediq_probability_floor", 0.01)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise ValueError("mediq_probability_floor must be a number in [0, 0.2)")
        if not 0.0 <= float(floor) < 0.2:
            raise ValueError("mediq_probability_floor must be a number in [0, 0.2)")

    def configure_for_run(self, config: Any) -> "MediQEnvironment":
        path = getattr(config, "mediq_data_path", None)
        if not path:
            raise ValueError(
                "mediq_data_path must point to the pinned official MediQ JSONL file"
            )
        all_tasks, self._excluded_source_ids, self._raw_row_count = (
            load_mediq_tasks_with_report(
                path,
                dataset=getattr(config, "mediq_dataset", "imedqa"),
                verify_official_hash=bool(
                    getattr(config, "mediq_verify_official_hash", True)
                ),
                skip_unusable_tasks=bool(
                    getattr(config, "mediq_skip_unusable_tasks", True)
                ),
            )
        )
        offset = int(getattr(config, "mediq_task_offset", 0))
        count = int(getattr(config, "mediq_num_trials", 5))
        self.tasks = all_tasks[offset : offset + count]
        if len(self.tasks) != count:
            raise ValueError(
                f"Requested {count} MediQ tasks at offset {offset}, found {len(self.tasks)}"
            )
        return self

    def set_questioner(self, model: Any) -> None:
        self.questioner = model

    def _questioner(self) -> Any:
        if self.questioner is None:
            raise RuntimeError("MediQ environment has no attached questioner")
        return self.questioner

    def _evaluation_model(self) -> Any:
        questioner = self._questioner()
        return getattr(questioner, "_mediq_evaluation_model", questioner)

    def _cache_for(self, model: Any) -> dict[tuple[str, float, str], str]:
        cache = getattr(model, "_mediq_shared_call_cache", None)
        if cache is None:
            cache = {}
            setattr(model, "_mediq_shared_call_cache", cache)
        return cache

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
        enabled = bool(
            getattr(self.config, "mediq_shared_call_cache_enabled", True)
        )
        cache = self._cache_for(model)
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
                max_new_tokens=int(
                    getattr(self.config, "openrouter_max_output_tokens", 2048)
                ),
            )
            if len(responses) != len(missing_messages):
                raise ValueError("Batched MediQ completion returned the wrong response count")
            for position, response in zip(missing_positions, responses):
                clean_response = str(response)
                results[position] = clean_response
                if enabled:
                    cache[keys[position]] = clean_response
                self._shared_cache_misses += 1
        return [str(result) for result in results]

    def _complete_parsed_many(
        self,
        model: Any,
        messages: Sequence[list[dict[str, str]]],
        temperature: float,
        *,
        namespace: str,
        parsers: Sequence[Callable[[str], Any]],
    ) -> list[Any]:
        if len(messages) != len(parsers):
            raise ValueError("MediQ parser count must match completion count")
        current_messages = [list(item) for item in messages]
        current_responses = self._cached_complete_many(
            model,
            current_messages,
            temperature,
            namespace=namespace,
        )
        results: list[Any | None] = [None] * len(messages)
        active = list(range(len(messages)))
        maximum = int(getattr(self.config, "mediq_structured_max_retries", 2))
        for attempt in range(maximum + 1):
            failures: list[tuple[int, ValueError]] = []
            for index in active:
                try:
                    results[index] = parsers[index](current_responses[index])
                except ValueError as exc:
                    failures.append((index, exc))
            if not failures:
                return results
            if attempt >= maximum:
                self._structured_parse_failures += len(failures)
                raise failures[0][1]
            self._structured_parse_retries += len(failures)
            retry_indices = [index for index, _error in failures]
            retry_messages: list[list[dict[str, str]]] = []
            for index, error in failures:
                current_messages[index] = repair_messages(
                    current_messages[index],
                    current_responses[index],
                    str(error),
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

    def _smoothed(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        floor = float(getattr(self.config, "mediq_probability_floor", 0.01))
        values = np.maximum(np.asarray(probabilities, dtype=float), floor)
        values /= float(np.sum(values))
        return tuple(float(value) for value in values)

    def trial_count(self, config: Any) -> int:
        return int(getattr(config, "mediq_num_trials", 5))

    def round_count(self, config: Any) -> int:
        return int(getattr(config, "mediq_num_rounds", 5))

    def run_seed(self, config: Any) -> int | None:
        return getattr(config, "mediq_seed", None)

    def sample_hidden_state(self, rng: np.random.Generator) -> MediQTask:
        task = self.tasks[int(rng.integers(0, len(self.tasks)))]
        self._active_task = task
        return task

    def sample_hidden_state_for_trial(
        self, trial_index: int, rng: np.random.Generator
    ) -> MediQTask:
        del rng
        task = self.tasks[trial_index]
        self._active_task = task
        return task

    def sample_hidden_states_for_trials(
        self, trial_indices: Sequence[int], rng: np.random.Generator
    ) -> list[MediQTask]:
        del rng
        return [self.tasks[index] for index in trial_indices]

    def prepare_action_batch(self, hidden_states: Sequence[MediQTask]) -> None:
        self._active_batch_tasks = tuple(hidden_states)

    def log_prior(self, hypothesis: str) -> float:
        del hypothesis
        return 0.0

    def _prior_states_many(
        self, tasks: Sequence[MediQTask], model: Any
    ) -> list[BeliefState[str]]:
        messages = [prior_messages(task) for task in tasks]
        parsed = self._complete_parsed_many(
            self._evaluation_model(),
            messages,
            0.0,
            namespace="questioner:prior",
            parsers=[
                (
                    lambda text, labels=task.option_labels: parse_distribution(
                        text, labels
                    )
                )
                for task in tasks
            ],
        )
        states = [
            BeliefState(task.option_labels, self._smoothed(probabilities))
            for task, probabilities in zip(tasks, parsed)
        ]
        for task, state in zip(tasks, states):
            self._task_by_belief_identity[id(state)] = task
        return states

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[str]:
        del config
        self.set_questioner(model)
        if self._active_task is None:
            raise RuntimeError("A MediQ task must be selected before belief initialization")
        return self._prior_states_many([self._active_task], model)[0]

    def initial_belief_states(
        self, trial_indices: Sequence[int], model: Any, config: Any
    ) -> list[BeliefState[str]]:
        del config
        self.set_questioner(model)
        return self._prior_states_many([self.tasks[index] for index in trial_indices], model)

    def _likelihood_for(
        self, hypothesis: str, action: MediQAction
    ) -> tuple[float, ...]:
        key = (hypothesis, action)
        if key not in self._likelihood_cache:
            self.outcome_likelihoods([hypothesis], action)
        return self._likelihood_cache[key]

    def log_likelihood(
        self,
        hypothesis: str,
        action: MediQAction,
        observation: MediQObservation,
    ) -> float:
        if not observation.mapped_cleanly or observation.mapped_outcome is None:
            return 0.0
        probabilities = self._likelihood_for(hypothesis, action)
        index = action.outcomes.index(observation.mapped_outcome)
        return math.log(max(probabilities[index], 1e-300))

    def log_likelihood_many(
        self,
        hypotheses: Sequence[str],
        action: MediQAction,
        observation: MediQObservation,
    ) -> np.ndarray:
        if not observation.mapped_cleanly or observation.mapped_outcome is None:
            return np.zeros(len(hypotheses), dtype=float)
        matrix = self.outcome_likelihoods(hypotheses, action)
        index = action.outcomes.index(observation.mapped_outcome)
        return np.log(np.maximum(matrix[:, index], 1e-300))

    def outcome_likelihoods(
        self, hypotheses: Sequence[str], action: MediQAction
    ) -> np.ndarray:
        return self.outcome_likelihoods_many([(hypotheses, action)])[0]

    def outcome_likelihoods_many(
        self,
        requests: Sequence[tuple[Sequence[str], MediQAction]],
    ) -> list[np.ndarray]:
        likelihood_mode = getattr(
            self.config, "mediq_likelihood_mode", "joint_option"
        )
        if likelihood_mode == "factored_record":
            return self._factored_outcome_likelihoods_many(requests)
        if likelihood_mode == "data_estimation":
            return self._data_estimation_outcome_likelihoods_many(requests)
        missing_keys: list[tuple[str, MediQAction]] = []
        missing_messages: list[list[dict[str, str]]] = []
        seen: set[tuple[str, MediQAction]] = set()
        for hypotheses, action in requests:
            for hypothesis in hypotheses:
                key = (hypothesis, action)
                if key not in self._likelihood_cache and key not in seen:
                    seen.add(key)
                    missing_keys.append(key)
                    missing_messages.append(likelihood_messages(hypothesis, action))
        if missing_messages:
            parsed = self._complete_parsed_many(
                self._evaluation_model(),
                missing_messages,
                0.0,
                namespace="questioner:likelihood",
                parsers=[
                    (
                        lambda text, outcomes=action.outcomes: parse_distribution(
                            text, outcomes
                        )
                    )
                    for _hypothesis, action in missing_keys
                ],
            )
            for key, probabilities in zip(missing_keys, parsed):
                self._likelihood_cache[key] = self._smoothed(probabilities)
        return [
            np.asarray(
                [self._likelihood_cache[(hypothesis, action)] for hypothesis in hypotheses],
                dtype=float,
            )
            for hypotheses, action in requests
        ]

    def _factored_outcome_likelihoods_many(
        self,
        requests: Sequence[tuple[Sequence[str], MediQAction]],
    ) -> list[np.ndarray]:
        actions: list[MediQAction] = []
        seen_actions: set[MediQAction] = set()
        missing_keys: list[tuple[str, MediQAction]] = []
        seen_keys: set[tuple[str, MediQAction]] = set()
        for hypotheses, action in requests:
            if action.outcomes != (YES_OUTCOME, NO_OUTCOME, UNAVAILABLE_OUTCOME):
                raise ValueError(
                    "factored_record likelihoods require canonical Yes, No, and "
                    "unavailable outcomes"
                )
            if action not in self._record_availability_cache and action not in seen_actions:
                seen_actions.add(action)
                actions.append(action)
            for hypothesis in hypotheses:
                key = (hypothesis, action)
                if key not in self._binary_likelihood_cache and key not in seen_keys:
                    seen_keys.add(key)
                    missing_keys.append(key)

        if actions:
            availability_labels = (
                ANSWERABLE_RECORD_OUTCOME,
                UNANSWERABLE_RECORD_OUTCOME,
            )
            parsed = self._complete_parsed_many(
                self._evaluation_model(),
                [record_availability_messages(action) for action in actions],
                0.0,
                namespace="questioner:record_availability",
                parsers=[
                    (
                        lambda text, labels=availability_labels: parse_distribution(
                            text, labels
                        )
                    )
                    for _action in actions
                ],
            )
            for action, probabilities in zip(actions, parsed):
                self._record_availability_cache[action] = self._smoothed(probabilities)

        if missing_keys:
            binary_labels = (YES_OUTCOME, NO_OUTCOME)
            parsed = self._complete_parsed_many(
                self._evaluation_model(),
                [
                    factored_likelihood_messages(hypothesis, action)
                    for hypothesis, action in missing_keys
                ],
                0.0,
                namespace="questioner:factored_likelihood",
                parsers=[
                    (
                        lambda text, labels=binary_labels: parse_distribution(
                            text, labels
                        )
                    )
                    for _key in missing_keys
                ],
            )
            for key, probabilities in zip(missing_keys, parsed):
                self._binary_likelihood_cache[key] = self._smoothed(probabilities)

        for hypotheses, action in requests:
            answerable, unavailable = self._record_availability_cache[action]
            for hypothesis in hypotheses:
                key = (hypothesis, action)
                if key in self._likelihood_cache:
                    continue
                yes, no = self._binary_likelihood_cache[key]
                self._likelihood_cache[key] = (
                    answerable * yes,
                    answerable * no,
                    unavailable,
                )

        return [
            np.asarray(
                [self._likelihood_cache[(hypothesis, action)] for hypothesis in hypotheses],
                dtype=float,
            )
            for hypotheses, action in requests
        ]

    def _data_estimation_outcome_likelihoods_many(
        self,
        requests: Sequence[tuple[Sequence[str], MediQAction]],
    ) -> list[np.ndarray]:
        actions: list[MediQAction] = []
        seen_actions: set[MediQAction] = set()
        for hypotheses, action in requests:
            if tuple(hypotheses) != action.task.option_labels:
                raise ValueError(
                    "data_estimation likelihoods require the complete ordered A-D support"
                )
            if action.outcomes != (YES_OUTCOME, NO_OUTCOME, UNAVAILABLE_OUTCOME):
                raise ValueError(
                    "data_estimation likelihoods require canonical Yes, No, and "
                    "unavailable outcomes"
                )
            if len(action.prior_probabilities) != len(hypotheses):
                raise ValueError(
                    "data_estimation likelihoods require the current prior on the action"
                )
            if action not in self._data_estimation_marginal_cache and action not in seen_actions:
                seen_actions.add(action)
                actions.append(action)

        if actions:
            parsed = self._complete_parsed_many(
                self._evaluation_model(),
                [data_estimation_outcome_messages(action) for action in actions],
                0.0,
                namespace="questioner:data_estimation_marginal",
                parsers=[
                    (
                        lambda text, outcomes=action.outcomes: parse_distribution(
                            text, outcomes
                        )
                    )
                    for action in actions
                ],
            )
            for action, probabilities in zip(actions, parsed):
                marginal = self._smoothed(probabilities)
                self._data_estimation_marginal_cache[action] = marginal
                self._record_availability_cache[action] = (
                    1.0 - marginal[2],
                    marginal[2],
                )

        missing_posterior_keys: list[tuple[str, MediQAction]] = []
        seen_posterior_keys: set[tuple[str, MediQAction]] = set()
        for _hypotheses, action in requests:
            for outcome in (YES_OUTCOME, NO_OUTCOME):
                key = (outcome, action)
                if (
                    key not in self._data_estimation_posterior_cache
                    and key not in seen_posterior_keys
                ):
                    seen_posterior_keys.add(key)
                    missing_posterior_keys.append(key)
        if missing_posterior_keys:
            parsed = self._complete_parsed_many(
                self._evaluation_model(),
                [
                    data_estimation_posterior_messages(action, outcome)
                    for outcome, action in missing_posterior_keys
                ],
                0.0,
                namespace="questioner:data_estimation_posterior",
                parsers=[
                    (
                        lambda text, labels=action.task.option_labels: parse_distribution(
                            text, labels
                        )
                    )
                    for _outcome, action in missing_posterior_keys
                ],
            )
            for key, probabilities in zip(missing_posterior_keys, parsed):
                self._data_estimation_posterior_cache[key] = self._smoothed(
                    probabilities
                )

        for hypotheses, action in requests:
            if all((hypothesis, action) in self._likelihood_cache for hypothesis in hypotheses):
                continue
            prior = np.asarray(action.prior_probabilities, dtype=float)
            prior /= float(np.sum(prior))
            marginal = np.asarray(
                self._data_estimation_marginal_cache[action], dtype=float
            )
            available_mass = float(marginal[0] + marginal[1])
            raw_available_joint = np.column_stack(
                [
                    marginal[outcome_index]
                    * np.asarray(
                        self._data_estimation_posterior_cache[(outcome, action)],
                        dtype=float,
                    )
                    for outcome_index, outcome in enumerate(
                        (YES_OUTCOME, NO_OUTCOME)
                    )
                ]
            )
            available_joint = _project_joint_to_marginals(
                raw_available_joint,
                available_mass * prior,
                marginal[:2],
            )
            joint = np.column_stack(
                [available_joint, marginal[2] * prior]
            )
            residual = max(
                float(np.max(np.abs(np.sum(joint, axis=1) - prior))),
                float(np.max(np.abs(np.sum(joint, axis=0) - marginal))),
            )
            self._data_estimation_projection_residuals[action] = residual
            likelihoods = joint / prior[:, None]
            for index, hypothesis in enumerate(hypotheses):
                self._likelihood_cache[(hypothesis, action)] = tuple(
                    float(value) for value in likelihoods[index]
                )

        return [
            np.asarray(
                [self._likelihood_cache[(hypothesis, action)] for hypothesis in hypotheses],
                dtype=float,
            )
            for hypotheses, action in requests
        ]

    def update_belief_state(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        model: Any,
        config: Any,
    ) -> BeliefState[str]:
        del model, config
        if not history:
            return belief_state
        action, observation = history[-1]
        if not observation.mapped_cleanly or observation.mapped_outcome is None:
            return belief_state
        log_scores = np.log(
            np.maximum(np.asarray(belief_state.probabilities, dtype=float), 1e-300)
        )
        log_scores += self.log_likelihood_many(
            belief_state.hypotheses, action, observation
        )
        updated = BeliefState.from_log_scores(belief_state.hypotheses, log_scores)
        result = BeliefState(updated.hypotheses, self._smoothed(updated.probabilities))
        self._task_by_belief_identity[id(result)] = action.task
        return result

    def update_belief_states(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
        model: Any,
        config: Any,
    ) -> list[BeliefState[str]]:
        return [
            self.update_belief_state(belief, history, model, config)
            for belief, history in zip(belief_states, histories)
        ]

    def _task_for(
        self,
        history: Sequence[tuple[MediQAction, MediQObservation]],
        fallback: MediQTask | None,
    ) -> MediQTask:
        if history:
            return history[0][0].task
        if fallback is None:
            raise RuntimeError("Could not associate MediQ state with a task")
        return fallback

    def _parse_candidates(
        self,
        text: str,
        task: MediQTask,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        expected: int,
    ) -> list[MediQAction]:
        raw = parse_json_object(text).get("candidates")
        if not isinstance(raw, list):
            raise ValueError("candidate response requires a candidates list")
        prior_queries = [action.query for action, _observation in history]
        transcript = tuple(
            (action.query, observation.reply) for action, observation in history
        )
        actions: list[MediQAction] = []
        seen_queries: list[str] = []
        rejected: list[str] = []
        if len(raw) != expected:
            rejected.append(f"response returned {len(raw)} candidates instead of {expected}")
        for item_index, item in enumerate(raw):
            if not isinstance(item, dict):
                rejected.append(f"candidate {item_index} is not an object")
                continue
            query = item.get("query")
            outcomes = item.get("outcomes")
            if not isinstance(query, str) or not isinstance(outcomes, list):
                rejected.append(
                    f"candidate {item_index} requires string query and list outcomes"
                )
                continue
            clean_query = query.strip()
            if any(
                _queries_semantically_equivalent(clean_query, prior)
                for prior in prior_queries
            ):
                rejected.append(
                    f"{clean_query!r} semantically repeats an earlier query"
                )
                continue
            if any(
                _queries_semantically_equivalent(clean_query, seen)
                for seen in seen_queries
            ):
                rejected.append(
                    f"{clean_query!r} semantically duplicates another candidate"
                )
                continue
            contract_error = _query_contract_error(task, clean_query)
            if contract_error is not None:
                rejected.append(f"{clean_query!r} {contract_error}")
                continue
            normalized_outcomes = _ensure_unavailable(outcomes)
            if (
                len(normalized_outcomes) != 3
                or {value.casefold() for value in normalized_outcomes[:-1]}
                != {YES_OUTCOME.casefold(), NO_OUTCOME.casefold()}
                or normalized_outcomes[-1] != UNAVAILABLE_OUTCOME
            ):
                rejected.append(
                    f"{clean_query!r} must use exactly Yes, No, and "
                    f"{UNAVAILABLE_OUTCOME!r} outcomes"
                )
                continue
            try:
                action = MediQAction(
                    query=clean_query,
                    outcomes=(YES_OUTCOME, NO_OUTCOME, UNAVAILABLE_OUTCOME),
                    task=task,
                    transcript=transcript,
                    prior_probabilities=belief_state.probabilities,
                )
            except ValueError as exc:
                rejected.append(f"{clean_query!r} has invalid outcomes: {exc}")
                continue
            actions.append(action)
            seen_queries.append(clean_query)
        if len(actions) != expected or len(raw) != expected:
            details = "; ".join(rejected) or "candidate count did not match"
            raise ValueError(
                f"Expected {expected} valid MediQ candidates, parsed {len(actions)}. "
                f"Rejected: {details}"
            )
        return actions

    def _validated_candidate_actions_many(
        self,
        tasks: Sequence[MediQTask],
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
        model: Any,
        *,
        count: int,
        temperature: float,
        namespace: str,
        naive: bool = False,
    ) -> list[list[MediQAction]]:
        if not (len(tasks) == len(belief_states) == len(histories)):
            raise ValueError("MediQ candidate inputs must have the same length")
        accepted: list[list[MediQAction]] = [[] for _task in tasks]
        prior_failures: list[list[tuple[MediQAction, str]]] = [
            [] for _task in tasks
        ]
        pending = list(range(len(tasks)))
        maximum = int(getattr(self.config, "mediq_structured_max_retries", 2))

        for semantic_attempt in range(maximum + 1):
            needed = {index: count - len(accepted[index]) for index in pending}
            requested = {
                index: (
                    max(needed[index], 4)
                    if naive and prior_failures[index] and needed[index] == 1
                    else max(needed[index], 2)
                    if accepted[index]
                    else needed[index]
                )
                for index in pending
            }
            generation_messages: list[list[dict[str, str]]] = []
            for index in pending:
                messages = candidate_messages(
                    tasks[index],
                    belief_states[index],
                    histories[index],
                    requested[index],
                    naive=naive,
                )
                if accepted[index] or prior_failures[index]:
                    accepted_text = "\n".join(
                        f"- {action.query}" for action in accepted[index]
                    ) or "None"
                    failure_text = "\n".join(
                        f"- {action.query}: {reason}"
                        for action, reason in prior_failures[index]
                    ) or "None"
                    messages = list(messages) + [
                        {
                            "role": "user",
                            "content": (
                                f"Already accepted queries:\n{accepted_text}\n\n"
                                f"Rejected queries:\n{failure_text}\n\n"
                                f"Generate exactly {requested[index]} replacement candidate(s). "
                                "Each replacement must test a different observable symptom, "
                                "history item, examination finding, or numeric test result from "
                                "every other replacement and rejected concept. Do not use a "
                                "diagnosis name as patient history. Do not repeat accepted queries. "
                                "Correct every rejection and "
                                "return only the requested strict JSON."
                            ),
                        }
                    ]
                generation_messages.append(messages)
            generated = self._complete_parsed_many(
                model,
                generation_messages,
                temperature,
                namespace=f"{namespace}:semantic_attempt:{semantic_attempt}",
                parsers=[
                    (
                        lambda text, index=index: self._parse_candidates(
                            text,
                            tasks[index],
                            belief_states[index],
                            histories[index],
                            requested[index],
                        )
                    )
                    for index in pending
                ],
            )
            validation_entries = [
                (index, action)
                for index, actions in zip(pending, generated)
                for action in actions
            ]
            judgments = self._complete_parsed_many(
                self._evaluation_model(),
                [
                    candidate_validation_messages(
                        tasks[index], action, histories[index]
                    )
                    for index, action in validation_entries
                ],
                0.0,
                namespace=(
                    f"questioner:candidate_validation:{namespace}:"
                    f"semantic_attempt:{semantic_attempt}"
                ),
                parsers=[parse_candidate_validation for _entry in validation_entries],
            )
            self._candidate_validation_checks += len(validation_entries)

            failures: dict[int, list[tuple[MediQAction, str]]] = {}
            for (index, action), (valid, reason) in zip(
                validation_entries, judgments
            ):
                self._candidate_validation_cache[action] = (valid, reason)
                if any(
                    _queries_semantically_equivalent(existing.query, action.query)
                    for existing in accepted[index]
                ):
                    failures.setdefault(index, []).append(
                        (action, "duplicates an already accepted query")
                    )
                elif valid and len(accepted[index]) < count:
                    accepted[index].append(action)
                else:
                    failures.setdefault(index, []).append((action, reason))

            set_indices = [index for index in pending if len(accepted[index]) >= 2]
            set_judgments = self._complete_parsed_many(
                self._evaluation_model(),
                [
                    candidate_set_validation_messages(
                        tasks[index], accepted[index], histories[index]
                    )
                    for index in set_indices
                ],
                0.0,
                namespace=(
                    f"questioner:candidate_set_validation:{namespace}:"
                    f"semantic_attempt:{semantic_attempt}"
                ),
                parsers=[
                    (
                        lambda text, index=index: parse_candidate_set_validation(
                            text, len(accepted[index])
                        )
                    )
                    for index in set_indices
                ],
            )
            self._candidate_set_validation_checks += len(set_indices)
            for index, (duplicate_groups, reason) in zip(
                set_indices, set_judgments
            ):
                snapshot = tuple(accepted[index])
                if not duplicate_groups:
                    self._candidate_set_validation_cache[snapshot] = (True, reason)
                    continue
                self._candidate_set_validation_cache[snapshot] = (False, reason)
                removed_indices = {
                    duplicate_index
                    for group in duplicate_groups
                    for duplicate_index in group[1:]
                }
                self._candidate_set_validation_rejections += len(removed_indices)
                for duplicate_index in sorted(removed_indices):
                    action = snapshot[duplicate_index]
                    failures.setdefault(index, []).append(
                        (action, f"set-level semantic duplicate: {reason}")
                    )
                accepted[index] = [
                    action
                    for action_index, action in enumerate(snapshot)
                    if action_index not in removed_indices
                ]

            retry_indices: list[int] = []
            for index in pending:
                invalid = failures.get(index, [])
                prior_failures[index] = invalid
                if len(accepted[index]) == count:
                    continue
                if semantic_attempt >= maximum:
                    self._candidate_validation_failures += 1
                    details = "; ".join(
                        f"{action.query!r}: {reason}" for action, reason in invalid
                    ) or "no valid replacement candidates were returned"
                    raise ValueError(
                        "MediQ candidate semantic validation failed after bounded "
                        f"repairs with {len(accepted[index])}/{count} accepted: {details}"
                    )
                self._candidate_validation_retries += 1
                retry_indices.append(index)
            if not retry_indices:
                for actions in accepted:
                    self._candidate_set_validation_cache.setdefault(
                        tuple(actions),
                        (True, "single candidate requires no set-level deduplication"),
                    )
                return accepted
            pending = retry_indices
        raise AssertionError("unreachable")

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        model: Any,
        config: Any,
    ) -> list[MediQAction]:
        task = self._task_for(history, self._active_task)
        count = int(getattr(config, "mediq_num_candidates", 5))
        return self._validated_candidate_actions_many(
            [task],
            [belief_state],
            [history],
            model,
            count=count,
            temperature=float(
                getattr(config, "generation_temperature_diverse", 0.7)
            ),
            namespace="questioner:candidates",
        )[0]

    def generate_candidate_actions_many(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
        model: Any,
        config: Any,
    ) -> list[list[MediQAction]]:
        if len(belief_states) != len(histories):
            raise ValueError("MediQ beliefs and histories must have the same length")
        fallbacks: Sequence[MediQTask | None] = [
            self._task_by_belief_identity.get(id(belief))
            for belief in belief_states
        ]
        tasks = [
            self._task_for(history, fallback)
            for history, fallback in zip(histories, fallbacks)
        ]
        count = int(getattr(config, "mediq_num_candidates", 5))
        return self._validated_candidate_actions_many(
            tasks,
            belief_states,
            histories,
            model,
            count=count,
            temperature=float(
                getattr(config, "generation_temperature_diverse", 0.7)
            ),
            namespace="questioner:candidates",
        )

    def _unavailable_outcome(self, action: MediQAction) -> str:
        for outcome in action.outcomes:
            if _is_unavailable(outcome):
                return outcome
        raise ValueError("MediQ action has no unavailable outcome")

    def _parse_patient_mapping(
        self, text: str, action: MediQAction
    ) -> tuple[str | None, bool]:
        canonical, clean = parse_mapping(text, action.outcomes)
        if clean and canonical is not None and _is_unavailable(canonical):
            raise ValueError(
                "an explicitly relevant patient fact cannot map to unavailable"
            )
        return canonical, clean

    def _patient_observations_many(
        self,
        actions: Sequence[MediQAction],
        tasks: Sequence[MediQTask],
    ) -> list[MediQObservation]:
        if len(actions) != len(tasks):
            raise ValueError("MediQ actions and tasks must have the same length")
        maximum_facts = int(getattr(self.config, "mediq_max_patient_facts", 2))
        maximum_retries = int(
            getattr(self.config, "mediq_structured_max_retries", 2)
        )
        selector_messages = [
            patient_fact_messages(task, action.query, maximum_facts)
            for action, task in zip(actions, tasks)
        ]
        pending = list(range(len(actions)))
        observations: list[MediQObservation | None] = [None] * len(actions)
        self._patient_observations += len(actions)

        for relevance_attempt in range(maximum_retries + 1):
            current_messages = [selector_messages[index] for index in pending]
            selections = self._complete_parsed_many(
                self.answerer,
                current_messages,
                float(getattr(self.config, "answer_temperature", 0.0)),
                namespace=f"answerer:fact_select:relevance_attempt:{relevance_attempt}",
                parsers=[
                    (
                        lambda text, task=tasks[index]: parse_fact_selection(
                            text,
                            num_facts=len(task.facts),
                            max_facts=maximum_facts,
                        )
                    )
                    for index in pending
                ],
            )

            replies: dict[int, str] = {}
            selected_by_index: dict[int, tuple[int, ...]] = {}
            cannot_by_index: dict[int, bool] = {}
            relevance_indices: list[int] = []
            relevance_prompts: list[list[dict[str, str]]] = []
            for index, (selected, cannot_answer) in zip(pending, selections):
                selected_by_index[index] = selected
                cannot_by_index[index] = cannot_answer
                if cannot_answer:
                    replies[index] = PATIENT_CANNOT_ANSWER
                    observations[index] = MediQObservation(
                        reply=PATIENT_CANNOT_ANSWER,
                        mapped_outcome=self._unavailable_outcome(actions[index]),
                        mapped_cleanly=True,
                        selected_fact_indices=(),
                        grounded=True,
                        relevant=True,
                        cannot_answer=True,
                    )
                    continue
                reply = "\n".join(tasks[index].facts[item] for item in selected)
                replies[index] = reply
                relevance_indices.append(index)
                relevance_prompts.append(
                    relevance_messages(actions[index].query, reply)
                )

            relevance_judgments = self._complete_parsed_many(
                self._evaluation_model(),
                relevance_prompts,
                0.0,
                namespace=(
                    "questioner:patient_relevance:"
                    f"relevance_attempt:{relevance_attempt}"
                ),
                parsers=[parse_relevance for _index in relevance_indices],
            )
            self._patient_relevance_checks += len(relevance_indices)

            retry_indices: list[int] = []
            relevance_reasons: dict[int, str] = {}
            mapping_indices: list[int] = []
            for index, (relevant, reason) in zip(
                relevance_indices, relevance_judgments
            ):
                if not relevant:
                    if relevance_attempt == 0:
                        self._patient_raw_irrelevant += 1
                    relevance_reasons[index] = reason
                    retry_indices.append(index)
                    continue
                mapping_indices.append(index)

            mappings = self._complete_parsed_many(
                self._evaluation_model(),
                [
                    mapping_messages(replies[index], actions[index])
                    for index in mapping_indices
                ],
                0.0,
                namespace=(
                    "questioner:patient_mapping:"
                    f"relevance_attempt:{relevance_attempt}"
                ),
                parsers=[
                    (
                        lambda text, action=actions[index]: self._parse_patient_mapping(
                            text, action
                        )
                    )
                    for index in mapping_indices
                ],
            )
            for index, (canonical, clean) in zip(mapping_indices, mappings):
                observations[index] = MediQObservation(
                    reply=replies[index],
                    mapped_outcome=canonical,
                    mapped_cleanly=clean and canonical is not None,
                    selected_fact_indices=selected_by_index[index],
                    grounded=True,
                    relevant=True,
                    cannot_answer=cannot_by_index[index],
                )

            if not retry_indices:
                if any(observation is None for observation in observations):
                    raise RuntimeError("MediQ patient batch left an observation unset")
                return [observation for observation in observations if observation is not None]
            if relevance_attempt >= maximum_retries:
                self._patient_relevance_failures += len(retry_indices)
                raise ValueError(
                    "MediQ patient selected irrelevant facts after bounded repairs"
                )
            self._patient_relevance_repairs += len(retry_indices)
            for index in retry_indices:
                selector_messages[index] = list(selector_messages[index]) + [
                    {
                        "role": "user",
                        "content": (
                            "The selected facts did not directly answer the doctor question. "
                            f"The entailment audit said: {relevance_reasons[index]} "
                            "Select a different directly relevant fact, or set cannot_answer=true. "
                            "Return strict JSON only."
                        ),
                    }
                ]
            pending = retry_indices
        raise AssertionError("unreachable")

    def observe(
        self,
        action: MediQAction,
        hidden_state: MediQTask,
        rng: np.random.Generator,
    ) -> MediQObservation:
        del rng
        return self._patient_observations_many([action], [hidden_state])[0]

    def observe_many(
        self,
        actions: Sequence[MediQAction],
        hidden_states: Sequence[MediQTask],
        rng: np.random.Generator,
    ) -> list[MediQObservation]:
        del rng
        return self._patient_observations_many(actions, hidden_states)

    def branch_observation(
        self, action: MediQAction, outcome_index: int
    ) -> MediQObservation:
        outcome = action.outcomes[outcome_index]
        return MediQObservation(
            reply=outcome,
            mapped_outcome=outcome,
            mapped_cleanly=True,
            selected_fact_indices=(),
            grounded=True,
            relevant=True,
            cannot_answer=_is_unavailable(outcome),
        )

    def build_eig_method(self, config: Any) -> CategoricalEIG:
        del config
        return CategoricalEIG()

    def build_full_two_step_eig_method(
        self, config: Any
    ) -> FullTwoStepCategoricalEIG:
        del config
        return FullTwoStepCategoricalEIG()

    def _belief_metrics(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        task: MediQTask,
    ) -> dict[str, float]:
        probabilities = np.asarray(belief_state.probabilities, dtype=float)
        labels = belief_state.hypotheses
        true_index = labels.index(task.answer_idx)
        prediction_index = int(np.argmax(probabilities))
        target = np.zeros(len(labels), dtype=float)
        target[true_index] = 1.0
        coverage = (
            sum(observation.mapped_cleanly for _action, observation in history)
            / len(history)
            if history
            else 0.0
        )
        grounded = (
            sum(observation.grounded for _action, observation in history) / len(history)
            if history
            else 0.0
        )
        relevant = (
            sum(observation.relevant for _action, observation in history) / len(history)
            if history
            else 0.0
        )
        metrics = {
            "accuracy": float(labels[prediction_index] == task.answer_idx),
            "correct_option_mass": float(probabilities[true_index]),
            "belief_top_probability": float(probabilities[prediction_index]),
            "belief_entropy": _entropy(probabilities),
            "belief_brier_score": float(np.sum((probabilities - target) ** 2)),
            "belief_log_loss": -math.log(max(float(probabilities[true_index]), 1e-300)),
            "questions_asked": float(len(history)),
            "answer_set_coverage": coverage,
            "latest_answer_mapped_cleanly": float(
                history[-1][1].mapped_cleanly if history else False
            ),
            "patient_grounding_rate": grounded,
            "patient_relevance_rate": relevant,
            "shared_call_cache_hits": float(self._shared_cache_hits),
            "shared_call_cache_misses": float(self._shared_cache_misses),
            "structured_parse_retries": float(self._structured_parse_retries),
            "structured_parse_failures": float(self._structured_parse_failures),
            **self._candidate_metrics(),
            **self._patient_metrics(),
        }
        if history and history[-1][0].prior_probabilities:
            action, observation = history[-1]
            prior = np.asarray(action.prior_probabilities, dtype=float)
            metrics["realized_entropy_drop"] = _entropy(prior) - _entropy(
                probabilities
            )
            metrics["realized_correct_option_mass_gain"] = float(
                probabilities[true_index] - prior[true_index]
            )
            metrics["realized_truth_log_probability_gain"] = math.log(
                max(float(probabilities[true_index]), 1e-300)
            ) - math.log(max(float(prior[true_index]), 1e-300))
            if observation.mapped_cleanly and observation.mapped_outcome is not None:
                likelihoods = self.outcome_likelihoods(labels, action)
                outcome_index = action.outcomes.index(observation.mapped_outcome)
                predictive = prior @ likelihoods
                observed_probability = float(predictive[outcome_index])
                unavailable_index = action.outcomes.index(UNAVAILABLE_OUTCOME)
                unavailable_likelihoods = likelihoods[:, unavailable_index]
                one_hot = np.zeros(len(action.outcomes), dtype=float)
                one_hot[outcome_index] = 1.0
                metrics["observed_outcome_predictive_probability"] = observed_probability
                metrics["observed_outcome_true_label_probability"] = float(
                    likelihoods[true_index, outcome_index]
                )
                metrics["observed_outcome_log_loss"] = -math.log(
                    max(observed_probability, 1e-300)
                )
                metrics["observed_outcome_brier_score"] = float(
                    np.sum((predictive - one_hot) ** 2)
                )
                metrics["selected_unavailable_likelihood_span"] = float(
                    np.max(unavailable_likelihoods)
                    - np.min(unavailable_likelihoods)
                )
                metrics["selected_record_answerability_probability"] = float(
                    1.0 - prior @ unavailable_likelihoods
                )
                if action in self._data_estimation_projection_residuals:
                    metrics["selected_joint_projection_residual"] = float(
                        self._data_estimation_projection_residuals[action]
                    )
        return metrics

    def round_metrics(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        hidden_state: MediQTask,
    ) -> dict[str, float]:
        return self._belief_metrics(belief_state, history, hidden_state)

    def _candidate_metrics(self) -> dict[str, float]:
        return {
            "candidate_validation_checks": float(self._candidate_validation_checks),
            "candidate_validation_retries": float(
                self._candidate_validation_retries
            ),
            "candidate_validation_failures": float(
                self._candidate_validation_failures
            ),
            "candidate_set_validation_checks": float(
                self._candidate_set_validation_checks
            ),
            "candidate_set_validation_rejections": float(
                self._candidate_set_validation_rejections
            ),
        }

    def _candidate_validation_for(
        self, action: MediQAction
    ) -> tuple[bool, str] | None:
        direct = self._candidate_validation_cache.get(action)
        if direct is not None:
            return direct
        matches = [
            judgment
            for candidate, judgment in self._candidate_validation_cache.items()
            if candidate.query == action.query
            and candidate.task == action.task
            and candidate.transcript == action.transcript
        ]
        return matches[-1] if matches else None

    def _candidate_set_validation_for(
        self, actions: Sequence[MediQAction]
    ) -> tuple[bool, str] | None:
        direct = self._candidate_set_validation_cache.get(tuple(actions))
        if direct is not None:
            return direct
        queries = tuple(action.query for action in actions)
        matches = [
            judgment
            for candidates, judgment in self._candidate_set_validation_cache.items()
            if tuple(candidate.query for candidate in candidates) == queries
        ]
        return matches[-1] if matches else None

    def _patient_metrics(self) -> dict[str, float]:
        observations = self._patient_observations
        return {
            "patient_observations": float(observations),
            "patient_relevance_checks": float(self._patient_relevance_checks),
            "patient_raw_irrelevant_selections": float(self._patient_raw_irrelevant),
            "patient_relevance_repairs": float(self._patient_relevance_repairs),
            "patient_relevance_failures": float(self._patient_relevance_failures),
            "patient_raw_irrelevant_rate": (
                self._patient_raw_irrelevant / observations if observations else 0.0
            ),
            "patient_final_irrelevance_rate": (
                self._patient_relevance_failures / observations if observations else 0.0
            ),
        }

    def generate_naive_action(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> MediQAction:
        del belief_state, method_name
        task = self._task_for(history, self._active_task)
        uniform = BeliefState.uniform(task.option_labels)
        action = self._validated_candidate_actions_many(
            [task],
            [uniform],
            [history],
            model,
            count=1,
            temperature=float(
                getattr(config, "generation_temperature_simple", 0.0)
            ),
            namespace="questioner:naive_question",
            naive=True,
        )[0][0]
        return MediQAction(
            query=action.query,
            outcomes=action.outcomes,
            task=action.task,
            transcript=action.transcript,
            prior_probabilities=(),
        )

    def generate_naive_actions_many(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[MediQAction]:
        del belief_states, method_name
        if len(self._active_batch_tasks) == len(histories):
            fallbacks: Sequence[MediQTask | None] = self._active_batch_tasks
        else:
            fallbacks = [None] * len(histories)
        tasks = [
            self._task_for(history, fallback)
            for history, fallback in zip(histories, fallbacks)
        ]
        uniforms = [BeliefState.uniform(task.option_labels) for task in tasks]
        parsed_sets = self._validated_candidate_actions_many(
            tasks,
            uniforms,
            histories,
            model,
            count=1,
            temperature=float(
                getattr(config, "generation_temperature_simple", 0.0)
            ),
            namespace="questioner:naive_question",
            naive=True,
        )
        return [
            MediQAction(
                query=action.query,
                outcomes=action.outcomes,
                task=action.task,
                transcript=action.transcript,
                prior_probabilities=(),
            )
            for action in (actions[0] for actions in parsed_sets)
        ]

    def naive_requires_belief_state(self, method_name: str | None = None) -> bool:
        del method_name
        return False

    def _decoded_states_many(
        self,
        tasks: Sequence[MediQTask],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
    ) -> list[BeliefState[str]]:
        messages = [
            posterior_messages(task, history)
            for task, history in zip(tasks, histories)
        ]
        parsed = self._complete_parsed_many(
            self._evaluation_model(),
            messages,
            0.0,
            namespace="questioner:naive_decode",
            parsers=[
                (
                    lambda text, labels=task.option_labels: parse_distribution(
                        text, labels
                    )
                )
                for task in tasks
            ],
        )
        return [
            BeliefState(task.option_labels, self._smoothed(probabilities))
            for task, probabilities in zip(tasks, parsed)
        ]

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[MediQAction, MediQObservation]],
        hidden_state: MediQTask,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        del belief_state, model, config, method_name
        decoded = self._decoded_states_many([hidden_state], [history])[0]
        return self._belief_metrics(decoded, history, hidden_state)

    def naive_metrics_after_observations(
        self,
        belief_states: Sequence[BeliefState[str]],
        histories: Sequence[Sequence[tuple[MediQAction, MediQObservation]]],
        hidden_states: Sequence[MediQTask],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[dict[str, float]]:
        del belief_states, model, config, method_name
        decoded = self._decoded_states_many(hidden_states, histories)
        return [
            self._belief_metrics(state, history, task)
            for state, history, task in zip(decoded, histories, hidden_states)
        ]

    def save_artifacts(
        self, run_result: Any, output_dir: Path, config: Any
    ) -> dict[str, Path]:
        del config
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "mediq_interactions.json"
        records: list[dict[str, Any]] = []
        for trial in run_result.trials:
            task = trial.hidden_state
            final_belief = trial.final_belief_state
            labels = task.option_labels
            turn_records: list[dict[str, Any]] = []
            for round_result in trial.rounds:
                candidate_scores = list(
                    (round_result.chosen.extras or {}).get("candidate_scores", [])
                )
                candidate_actions = list(round_result.candidates) or [
                    round_result.chosen.action
                ]
                candidate_details: list[dict[str, Any]] = []
                for candidate_index, candidate in enumerate(candidate_actions):
                    prior = np.asarray(candidate.prior_probabilities, dtype=float)
                    semantic_validation = self._candidate_validation_for(candidate)
                    cached = all(
                        (label, candidate) in self._likelihood_cache for label in labels
                    )
                    likelihoods = (
                        np.asarray(
                            [
                                self._likelihood_cache[(label, candidate)]
                                for label in labels
                            ],
                            dtype=float,
                        )
                        if cached
                        else None
                    )
                    predictive = (
                        prior @ likelihoods
                        if likelihoods is not None and prior.size == len(labels)
                        else None
                    )
                    candidate_details.append(
                        {
                            "query": candidate.query,
                            "outcomes": list(candidate.outcomes),
                            "semantic_validation": (
                                {
                                    "valid": semantic_validation[0],
                                    "reason": semantic_validation[1],
                                }
                                if semantic_validation is not None
                                else None
                            ),
                            "score": (
                                candidate_scores[candidate_index]
                                if candidate_index < len(candidate_scores)
                                else None
                            ),
                            "prior": (
                                dict(zip(labels, prior.tolist(), strict=True))
                                if prior.size == len(labels)
                                else None
                            ),
                            "likelihoods": (
                                {
                                    label: likelihoods[index].tolist()
                                    for index, label in enumerate(labels)
                                }
                                if likelihoods is not None
                                else None
                            ),
                            "record_answerability_probability": (
                                self._record_availability_cache[candidate][0]
                                if candidate in self._record_availability_cache
                                else None
                            ),
                            "data_estimation_marginal": (
                                dict(
                                    zip(
                                        candidate.outcomes,
                                        self._data_estimation_marginal_cache[candidate],
                                        strict=True,
                                    )
                                )
                                if candidate in self._data_estimation_marginal_cache
                                else None
                            ),
                            "data_estimation_elicited_posteriors": (
                                {
                                    outcome: dict(
                                        zip(
                                            labels,
                                            self._data_estimation_posterior_cache[
                                                (outcome, candidate)
                                            ],
                                            strict=True,
                                        )
                                    )
                                    for outcome in (YES_OUTCOME, NO_OUTCOME)
                                }
                                if all(
                                    (outcome, candidate)
                                    in self._data_estimation_posterior_cache
                                    for outcome in (YES_OUTCOME, NO_OUTCOME)
                                )
                                else None
                            ),
                            "joint_projection_residual": (
                                self._data_estimation_projection_residuals[candidate]
                                if candidate
                                in self._data_estimation_projection_residuals
                                else None
                            ),
                            "unavailable_likelihood_span": (
                                float(
                                    np.max(likelihoods[:, -1])
                                    - np.min(likelihoods[:, -1])
                                )
                                if likelihoods is not None
                                else None
                            ),
                            "predictive_outcome_probabilities": (
                                dict(
                                    zip(
                                        candidate.outcomes,
                                        predictive.tolist(),
                                        strict=True,
                                    )
                                )
                                if predictive is not None
                                else None
                            ),
                        }
                    )
                set_validation = self._candidate_set_validation_for(candidate_actions)
                turn_records.append(
                    {
                        "query": round_result.chosen.action.query,
                        "outcomes": list(round_result.chosen.action.outcomes),
                        "candidate_queries": [
                            candidate.query for candidate in candidate_actions
                        ],
                        "candidate_details": candidate_details,
                        "candidate_set_semantic_validation": (
                            {
                                "valid": set_validation[0],
                                "reason": set_validation[1],
                            }
                            if set_validation is not None
                            else None
                        ),
                        "selected_score": round_result.chosen.score,
                        "selection_extras": round_result.chosen.extras,
                        "reply": round_result.observation.reply,
                        "selected_fact_indices": list(
                            round_result.observation.selected_fact_indices
                        ),
                        "mapped_outcome": round_result.observation.mapped_outcome,
                        "mapped_cleanly": round_result.observation.mapped_cleanly,
                        "grounded": round_result.observation.grounded,
                        "relevant": round_result.observation.relevant,
                        "cannot_answer": round_result.observation.cannot_answer,
                        "metrics": round_result.metrics,
                    }
                )
            records.append(
                {
                    "task_id": task.task_id,
                    "source_id": task.source_id,
                    "dataset": task.dataset,
                    "question": task.question,
                    "options": dict(task.options),
                    "answer_idx": task.answer_idx,
                    "answer": task.answer,
                    "initial_info": task.initial_info,
                    "full_context": list(task.context),
                    "facts": list(task.facts),
                    "turns": turn_records,
                    "final_belief": (
                        {
                            label: probability
                            for label, probability in zip(
                                final_belief.hypotheses,
                                final_belief.probabilities,
                            )
                        }
                        if final_belief is not None and final_belief.hypotheses
                        else None
                    ),
                    "final_metrics": trial.final_metrics,
                }
            )
        path.write_text(json.dumps(records, indent=2) + "\n")
        manifest_path = output_dir / "mediq_data_manifest.json"
        dataset = str(getattr(self.config, "mediq_dataset", "imedqa"))
        manifest_path.write_text(
            json.dumps(
                {
                    "repository": MEDIQ_REPOSITORY,
                    "commit": MEDIQ_COMMIT,
                    "dataset": dataset,
                    "likelihood_mode": str(
                        getattr(self.config, "mediq_likelihood_mode", "joint_option")
                    ),
                    "expected_sha256": (
                        MEDIQ_IMEDQA_DEV_SHA256
                        if dataset == "imedqa"
                        else MEDIQ_ICRAFT_MD_SHA256
                    ),
                    "raw_row_count": self._raw_row_count,
                    "usable_row_count": self._raw_row_count
                    - len(self._excluded_source_ids),
                    "skip_unusable_tasks": bool(
                        getattr(self.config, "mediq_skip_unusable_tasks", True)
                    ),
                    "excluded_source_ids": list(self._excluded_source_ids),
                    "task_offset_within_usable_rows": int(
                        getattr(self.config, "mediq_task_offset", 0)
                    ),
                    "selected_source_ids": [task.source_id for task in self.tasks],
                },
                indent=2,
            )
            + "\n"
        )
        return {
            "mediq_data_manifest": manifest_path,
            "mediq_interactions": path,
        }
