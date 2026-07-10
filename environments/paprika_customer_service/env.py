"""Core environment adapter for Paprika customer-service troubleshooting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from methods.categorical_eig import CategoricalEIG

from .data import load_paprika_tasks
from .parsing import parse_distribution, parse_json_object, parse_string_list
from .prompts import (
    candidate_messages,
    customer_messages,
    hypothesis_messages,
    judge_messages,
    likelihood_messages,
    mapping_messages,
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
        self._scenario_by_support: dict[tuple[str, ...], str] = {}
        self._likelihood_cache: dict[tuple[str, PaprikaAction], tuple[float, ...]] = {}

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

    def _questioner(self) -> Any:
        if self.questioner is None:
            raise RuntimeError("Paprika environment has no attached questioner")
        return self.questioner

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
            text = _complete(
                self._questioner(),
                likelihood_messages(hypothesis, action),
                float(getattr(self.config, "generation_temperature_simple", 0.0)),
            )
            self._likelihood_cache[key] = parse_distribution(text, action.outcomes)
        return self._likelihood_cache[key]

    def outcome_likelihoods(self, hypotheses: Sequence[str], action: PaprikaAction) -> np.ndarray:
        return np.asarray([self._likelihood_for(hypothesis, action) for hypothesis in hypotheses])

    def log_likelihood_many(self, hypotheses: Sequence[str], action: PaprikaAction, observation: PaprikaObservation) -> np.ndarray:
        return np.asarray([self.log_likelihood(h, action, observation) for h in hypotheses])

    def _initial_for_task(self, task: PaprikaTask, model: Any, config: Any) -> BeliefState[str]:
        count = int(getattr(config, "paprika_num_hypotheses", 12))
        text = _complete(model, hypothesis_messages(task.scenario, count), float(getattr(config, "generation_temperature_diverse", 1.0)))
        hypotheses = parse_string_list(text, "hypotheses", minimum=count, maximum=count)
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
        del model, config
        if (
            not history
            or not history[-1][1].mapped_cleanly
            or history[-1][1].mapped_outcome is None
        ):
            return belief_state
        action, observation = history[-1]
        log_weights = np.log(np.maximum(np.asarray(belief_state.probabilities), 1e-300))
        log_weights += self.log_likelihood_many(belief_state.hypotheses, action, observation)
        updated = BeliefState.from_log_scores(belief_state.hypotheses, log_weights)
        self._scenario_by_support[updated.hypotheses] = action.scenario
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
                actions.append(PaprikaAction(item["query"], tuple(item["outcomes"]), scenario, transcript))
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
        text = _complete(model, candidate_messages(scenario, belief_state.hypotheses, history, count), float(getattr(config, "generation_temperature_diverse", 1.0)))
        return self._parse_candidates(text, scenario, history, count)

    def observe(self, action: PaprikaAction, hidden_state: PaprikaTask, rng: np.random.Generator) -> PaprikaObservation:
        del rng
        reply = _complete(self.answerer, customer_messages(action, hidden_state.solution), float(getattr(self.config, "answer_temperature", 0.7))).strip()
        customer_goal = reply.casefold() == "goal reached"
        judge = _complete(self._questioner(), judge_messages(hidden_state.scenario, hidden_state.solution, action.query), 0.0)
        goal = customer_goal or ("<VALID>" in judge and "<NOTVALID>" not in judge)
        if customer_goal:
            return PaprikaObservation(reply=reply, mapped_outcome=None, mapped_cleanly=True, goal_reached=True)
        mapping = parse_json_object(_complete(self._questioner(), mapping_messages(reply, action.outcomes), 0.0))
        selected = mapping.get("outcome")
        clean = mapping.get("clean") is True and isinstance(selected, str)
        canonical = next((outcome for outcome in action.outcomes if clean and outcome.casefold() == selected.strip().casefold()), None)
        return PaprikaObservation(reply=reply, mapped_outcome=canonical, mapped_cleanly=canonical is not None, goal_reached=goal)

    def round_metrics(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], hidden_state: PaprikaTask) -> dict[str, float]:
        latest = history[-1][1]
        solved = any(observation.goal_reached for _action, observation in history)
        clean_count = sum(observation.mapped_cleanly for _action, observation in history)
        exact_mass = sum(probability for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities) if hypothesis.casefold() == hidden_state.solution.casefold())
        return {"resolved": float(solved), "turns_used": float(len(history)), "answer_set_coverage": clean_count / len(history), "latest_answer_mapped_cleanly": float(latest.mapped_cleanly), "true_solution_exact_mass": float(exact_mass)}

    def early_stop(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], hidden_state: PaprikaTask, latest_observation: PaprikaObservation) -> bool:
        del belief_state, history, hidden_state
        return latest_observation.goal_reached

    def build_eig_method(self, config: Any) -> CategoricalEIG:
        del config
        return CategoricalEIG()

    def generate_naive_action(self, belief_state: BeliefState[str], history: Sequence[tuple[PaprikaAction, PaprikaObservation]], model: Any, config: Any, *, method_name: str | None = None) -> PaprikaAction:
        del method_name
        scenario = history[0][0].scenario if history else self._scenario_by_support.get(belief_state.hypotheses)
        if not scenario and self._active_task is not None:
            scenario = self._active_task.scenario
        if not scenario:
            raise RuntimeError("Could not determine Paprika scenario")
        text = _complete(model, candidate_messages(scenario, (), history, 1), float(getattr(config, "generation_temperature_simple", 0.7)))
        return self._parse_candidates(text, scenario, history, 1)[0]

    def save_artifacts(self, run_result: Any, output_dir: Path, config: Any) -> dict[str, Path]:
        del config
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "paprika_smoke.json"
        records = []
        for trial in run_result.trials:
            records.append({"task_id": trial.hidden_state.task_id, "scenario": trial.hidden_state.scenario, "solution": trial.hidden_state.solution, "turns": [{"query": round_result.chosen.action.query, "outcomes": list(round_result.chosen.action.outcomes), "reply": round_result.observation.reply, "mapped_outcome": round_result.observation.mapped_outcome, "mapped_cleanly": round_result.observation.mapped_cleanly, "goal_reached": round_result.observation.goal_reached} for round_result in trial.rounds], "final_metrics": trial.final_metrics})
        path.write_text(json.dumps(records, indent=2) + "\n")
        return {"paprika_smoke": path}
