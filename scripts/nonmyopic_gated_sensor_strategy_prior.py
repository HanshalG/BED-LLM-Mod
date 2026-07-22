"""Paired LLM branch-policy experiment on exact gated sensor diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any, Literal, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gated_sensor import (
    ExactGatedStrategyScore,
    GatedBranchStrategy,
    GatedSensorModel,
    GatedStrategyParseError,
    SensorState,
    parse_gated_strategy_cell,
    score_gated_strategy_exact,
)
from environments.gated_sensor.model import EPSILON
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_oracle import exact_action_values


ArmName = Literal["strategy_eig", "shared_d1", "exhaustive_d1", "random_strategy", "exhaustive_d2"]
ARMS: tuple[ArmName, ...] = (
    "strategy_eig",
    "shared_d1",
    "exhaustive_d1",
    "random_strategy",
    "exhaustive_d2",
)


class StrategyProposalError(RuntimeError):
    """A model response failed the bounded validation policy."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class StrategyConfig:
    num_trials: int = 30
    num_rounds: int = 8
    num_strategies: int = 6
    seed: int = 24_092
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 32
    temperature: float = 0.0
    validation_retries: int = 1
    screen_accuracy: float = 0.65
    precise_accuracy: float = 0.95

    def validate(self) -> None:
        if min(
            self.num_trials,
            self.num_rounds,
            self.num_strategies,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, strategy, bootstrap, and concurrency counts must be positive")
        if self.num_strategies < 4:
            raise ValueError("num_strategies must cover three initial activation roots and a measurement root")
        if self.validation_retries != 1:
            raise ValueError("the registered interface permits exactly one validation retry")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        GatedSensorModel(
            screen_accuracy=self.screen_accuracy,
            precise_accuracy=self.precise_accuracy,
        )


@dataclass(frozen=True)
class StrategyCell:
    strategies: tuple[GatedBranchStrategy, ...]
    scores: tuple[ExactGatedStrategyScore, ...]
    raw_response: str
    cache_hit: bool


@dataclass(frozen=True)
class Selection:
    action: str
    planning_score: float
    immediate_eig: float
    candidate_roots: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    selected_strategy: str | None
    candidate_strategies: tuple[str, ...]
    scorer_units: int
    logical_llm_calls: int


@dataclass
class PolicyState:
    belief: np.ndarray
    sensor_state: SensorState
    history: tuple[tuple[str, str | None], ...] = ()
    action_counts: dict[str, int] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def _required_activation_roots(model: GatedSensorModel, state: SensorState, horizon: int) -> tuple[str, ...]:
    if horizon <= 1:
        return ()
    return tuple(action for action in model.legal_actions(state) if action.startswith("activate:"))


def _predicate_summary(model: GatedSensorModel, belief: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predicate in model.predicates:
        positive = float(
            sum(
                probability
                for hidden, probability in zip(model.hidden_states, belief)
                if predicate.evaluate(hidden)
            )
        )
        entropy = 0.0
        for probability in (positive, 1.0 - positive):
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        rows.append(
            {
                "predicate": predicate.name,
                "p_true": round(positive, 8),
                "binary_entropy": round(entropy, 8),
            }
        )
    return rows


def _repair_terminal_roots(
    response: str,
    *,
    model: GatedSensorModel,
    state: SensorState,
) -> tuple[str, int]:
    """Replace duplicate or activation roots in a horizon-one cell deterministically."""

    normalized = response.strip()
    if "```json\n" in normalized:
        start = normalized.rfind("```json\n") + len("```json\n")
        end = normalized.find("\n```", start)
        if end < 0:
            return response, 0
        normalized = normalized[start:end]
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return response, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("strategies"), list):
        return response, 0
    available = [action for action in model.legal_actions(state) if not action.startswith("activate:")]
    used: set[str] = set()
    repairs = 0
    for item in payload["strategies"]:
        if not isinstance(item, dict):
            continue
        root = item.get("root_action")
        if isinstance(root, str) and root in available and root not in used:
            used.add(root)
            continue
        replacement = next((action for action in available if action not in used), None)
        if replacement is None:
            return response, 0
        item["root_action"] = replacement
        used.add(replacement)
        repairs += 1
    if repairs == 0:
        return response, 0
    return json.dumps(payload, sort_keys=True, separators=(",", ":")), repairs


class GatedStrategyProvider:
    """Generate complete branch-policy cells with one bounded repair."""

    def __init__(self, chat_model: ChatModel, config: StrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
        self._cache: dict[tuple[Any, ...], StrategyCell] = {}
        self._lock = threading.Lock()
        self.logical_calls = 0
        self.cache_hits = 0
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.terminal_root_repairs = 0

    @staticmethod
    def _state_from_history(
        model: GatedSensorModel,
        serialized_history: list[dict[str, Any]],
    ) -> tuple[SensorState, np.ndarray, tuple[tuple[str, str | None], ...]]:
        state = model.initial_state
        belief = model.initial_belief.copy()
        history: list[tuple[str, str | None]] = []
        for index, row in enumerate(serialized_history):
            if not isinstance(row, dict) or set(row) != {"action", "observation"}:
                raise ValueError(f"history row {index} has invalid fields")
            action = row["action"]
            outcome = row["observation"]
            if not isinstance(action, str) or action not in model.legal_actions(state):
                raise ValueError(f"history row {index} has illegal action {action!r}")
            if outcome not in model.outcomes(action):
                raise ValueError(f"history row {index} has invalid observation {outcome!r}")
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        return state, belief, tuple(history)

    def load_failure_cache(self, path: Path, model: GatedSensorModel) -> dict[str, Any]:
        """Revalidate and cache every accepted cell from a matching failed run."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("status") != "failed_closed":
            raise ValueError("resume artifact is not a failed-closed gated strategy run")
        if payload.get("config") != json.loads(json.dumps(asdict(self.config))):
            raise ValueError("resume artifact config does not match the requested run")
        requests = payload.get("candidate_requests")
        invalid = payload.get("invalid_responses")
        if not isinstance(requests, list) or not isinstance(invalid, list):
            raise ValueError("resume artifact request logs are invalid")
        loaded: dict[tuple[Any, ...], StrategyCell] = {}
        repairs = 0
        for index, request in enumerate(requests):
            try:
                trial_index = int(request["trial_index"])
                horizon = int(request["horizon"])
                state, belief, history = self._state_from_history(model, request["history"])
                if request["active_panel"] != state.active_panel:
                    raise ValueError("recorded active panel does not match history")
                response = str(request.get("validated_response", request["raw_response"]))
                activation_roots = _required_activation_roots(model, state, horizon)
                strategies = parse_gated_strategy_cell(
                    response,
                    model=model,
                    state=state,
                    horizon=horizon,
                    expected_count=self.config.num_strategies,
                    required_activation_roots=activation_roots,
                )
                scores = tuple(
                    score_gated_strategy_exact(
                        model,
                        strategy,
                        state=state,
                        belief=belief,
                        horizon=horizon,
                    )
                    for strategy in strategies
                )
            except (KeyError, TypeError, ValueError, GatedStrategyParseError) as exc:
                raise ValueError(f"accepted request {index} cannot be revalidated: {exc}") from exc
            key = (trial_index, state.active_panel, history, horizon)
            cell = StrategyCell(strategies, scores, response, False)
            existing = loaded.get(key)
            if existing is not None and existing.raw_response != response:
                raise ValueError(f"accepted request {index} conflicts with an earlier cell")
            loaded[key] = cell
            repairs += int(request.get("terminal_root_repairs", 0))
        with self._lock:
            self._cache.update(loaded)
            self.physical_requests.extend(requests)
            self.invalid_responses.extend(invalid)
            self.terminal_root_repairs += repairs
        return {
            "failure_artifact": str(path),
            "accepted_cells_reused": len(loaded),
            "rejected_responses_preserved": len(invalid),
        }

    def _messages(
        self,
        model: GatedSensorModel,
        *,
        state: SensorState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        horizon: int,
    ) -> list[dict[str, str]]:
        legal_roots = model.legal_actions(state)
        activation_roots = _required_activation_roots(model, state, horizon)
        measurement_slots = self.config.num_strategies - len(activation_roots)
        if measurement_slots <= 0:
            raise StrategyProposalError("candidate count leaves no measurement-root slot")
        menus: dict[str, dict[str, list[str]]] = {}
        for root in legal_roots:
            child_state = model.next_state(state, root)
            child_legal = model.legal_actions(child_state)
            outcome_keys = () if horizon <= 1 else tuple(
                "none" if outcome is None else outcome for outcome in model.outcomes(root)
            )
            menus[root] = {}
            for outcome in outcome_keys:
                choices = child_legal
                if root.startswith("activate:"):
                    choices = tuple(action for action in choices if action.startswith("precise:"))
                menus[root][outcome] = list(choices)
        system = (
            "You propose short contingent experimental-design policies for exact fault diagnosis. "
            "A separate program validates every action, enumerates observation branches, and selects "
            "the policy with highest exact information gain. Return JSON only."
        )
        instructions = [
            "STRATEGY_SCHEMA=gated_branch_policy_v1",
            f"Return exactly {self.config.num_strategies} behaviorally distinct strategies.",
            'Schema: {"strategies":[{"name":"short name","description":"reason",'
            '"root_action":"ACTION_ID","followups":{"OUTCOME":"ACTION_ID"}}]}',
            f"Planning horizon: {horizon} action(s).",
            f"Current active panel: {state.active_panel or 'none'}.",
            f"Panels and precise-test predicates: {json.dumps(model.panels, sort_keys=True)}",
            (
                f"The first {len(activation_roots)} strategies are machine-assigned activation slots. "
                f"Use these root_action IDs in exactly this order: {json.dumps(activation_roots)}. "
                "Each must use exactly {\"none\":\"PRECISE_ACTION\"}, selected from its menu."
                if horizon > 1
                else "At horizon one every strategy must use followups:{} exactly."
            ),
            (
                f"The remaining {measurement_slots} strategies must use distinct measurement roots "
                "(screen: or precise:), never activation roots. A measurement root must provide exactly "
                "the positive and negative followups listed in its menu."
                if horizon > 1
                else "Choose distinct legal measurement roots; do not choose activation roots because no followup remains."
            ),
            "Use posterior uncertainty and predicate overlap: favor tests that separate plausible fault codes, "
            "avoid redundant predicates, and make positive/negative followups adapt to what each result resolves.",
            "Current exact predicate marginals:",
            json.dumps(_predicate_summary(model, belief), separators=(",", ":")),
            "History:",
            json.dumps(
                [{"action": action, "observation": outcome} for action, outcome in history],
                separators=(",", ":"),
            ),
            "MACHINE_READABLE_MENUS=" + json.dumps(menus, sort_keys=True, separators=(",", ":")),
        ]
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(instructions)}]

    def propose(
        self,
        model: GatedSensorModel,
        *,
        trial_index: int,
        state: SensorState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        horizon: int,
    ) -> StrategyCell:
        key = (trial_index, state.active_panel, history, horizon)
        with self._lock:
            self.logical_calls += 1
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return StrategyCell(cached.strategies, cached.scores, cached.raw_response, True)
        messages = self._messages(model, state=state, belief=belief, history=history, horizon=horizon)
        context = {
            "trial_index": trial_index,
            "active_panel": state.active_panel,
            "history": [{"action": action, "observation": outcome} for action, outcome in history],
            "horizon": horizon,
        }
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            responses = self.chat_model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise StrategyProposalError("model did not return exactly one response")
            original_response = responses[0]
            response = original_response
            terminal_repairs = 0
            if horizon <= 1:
                response, terminal_repairs = _repair_terminal_roots(
                    response,
                    model=model,
                    state=state,
                )
            try:
                activation_roots = _required_activation_roots(model, state, horizon)
                strategies = parse_gated_strategy_cell(
                    response,
                    model=model,
                    state=state,
                    horizon=horizon,
                    expected_count=self.config.num_strategies,
                    required_activation_roots=activation_roots,
                )
                scores = tuple(
                    score_gated_strategy_exact(
                        model,
                        strategy,
                        state=state,
                        belief=belief,
                        horizon=horizon,
                    )
                    for strategy in strategies
                )
            except GatedStrategyParseError as exc:
                error = exc
                with self._lock:
                    self.invalid_responses.append(
                        {**context, "attempt": attempt, "error": str(exc), "raw_response": original_response}
                    )
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": original_response},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid response: {exc}. Return the entire corrected JSON cell only. "
                                "Copy action IDs and outcome keys literally from the menus."
                            ),
                        },
                    ]
                continue
            cell = StrategyCell(strategies, scores, response, False)
            with self._lock:
                self.terminal_root_repairs += terminal_repairs
                self.physical_requests.append(
                    {
                        **context,
                        "attempt": attempt,
                        "raw_response": original_response,
                        "validated_response": response,
                        "terminal_root_repairs": terminal_repairs,
                    }
                )
                self._cache[key] = cell
            return cell
        raise StrategyProposalError(f"strategy cell failed after two attempts: {error}")


class DeterministicStrategyModel:
    """Valid zero-spend model for end-to-end mechanics tests."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic model supports one response")
        content = messages[-1]["content"]
        count = int(content.split("Return exactly ", 1)[1].split(" behaviorally", 1)[0])
        horizon = int(content.split("Planning horizon: ", 1)[1].split(" action", 1)[0])
        menus = json.loads(content.split("MACHINE_READABLE_MENUS=", 1)[1].split("\n", 1)[0])
        activation_roots = [root for root in menus if root.startswith("activate:")] if horizon > 1 else []
        measurement_roots = [root for root in menus if not root.startswith("activate:")]
        roots = [*activation_roots, *measurement_roots[: count - len(activation_roots)]]
        strategies = []
        for index, root in enumerate(roots):
            followups = {
                outcome: choices[(index + branch_index) % len(choices)]
                for branch_index, (outcome, choices) in enumerate(menus[root].items())
            }
            strategies.append(
                {
                    "name": f"deterministic-{index}",
                    "description": "A complete legal policy for deterministic mechanics testing.",
                    "root_action": root,
                    "followups": followups,
                }
            )
        return [json.dumps({"strategies": strategies})]


def _choose(scores: list[float] | tuple[float, ...]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _strategy_selection(
    model: GatedSensorModel,
    provider: GatedStrategyProvider,
    state: PolicyState,
    *,
    trial_index: int,
    horizon: int,
    myopic: bool,
) -> Selection:
    cell = provider.propose(
        model,
        trial_index=trial_index,
        state=state.sensor_state,
        belief=state.belief,
        history=state.history,
        horizon=horizon,
    )
    scores = tuple(
        model.expected_information_gain(state.belief, strategy.root_action)
        if myopic
        else exact_score.eig
        for strategy, exact_score in zip(cell.strategies, cell.scores, strict=True)
    )
    index = _choose(scores)
    strategy = cell.strategies[index]
    return Selection(
        action=strategy.root_action,
        planning_score=scores[index],
        immediate_eig=model.expected_information_gain(state.belief, strategy.root_action),
        candidate_roots=tuple(candidate.root_action for candidate in cell.strategies),
        candidate_scores=scores,
        selected_strategy=strategy.raw_text,
        candidate_strategies=tuple(candidate.raw_text for candidate in cell.strategies),
        scorer_units=(len(cell.strategies) if myopic else sum(score.expanded_decision_nodes for score in cell.scores)),
        logical_llm_calls=1,
    )


def _random_strategies(
    model: GatedSensorModel,
    *,
    state: SensorState,
    horizon: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[GatedBranchStrategy, ...]:
    legal = model.legal_actions(state)
    activation_roots = _required_activation_roots(model, state, horizon)
    measurement_roots = [action for action in legal if not action.startswith("activate:")]
    selected = rng.choice(measurement_roots, size=count - len(activation_roots), replace=False)
    roots = (*activation_roots, *(str(action) for action in selected))
    items: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        followups: dict[str, str] = {}
        if horizon > 1:
            child_state = model.next_state(state, root)
            choices = model.legal_actions(child_state)
            if root.startswith("activate:"):
                choices = tuple(action for action in choices if action.startswith("precise:"))
            for outcome in model.outcomes(root):
                key = "none" if outcome is None else outcome
                followups[key] = choices[int(rng.integers(len(choices)))]
        items.append(
            {
                "name": f"random-{index}",
                "description": "A policy sampled from the matched legal branch grammar.",
                "root_action": root,
                "followups": followups,
            }
        )
    return parse_gated_strategy_cell(
        json.dumps({"strategies": items}),
        model=model,
        state=state,
        horizon=horizon,
        expected_count=count,
        required_activation_roots=activation_roots,
    )


def _random_selection(
    model: GatedSensorModel,
    state: PolicyState,
    config: StrategyConfig,
    *,
    trial_index: int,
    round_index: int,
    horizon: int,
) -> Selection:
    rng = np.random.default_rng(_stable_seed(config.seed, "random-cell", trial_index, round_index))
    strategies = _random_strategies(
        model,
        state=state.sensor_state,
        horizon=horizon,
        count=config.num_strategies,
        rng=rng,
    )
    scores = tuple(
        score_gated_strategy_exact(
            model,
            strategy,
            state=state.sensor_state,
            belief=state.belief,
            horizon=horizon,
        )
        for strategy in strategies
    )
    index = _choose(tuple(score.eig for score in scores))
    strategy = strategies[index]
    return Selection(
        action=strategy.root_action,
        planning_score=scores[index].eig,
        immediate_eig=model.expected_information_gain(state.belief, strategy.root_action),
        candidate_roots=tuple(candidate.root_action for candidate in strategies),
        candidate_scores=tuple(score.eig for score in scores),
        selected_strategy=strategy.raw_text,
        candidate_strategies=tuple(candidate.raw_text for candidate in strategies),
        scorer_units=sum(score.expanded_decision_nodes for score in scores),
        logical_llm_calls=0,
    )


def _exhaustive_selection(
    model: GatedSensorModel,
    state: PolicyState,
    *,
    depth: int,
) -> Selection:
    values, scorer_units = exact_action_values(
        model,
        state=state.sensor_state,
        belief=state.belief,
        depth=depth,
    )
    roots = tuple(values)
    scores = tuple(values[root] for root in roots)
    index = _choose(scores)
    action = roots[index]
    return Selection(
        action=action,
        planning_score=scores[index],
        immediate_eig=model.expected_information_gain(state.belief, action),
        candidate_roots=roots,
        candidate_scores=scores,
        selected_strategy=None,
        candidate_strategies=(),
        scorer_units=scorer_units,
        logical_llm_calls=0,
    )


def _apply_selection(
    model: GatedSensorModel,
    state: PolicyState,
    selection: Selection,
    *,
    arm: ArmName,
    truth_index: int,
    config: StrategyConfig,
    trial_index: int,
    round_index: int,
) -> None:
    if selection.action not in model.legal_actions(state.sensor_state):
        raise AssertionError(f"selected illegal action {selection.action} for {arm}")
    if selection.action.startswith("activate:"):
        outcome: str | None = None
    else:
        repeat_index = state.action_counts.get(selection.action, 0)
        state.action_counts[selection.action] = repeat_index + 1
        probability_positive = float(model.likelihood_vector(selection.action, "positive")[truth_index])
        outcome = (
            "positive"
            if _uniform(
                config.seed,
                "gated-strategy-observation",
                trial_index,
                selection.action,
                repeat_index,
            )
            < probability_positive
            else "negative"
        )
    entropy_before = model.entropy(state.belief)
    posterior = model.posterior(state.belief, selection.action, outcome)
    state.steps.append(
        {
            "round": round_index + 1,
            "state_before": state.sensor_state.active_panel,
            "action": selection.action,
            "observation": outcome,
            "entropy_before": entropy_before,
            "entropy_after": model.entropy(posterior),
            "realized_entropy_drop": entropy_before - model.entropy(posterior),
            "truth_log_probability": math.log(
                max(float(posterior[truth_index]), np.finfo(float).tiny)
            ),
            **asdict(selection),
        }
    )
    state.history = state.history + ((selection.action, outcome),)
    state.sensor_state = model.next_state(state.sensor_state, selection.action)
    state.belief = posterior


def _run_trial(
    trial_index: int,
    *,
    model: GatedSensorModel,
    provider: GatedStrategyProvider,
    config: StrategyConfig,
) -> dict[str, Any]:
    truth_rng = np.random.default_rng(_stable_seed(config.seed, "gated-strategy-truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    states = {
        arm: PolicyState(model.initial_belief.copy(), model.initial_state)
        for arm in ARMS
    }
    for round_index in range(config.num_rounds):
        horizon = min(2, config.num_rounds - round_index)
        for arm in ARMS:
            state = states[arm]
            if arm == "strategy_eig":
                selection = _strategy_selection(
                    model, provider, state, trial_index=trial_index, horizon=horizon, myopic=False
                )
            elif arm == "shared_d1":
                selection = _strategy_selection(
                    model, provider, state, trial_index=trial_index, horizon=horizon, myopic=True
                )
            elif arm == "random_strategy":
                selection = _random_selection(
                    model,
                    state,
                    config,
                    trial_index=trial_index,
                    round_index=round_index,
                    horizon=horizon,
                )
            elif arm == "exhaustive_d1":
                selection = _exhaustive_selection(model, state, depth=1)
            else:
                selection = _exhaustive_selection(model, state, depth=horizon)
            _apply_selection(
                model,
                state,
                selection,
                arm=arm,
                truth_index=truth_index,
                config=config,
                trial_index=trial_index,
                round_index=round_index,
            )
    traces: dict[str, Any] = {}
    for arm, state in states.items():
        entropy = [step["entropy_after"] for step in state.steps]
        truth = [step["truth_log_probability"] for step in state.steps]
        traces[arm] = {
            "arm": arm,
            "trial_index": trial_index,
            "truth_index": truth_index,
            "entropy_auc": float(np.mean(entropy)),
            "truth_log_probability_auc": float(np.mean(truth)),
            "final_entropy": entropy[-1],
            "final_truth_log_probability": truth[-1],
            "final_map_accuracy": float(model.decode_map_index(state.belief) == truth_index),
            "steps": state.steps,
        }
    return traces


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    batches = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        batches.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(batches)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _comparison(
    strategy: list[dict[str, Any]],
    control: list[dict[str, Any]],
    *,
    label: str,
    config: StrategyConfig,
) -> dict[str, Any]:
    entropy = np.asarray(
        [control[index]["entropy_auc"] - strategy[index]["entropy_auc"] for index in range(len(strategy))]
    )
    truth = np.asarray(
        [
            strategy[index]["truth_log_probability_auc"]
            - control[index]["truth_log_probability_auc"]
            for index in range(len(strategy))
        ]
    )
    return {
        "entropy_auc_gain_mean": float(np.mean(entropy)),
        "entropy_auc_gain_ci95": _bootstrap_ci(
            entropy,
            seed=_stable_seed(config.seed, label, "entropy-bootstrap"),
            replicates=config.bootstrap_replicates,
        ),
        "truth_log_probability_auc_gain_mean": float(np.mean(truth)),
        "truth_log_probability_auc_gain_ci95": _bootstrap_ci(
            truth,
            seed=_stable_seed(config.seed, label, "truth-bootstrap"),
            replicates=config.bootstrap_replicates,
        ),
        "entropy_auc_wins_ties_losses": [
            int(np.count_nonzero(entropy > EPSILON)),
            int(np.count_nonzero(np.abs(entropy) <= EPSILON)),
            int(np.count_nonzero(entropy < -EPSILON)),
        ],
        "entropy_auc_paired_values": entropy.tolist(),
        "truth_log_probability_auc_paired_values": truth.tolist(),
    }


def run_experiment(provider: GatedStrategyProvider, config: StrategyConfig) -> dict[str, Any]:
    config.validate()
    model = GatedSensorModel(
        screen_accuracy=config.screen_accuracy,
        precise_accuracy=config.precise_accuracy,
    )
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        trials = list(
            executor.map(
                lambda trial_index: _run_trial(
                    trial_index,
                    model=model,
                    provider=provider,
                    config=config,
                ),
                range(config.num_trials),
            )
        )
    traces = {arm: [trial[arm] for trial in trials] for arm in ARMS}
    comparisons = {
        f"strategy_eig_minus_{control}": _comparison(
            traces["strategy_eig"],
            traces[control],
            label=f"strategy-minus-{control}",
            config=config,
        )
        for control in ("shared_d1", "exhaustive_d1", "random_strategy", "exhaustive_d2")
    }
    primary_controls = ("shared_d1", "exhaustive_d1", "random_strategy")
    mechanics = {
        "paired_trials_and_truths": all(
            [(trace["trial_index"], trace["truth_index"]) for trace in traces[arm]]
            == [(trace["trial_index"], trace["truth_index"]) for trace in traces["strategy_eig"]]
            for arm in ARMS
        ),
        "all_selected_actions_legal": all(
            step["action"] in model.legal_actions(SensorState(step["state_before"]))
            for arm_traces in traces.values()
            for trace in arm_traces
            for step in trace["steps"]
        ),
        "strategy_initial_cells_cover_all_activation_roots": all(
            set(step["candidate_roots"][:3]) == {"activate:A", "activate:B", "activate:C"}
            for trace in traces["strategy_eig"]
            for step in trace["steps"][:1]
        ),
        "rollout_scoring_llm_calls": 0,
        "bounded_validation_retries": len(provider.invalid_responses),
    }
    gate_passed = (
        all(mechanics[key] for key in (
            "paired_trials_and_truths",
            "all_selected_actions_legal",
            "strategy_initial_cells_cover_all_activation_roots",
        ))
        and all(
            comparisons[f"strategy_eig_minus_{control}"]["entropy_auc_gain_ci95"][0] > 0.0
            for control in primary_controls
        )
    )
    return {
        "schema_version": 1,
        "config": asdict(config),
        "comparisons": comparisons,
        "mechanics": mechanics,
        "gate": {
            "passed": gate_passed,
            "rule": "StrategyEIG entropy-AUC lower CI exceeds shared d1, exhaustive d1, and matched random",
        },
        "provider": {
            "logical_calls": provider.logical_calls,
            "physical_requests": len(provider.physical_requests) + len(provider.invalid_responses),
            "accepted_requests": len(provider.physical_requests),
            "invalid_responses": len(provider.invalid_responses),
            "cache_hits": provider.cache_hits,
            "terminal_root_repairs": provider.terminal_root_repairs,
        },
        "traces": traces,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_rocksample_7_8_scale_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/gated_sensor_strategy_prior_20260722"),
    )
    parser.add_argument("--run-id", default="nonmyopic-gated-sensor-strategy-prior-20260722")
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--num-strategies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=24_092)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-failure", type=Path)
    args = parser.parse_args()
    config = StrategyConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_strategies=args.num_strategies,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicStrategyModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        if not runtime_config.model_pairs:
            raise ValueError("strategy config requires a questioner model")
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = GatedStrategyProvider(chat_model, config)
    resume_info = None
    if args.resume_failure is not None:
        resume_model = GatedSensorModel(
            screen_accuracy=config.screen_accuracy,
            precise_accuracy=config.precise_accuracy,
        )
        resume_info = provider.load_failure_cache(args.resume_failure, resume_model)
    try:
        summary = run_experiment(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
            "resume": resume_info,
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    summary["usage"] = _usage_snapshot(chat_model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    summary["resume"] = resume_info
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": summary["gate"], "provider": summary["provider"]}, indent=2))


if __name__ == "__main__":
    main()
