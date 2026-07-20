"""Paired StrategyEIG anchor on exact Rock Diagnosis.

The LLM supplies compact reactive strategies. Rock dynamics, posterior updates,
strategy rollout EIG, policy selection, observations, and decoding are exact.
"""

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

from environments.rock_diagnosis import (
    ExactRockStrategyScore,
    RockDiagnosisModel,
    RockStrategy,
    RockStrategyExecutionError,
    RockStrategyExecutor,
    RockStrategyParseError,
    get_paper_map,
    parse_rock_strategy,
    random_rock_strategy_text,
    score_rock_strategy_exact,
)
from environments.rock_diagnosis.core import EPSILON
from helpers import Config, load_config
from model_factory import build_model_adapter


History = tuple[tuple[str, str | None], ...]
ArmName = Literal["strategy_eig", "exhaustive_d2", "shared_d1", "width", "random_strategy"]
StrategySchema = Literal["reactive_rules_v1", "branch_policy_v2"]
PrimaryEndpoint = Literal["final_entropy", "entropy_auc"]
ARMS: tuple[ArmName, ...] = (
    "strategy_eig",
    "exhaustive_d2",
    "shared_d1",
    "width",
    "random_strategy",
)


class StrategyProposalError(RuntimeError):
    """A complete strategy or width cell failed the bounded validation policy."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class L1Config:
    map_names: tuple[str, ...] = ("3-6", "5-7")
    num_trials_per_map: int = 30
    num_rounds: int = 8
    num_strategies: int = 4
    planning_horizon: int = 2
    seed: int = 12_032
    bootstrap_replicates: int = 10_000
    temperature: float = 0.0
    validation_retries: int = 1
    trial_concurrency: int = 32
    strategy_schema: StrategySchema = "reactive_rules_v1"
    primary_endpoint: PrimaryEndpoint = "final_entropy"

    def validate(self) -> None:
        if not self.map_names:
            raise ValueError("map_names must not be empty")
        for map_name in self.map_names:
            get_paper_map(map_name)
        if min(
            self.num_trials_per_map,
            self.num_rounds,
            self.planning_horizon,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, strategy, horizon, and bootstrap counts must be positive")
        if self.num_strategies < 2:
            raise ValueError("num_strategies must be at least two for the registered root-mix control")
        if self.planning_horizon != 2:
            raise ValueError("the registered L1 anchor uses planning_horizon=2")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.validation_retries != 1:
            raise ValueError("the registered interface permits exactly one validation-feedback retry")
        if self.strategy_schema not in {"reactive_rules_v1", "branch_policy_v2"}:
            raise ValueError("strategy_schema must be reactive_rules_v1 or branch_policy_v2")
        if self.primary_endpoint not in {"final_entropy", "entropy_auc"}:
            raise ValueError("primary_endpoint must be final_entropy or entropy_auc")


@dataclass(frozen=True)
class StrategyCell:
    strategies: tuple[RockStrategy, ...]
    exact_scores: tuple[ExactRockStrategyScore, ...]
    raw_response: str
    cache_hit: bool


@dataclass(frozen=True)
class WidthCell:
    action_ids: tuple[str, ...]
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
    distinct_scored_candidates: int
    exhaustive_value: float
    exhaustive_fraction: float
    logical_llm_calls: int


@dataclass
class PolicyState:
    belief: np.ndarray
    position: tuple[int, int]
    history: History = ()
    check_counts: dict[tuple[tuple[int, int], int], int] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def _normalize_json_response(response: str) -> str:
    normalized = response.strip()
    if normalized.startswith("```json\n"):
        if not normalized.endswith("\n```"):
            raise StrategyProposalError("response has an incomplete JSON fence")
        normalized = normalized[len("```json\n") : -len("\n```")]
    return normalized


def parse_strategy_cell(
    response: str,
    *,
    model: RockDiagnosisModel,
    expected_count: int,
) -> tuple[RockStrategy, ...]:
    try:
        payload = json.loads(_normalize_json_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("strategy response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"strategies"}:
        raise StrategyProposalError("strategy response must contain exactly the strategies key")
    items = payload["strategies"]
    if not isinstance(items, list) or len(items) != expected_count:
        raise StrategyProposalError(f"expected exactly {expected_count} strategies")
    strategies: list[RockStrategy] = []
    canonical: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise StrategyProposalError(f"strategy {index} must be a JSON object")
        text = json.dumps(item, sort_keys=True, separators=(",", ":"))
        try:
            strategy = parse_rock_strategy(text, model)
        except RockStrategyParseError as exc:
            raise StrategyProposalError(f"strategy {index} failed grammar validation: {exc}") from exc
        if text in canonical:
            raise StrategyProposalError("strategies must be distinct")
        canonical.add(text)
        strategies.append(strategy)
    return tuple(strategies)


def _action_payload(model: RockDiagnosisModel, action_id: str) -> dict[str, Any]:
    if model.is_move(action_id):
        return {"kind": "move", "direction": action_id.removeprefix("move-")}
    rock_id = model.check_id(action_id)
    if rock_id is None:
        raise StrategyProposalError(f"unknown Rock action ID: {action_id}")
    return {"kind": "check_rock", "rock_id": rock_id}


def _branch_outcome_keys(model: RockDiagnosisModel, root_action: str, horizon: int) -> tuple[str, ...]:
    if horizon <= 1:
        return ()
    return tuple("none" if outcome is None else str(outcome) for outcome in model.outcomes(root_action))


def _required_move_policy_count(
    model: RockDiagnosisModel,
    position: tuple[int, int],
    *,
    horizon: int,
    total_count: int,
) -> int:
    if horizon <= 1:
        return 0
    legal_move_count = sum(model.is_move(action) for action in model.legal_actions(position))
    return min(legal_move_count, max(1, total_count // 2))


def _compile_branch_policy(
    item: Any,
    *,
    model: RockDiagnosisModel,
    position: tuple[int, int],
    horizon: int,
    index: int,
) -> tuple[RockStrategy, tuple[Any, ...]]:
    if not isinstance(item, dict) or set(item) != {"name", "description", "root_action", "followups"}:
        raise StrategyProposalError(
            f"strategy {index} must contain exactly name, description, root_action, and followups"
        )
    name = item["name"]
    description = item["description"]
    root_action = item["root_action"]
    followups = item["followups"]
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise StrategyProposalError(f"strategy {index} name must be a non-empty string of at most 80 characters")
    if not isinstance(description, str) or not description.strip() or len(description) > 800:
        raise StrategyProposalError(
            f"strategy {index} description must be a non-empty string of at most 800 characters"
        )
    if not isinstance(root_action, str) or root_action not in model.legal_actions(position):
        raise StrategyProposalError(f"strategy {index} root_action must be legal at {position}")
    if not isinstance(followups, dict):
        raise StrategyProposalError(f"strategy {index} followups must be a JSON object")
    expected_keys = _branch_outcome_keys(model, root_action, horizon)
    if set(followups) != set(expected_keys):
        raise StrategyProposalError(
            f"strategy {index} followups must contain exactly {list(expected_keys)} for root {root_action}"
        )
    child_position = model.next_position(position, root_action)
    child_legal = model.legal_actions(child_position)
    for outcome_key in expected_keys:
        action = followups[outcome_key]
        if not isinstance(action, str):
            raise StrategyProposalError(
                f"strategy {index} followup {outcome_key!r} must be a string action ID, not an object"
            )
        if action not in child_legal:
            raise StrategyProposalError(
                f"strategy {index} followup {outcome_key!r} must be legal at {child_position}"
            )

    rules: list[dict[str, Any]] = []
    if expected_keys == ("good", "bad"):
        check_id = model.check_id(root_action)
        assert check_id is not None
        rules.extend(
            [
                {
                    "when": [
                        {"kind": "step_at_least", "value": 1},
                        {"kind": "last_observation", "rock_id": check_id, "outcome": "good"},
                    ],
                    "action": _action_payload(model, followups["good"]),
                },
                {
                    "when": [{"kind": "step_at_least", "value": 1}],
                    "action": _action_payload(model, followups["bad"]),
                },
            ]
        )
    elif expected_keys == ("none",):
        rules.append(
            {
                "when": [{"kind": "step_at_least", "value": 1}],
                "action": _action_payload(model, followups["none"]),
            }
        )
    rules.append({"when": [], "action": _action_payload(model, root_action)})
    compiled = {
        "name": name.strip(),
        "description": description.strip(),
        "rules": rules,
    }
    raw_text = json.dumps(item, sort_keys=True, separators=(",", ":"))
    try:
        parsed = parse_rock_strategy(json.dumps(compiled, separators=(",", ":")), model)
    except RockStrategyParseError as exc:
        raise StrategyProposalError(f"strategy {index} could not compile: {exc}") from exc
    strategy = RockStrategy(
        name=parsed.name,
        description=parsed.description,
        rules=parsed.rules,
        raw_text=raw_text,
    )
    signature = (root_action, *(followups[key] for key in expected_keys))
    return strategy, signature


def parse_branch_strategy_cell(
    response: str,
    *,
    model: RockDiagnosisModel,
    position: tuple[int, int],
    horizon: int,
    expected_count: int,
) -> tuple[RockStrategy, ...]:
    try:
        payload = json.loads(_normalize_json_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("strategy response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"strategies"}:
        raise StrategyProposalError("strategy response must contain exactly the strategies key")
    items = payload["strategies"]
    if not isinstance(items, list) or len(items) != expected_count:
        raise StrategyProposalError(f"expected exactly {expected_count} strategies")
    strategies: list[RockStrategy] = []
    signatures: set[tuple[Any, ...]] = set()
    for index, item in enumerate(items):
        strategy, signature = _compile_branch_policy(
            item,
            model=model,
            position=position,
            horizon=horizon,
            index=index,
        )
        if signature in signatures:
            raise StrategyProposalError("strategies must be behaviorally distinct")
        signatures.add(signature)
        strategies.append(strategy)
    return tuple(strategies)


def parse_width_cell(response: str, *, allowed_actions: tuple[str, ...]) -> tuple[str, ...]:
    try:
        payload = json.loads(_normalize_json_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("width response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"action_ids"}:
        raise StrategyProposalError("width response must contain exactly the action_ids key")
    action_ids = payload["action_ids"]
    if not isinstance(action_ids, list) or not all(isinstance(item, str) for item in action_ids):
        raise StrategyProposalError("action_ids must be a list of strings")
    if len(action_ids) != len(allowed_actions) or len(set(action_ids)) != len(action_ids):
        raise StrategyProposalError("width response must contain each legal action exactly once")
    if set(action_ids) != set(allowed_actions):
        raise StrategyProposalError("width response contains missing or illegal actions")
    return tuple(action_ids)


def _state_label(state: tuple[str, ...]) -> str:
    return "".join("G" if str(value).lower() == "good" else "B" for value in state)


def _history_text(history: History) -> str:
    if not history:
        return "No actions have been taken."
    return "\n".join(
        f"- {action}: {'no observation' if outcome is None else outcome}" for action, outcome in history
    )


def _rock_marginals(model: RockDiagnosisModel, belief: np.ndarray) -> list[float]:
    return [
        float(
            sum(
                probability
                for state, probability in zip(model.hidden_states, belief)
                if str(state[rock_id]).lower() == "good"
            )
        )
        for rock_id in range(model.num_rocks)
    ]


class LLMRockStrategyProvider:
    """Generate strict strategy cells and width orderings with bounded repair."""

    def __init__(self, chat_model: ChatModel, config: L1Config) -> None:
        self.chat_model = chat_model
        self.config = config
        self._strategy_cache: dict[tuple[Any, ...], StrategyCell] = {}
        self._width_cache: dict[tuple[Any, ...], WidthCell] = {}
        self._lock = threading.Lock()
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.logical_strategy_calls = 0
        self.logical_width_calls = 0
        self.cache_hits = 0

    def _strategy_messages(
        self,
        model: RockDiagnosisModel,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        horizon: int,
    ) -> list[dict[str, str]]:
        if self.config.strategy_schema == "branch_policy_v2":
            return self._branch_strategy_messages(
                model,
                position=position,
                belief=belief,
                history=history,
                horizon=horizon,
            )
        system = (
            "You generate compact contingent policies for an exact Rock Diagnosis information task. "
            "Return exactly one JSON object and no prose. A separate program compiles every rule, "
            "enumerates all observation branches, and selects the strategy with highest exact total EIG."
        )
        schema = (
            '{"strategies":[{"name":"short name","description":"plain-language plan and why future '
            'positions help","rules":[{"when":[{"kind":"step_at_least","value":1}],'
            '"action":{"kind":"check_rock","rock_id":1}},{"when":[],"action":'
            '{"kind":"target_rock","rock_id":1,"path":"x_first"}}]}]}'
        )
        predicate_help = (
            "Allowed predicates: step_at_least/value, step_at_most/value, "
            "last_observation/rock_id/outcome, ever_observed/rock_id/outcome, "
            "good_probability_at_least/rock_id/threshold, good_probability_at_most/rock_id/threshold, "
            "distance_at_most/rock_id/distance, at_rock/rock_id. Each rule's when is an AND-list. "
            "Rules are first-match and only the final fallback has when:[]."
        )
        syntax_warning = (
            'Predicate objects always put the predicate name in kind, for example '
            '{"kind":"at_rock","rock_id":0}; never use {"at_rock":{...}}. '
            'Observation outcomes are exactly the lowercase strings "good" and "bad"; never use G/B.'
        )
        action_help = (
            "Allowed actions: target_rock/rock_id/path (x_first or y_first), check_rock/rock_id, "
            "or move/direction (NORTH, EAST, SOUTH, WEST). target_rock moves one legal step toward "
            "the rock and checks it when reached. check_rock is legal from EVERY grid position: it is a "
            "remote sensor whose accuracy decreases with distance, so an immediate check must use "
            "check_rock directly rather than first requiring at_rock. Direct moves that become illegal "
            "on any rollout branch invalidate the whole response."
        )
        posterior_lines = [
            f"- {_state_label(state)}: {float(probability):.8f}"
            for state, probability in sorted(
                zip(model.hidden_states, belief), key=lambda item: -float(item[1])
            )
        ]
        user_lines = [
            f"Return exactly {self.config.num_strategies} distinct strategies using this schema:",
            schema,
            predicate_help,
            syntax_warning,
            action_help,
            f"Planning horizon: {horizon} action(s).",
            f"Grid side length: {model.map_spec.grid_size}.",
            f"Current rover position: {position}.",
            f"Rock coordinates by ID: {list(enumerate(model.map_spec.rock_positions))}.",
            f"Current marginal P(good) by rock ID: {[round(value, 8) for value in _rock_marginals(model, belief)]}.",
            "Exact full posterior over rock vectors:",
            *posterior_lines,
            "History:",
            _history_text(history),
            "Legal root actions:",
            ", ".join(model.legal_actions(position)),
            "Make the strategies behaviorally diverse. Include movement when it can improve the next check, "
            "and encode the future check explicitly with a step or observation rule. At horizon 2, the cell "
            "must include at least one strategy whose action AT THE CURRENT STATE is a move and at least one "
            "whose action AT THE CURRENT STATE is a direct check. Guarantee the check root by giving one "
            "strategy an unconditional final fallback like "
            '{"when":[],"action":{"kind":"check_rock","rock_id":1}}. Its name or an at_rock rule does '
            "not make it a check root if that rule does not currently match.",
        ]
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}]

    def _branch_strategy_messages(
        self,
        model: RockDiagnosisModel,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        horizon: int,
    ) -> list[dict[str, str]]:
        legal_roots = model.legal_actions(position)
        required_move_count = _required_move_policy_count(
            model,
            position,
            horizon=horizon,
            total_count=self.config.num_strategies,
        )
        required_check_count = self.config.num_strategies - required_move_count
        menus: dict[str, dict[str, list[str]]] = {}
        root_geometry: dict[str, dict[str, Any]] = {}
        for root_action in legal_roots:
            child_position = model.next_position(position, root_action)
            child_legal = list(model.legal_actions(child_position))
            menus[root_action] = {
                outcome: child_legal
                for outcome in _branch_outcome_keys(model, root_action, horizon)
            }
            root_geometry[root_action] = {
                "child_position": list(child_position),
                "child_manhattan_distance_by_rock_id": [
                    abs(child_position[0] - rock_position[0])
                    + abs(child_position[1] - rock_position[1])
                    for rock_position in model.map_spec.rock_positions
                ],
            }
        system = (
            "You design short contingent policies for an exact Rock Diagnosis information task. "
            "A separate program validates every action, enumerates every observation branch, and "
            "selects the policy with the highest exact total information gain. Return JSON only."
        )
        schema = (
            '{"strategies":[{"name":"short name","description":"why this policy is useful",'
            '"root_action":"ACTION_ID","followups":{}}]}'
            if horizon <= 1
            else (
                '{"strategies":[{"name":"short name","description":"why this policy is useful",'
                '"root_action":"ACTION_ID","followups":{"OUTCOME_KEY":"ACTION_ID"}}]}'
            )
        )
        posterior_lines = [
            f"- {_state_label(state)}: {float(probability):.8f}"
            for state, probability in sorted(
                zip(model.hidden_states, belief), key=lambda item: -float(item[1])
            )
        ]
        instructions = [
            "STRATEGY_SCHEMA=branch_policy_v2",
            f"Return exactly {self.config.num_strategies} behaviorally distinct strategies.",
            f"Schema: {schema}",
            f"Planning horizon: {horizon} action(s).",
            (
                f"CURRENT HORIZON IS 1: choose exactly {self.config.num_strategies} different legal "
                "root_action IDs. Because followups are empty, repeating a root is a duplicate and is "
                "invalid. Every strategy must use followups:{} exactly, with no outcome keys or future "
                "actions, even for movement roots."
                if horizon <= 1
                else "CURRENT HORIZON IS 2: every strategy must provide the exact branch followups below."
            ),
            "root_action must be one listed root action ID.",
            (
                'At horizon 2 a movement root uses exactly {"none":"FOLLOWUP_ID"}; a check root '
                'uses exactly {"good":"FOLLOWUP_ID","bad":"FOLLOWUP_ID"}. Never use "none" '
                "for a check root and never omit either good or bad."
            ),
            (
                'Every followup value is a JSON string action ID, for example "good":"check-2". '
                'Never nest an object such as "good":{"check-2":"none"}.'
            ),
            (
                "For horizon 2, followups must contain exactly the outcome keys shown for that root "
                "and each value must be chosen from that branch's legal-action menu. For horizon 1, "
                "followups must be {}."
            ),
            (
                f"At horizon 2 include exactly {required_move_count} movement-root strategies with "
                f"distinct root_action values and exactly {required_check_count} direct-check-root "
                "strategies. Every movement-root strategy must use a direct check action as its none "
                "followup. Use the description to explain the information-seeking logic, but do not add fields."
            ),
            (
                "Behaviorally distinct means no two strategies may repeat the same root_action plus "
                "the same followup action(s), even if their names or descriptions differ."
            ),
            f"Grid side length: {model.map_spec.grid_size}.",
            f"Current rover position: {position}.",
            f"Rock coordinates by ID: {list(enumerate(model.map_spec.rock_positions))}.",
            (
                "Coordinate convention: EAST increases the first coordinate, WEST decreases it, "
                "SOUTH increases the second coordinate, and NORTH decreases it. Movement produces "
                "no information immediately. A check is legal remotely from every position, but its "
                "accuracy decreases exponentially with Manhattan distance to that rock. A useful "
                "movement policy should therefore follow with a check whose distance the move reduced."
            ),
            f"Current marginal P(good) by rock ID: {[round(value, 8) for value in _rock_marginals(model, belief)]}.",
            "Exact full posterior over rock vectors:",
            *posterior_lines,
            "History:",
            _history_text(history),
            "ROOT_GEOMETRY=" + json.dumps(root_geometry, sort_keys=True, separators=(",", ":")),
            "MACHINE_READABLE_MENUS=" + json.dumps(menus, sort_keys=True, separators=(",", ":")),
        ]
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(instructions)}]

    def _width_messages(
        self,
        model: RockDiagnosisModel,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
    ) -> list[dict[str, str]]:
        legal = model.legal_actions(position)
        system = (
            "You provide a full legal-action ordering for a myopic Rock Diagnosis width control. "
            "Return exactly one JSON object and no prose."
        )
        user = "\n".join(
            [
                f"Current position: {position}.",
                f"Rock coordinates: {list(enumerate(model.map_spec.rock_positions))}.",
                f"Marginal P(good): {[round(value, 8) for value in _rock_marginals(model, belief)]}.",
                "History:",
                _history_text(history),
                "Legal action IDs:",
                ", ".join(legal),
                f'Return all {len(legal)} legal IDs exactly once: {{"action_ids":[...]}}',
            ]
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _complete_with_one_retry(
        self,
        *,
        messages: list[dict[str, str]],
        request_type: str,
        context: dict[str, Any],
        parser: Any,
    ) -> tuple[Any, str]:
        last_error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            responses = self.chat_model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise StrategyProposalError("model did not return exactly one response")
            response = responses[0]
            try:
                value = parser(response)
            except (StrategyProposalError, RockStrategyExecutionError) as exc:
                last_error = exc
                with self._lock:
                    self.invalid_responses.append(
                        {**context, "request_type": request_type, "attempt": attempt, "error": str(exc), "raw_response": response}
                    )
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                f"The previous response is invalid: {exc}. Correct the entire cell using the "
                                "registered schema. Return exactly one JSON object and no prose."
                            ),
                        },
                    ]
                continue
            with self._lock:
                self.physical_requests.append(
                    {**context, "request_type": request_type, "attempt": attempt, "raw_response": response}
                )
            return value, response
        raise StrategyProposalError(
            f"{request_type} cell failed after {self.config.validation_retries + 1} attempts: {last_error}"
        )

    def propose_strategies(
        self,
        model: RockDiagnosisModel,
        *,
        map_name: str,
        trial_index: int,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        horizon: int,
    ) -> StrategyCell:
        key = (map_name, trial_index, position, history, horizon)
        with self._lock:
            self.logical_strategy_calls += 1
            cached = self._strategy_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return StrategyCell(cached.strategies, cached.exact_scores, cached.raw_response, True)

        def parse_and_validate(
            response: str,
        ) -> tuple[tuple[RockStrategy, ...], tuple[ExactRockStrategyScore, ...]]:
            if self.config.strategy_schema == "branch_policy_v2":
                strategies = parse_branch_strategy_cell(
                    response,
                    model=model,
                    position=position,
                    horizon=horizon,
                    expected_count=self.config.num_strategies,
                )
            else:
                strategies = parse_strategy_cell(
                    response, model=model, expected_count=self.config.num_strategies
                )
            scores = tuple(
                score_rock_strategy_exact(
                    model,
                    strategy,
                    position=position,
                    belief=belief,
                    history=history,
                    horizon=horizon,
                )
                for strategy in strategies
            )
            if horizon > 1 and self.config.strategy_schema == "branch_policy_v2":
                roots = [str(score.root_action) for score in scores]
                required_moves = _required_move_policy_count(
                    model,
                    position,
                    horizon=horizon,
                    total_count=self.config.num_strategies,
                )
                move_indices = [index for index, action in enumerate(roots) if model.is_move(action)]
                check_indices = [
                    index for index, action in enumerate(roots) if model.check_id(action) is not None
                ]
                branch_payloads = [json.loads(strategy.raw_text) for strategy in strategies]
                move_roots = {roots[index] for index in move_indices}
                move_then_check = all(
                    str(branch_payloads[index]["followups"]["none"]).startswith("check-")
                    for index in move_indices
                )
                if (
                    len(move_indices) != required_moves
                    or len(move_roots) != required_moves
                    or len(check_indices) != self.config.num_strategies - required_moves
                    or not move_then_check
                ):
                    raise StrategyProposalError(
                        f"horizon-2 branch cells need exactly {required_moves} distinct movement roots "
                        f"whose none followup is a check and {self.config.num_strategies - required_moves} "
                        f"check roots; compiled root actions were {roots}"
                    )
            elif horizon > 1:
                roots = [str(score.root_action) for score in scores]
                if not any(model.is_move(action) for action in roots) or not any(
                    model.check_id(action) is not None for action in roots
                ):
                    raise StrategyProposalError(
                        "horizon-2 strategy cells need at least one move root and at least one check root; "
                        f"compiled root actions were {roots}. Make one strategy's currently matching action "
                        "a direct check_rock (an unconditional check fallback guarantees this) and keep "
                        "another strategy's current root as movement"
                    )
            return strategies, scores

        context = {
            "map_name": map_name,
            "trial_index": trial_index,
            "position": list(position),
            "history": _serialize_history(history),
            "horizon": horizon,
        }
        validated, raw_response = self._complete_with_one_retry(
            messages=self._strategy_messages(
                model, position=position, belief=belief, history=history, horizon=horizon
            ),
            request_type="strategy",
            context=context,
            parser=parse_and_validate,
        )
        strategies, exact_scores = validated
        cell = StrategyCell(strategies, exact_scores, raw_response, False)
        with self._lock:
            self._strategy_cache[key] = cell
        return cell

    def propose_width_order(
        self,
        model: RockDiagnosisModel,
        *,
        map_name: str,
        trial_index: int,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
    ) -> WidthCell:
        key = (map_name, trial_index, position, history)
        with self._lock:
            self.logical_width_calls += 1
            cached = self._width_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return WidthCell(cached.action_ids, cached.raw_response, True)
        legal = model.legal_actions(position)
        context = {
            "map_name": map_name,
            "trial_index": trial_index,
            "position": list(position),
            "history": _serialize_history(history),
        }
        action_ids, raw_response = self._complete_with_one_retry(
            messages=self._width_messages(model, position=position, belief=belief, history=history),
            request_type="width",
            context=context,
            parser=lambda response: parse_width_cell(response, allowed_actions=legal),
        )
        cell = WidthCell(action_ids, raw_response, False)
        with self._lock:
            self._width_cache[key] = cell
        return cell


class DeterministicStrategyModel:
    """No-spend model that emits complete valid cells for mechanics tests."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic strategy model supports one response")
        content = messages[-1]["content"]
        if "Legal action IDs:\n" in content:
            legal = content.split("Legal action IDs:\n", maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
            preferred = ["move-EAST", "check-1", "check-2", "check-0", "move-NORTH", "move-SOUTH", "move-WEST"]
            ordered = [action for action in preferred if action in legal]
            ordered.extend(action for action in legal if action not in ordered)
            return [json.dumps({"action_ids": ordered})]
        if "STRATEGY_SCHEMA=branch_policy_v2" in content:
            count = int(content.split("Return exactly ", maxsplit=1)[1].split(" behaviorally", maxsplit=1)[0])
            horizon = int(content.split("Planning horizon: ", maxsplit=1)[1].split(" action", maxsplit=1)[0])
            menus = json.loads(content.split("MACHINE_READABLE_MENUS=", maxsplit=1)[1].split("\n", maxsplit=1)[0])
            roots = list(menus)
            moves = [action for action in roots if action.startswith("move-")]
            checks = [action for action in roots if action.startswith("check-")]
            move_count = min(len(moves), max(1, count // 2)) if horizon > 1 else 0
            if horizon > 1:
                ordered_roots = [
                    *moves[:move_count],
                    *(checks[index % len(checks)] for index in range(count - move_count)),
                ]
            else:
                ordered_roots = roots[:count]
            strategies: list[dict[str, Any]] = []
            signatures: set[tuple[Any, ...]] = set()
            for offset in range(1000):
                if len(strategies) >= len(ordered_roots):
                    break
                root = ordered_roots[len(strategies)]
                outcomes = list(menus[root])
                followups: dict[str, str] = {}
                for branch_index, outcome in enumerate(outcomes):
                    choices = menus[root][outcome]
                    if root.startswith("move-"):
                        checks = [action for action in choices if action.startswith("check-")]
                        followups[outcome] = checks[offset % len(checks)] if checks else choices[0]
                    else:
                        followups[outcome] = choices[(offset + branch_index) % len(choices)]
                signature = (root, *(followups[outcome] for outcome in outcomes))
                if signature in signatures:
                    ordered_roots.append(root)
                    continue
                signatures.add(signature)
                strategies.append(
                    {
                        "name": f"deterministic-branch-{len(strategies)}",
                        "description": "A complete legal branch policy for deterministic mechanics testing.",
                        "root_action": root,
                        "followups": followups if horizon > 1 else {},
                    }
                )
                if len(strategies) == count:
                    return [json.dumps({"strategies": strategies})]
            raise AssertionError("deterministic branch generator could not fill the requested cell")
        count = int(content.split("Return exactly ", maxsplit=1)[1].split(" distinct strategies", maxsplit=1)[0])
        coordinates_text = content.split("Rock coordinates by ID: ", maxsplit=1)[1].split(".\n", maxsplit=1)[0]
        num_rocks = coordinates_text.count("),") + 1
        strategies: list[dict[str, Any]] = []
        for index in range(count):
            rock_id = index % num_rocks
            path = "x_first" if index % 2 == 0 else "y_first"
            if index == count - 1:
                strategies.append(
                    {
                        "name": f"immediate-check-{rock_id}",
                        "description": f"Check rock {rock_id} immediately and continue checking it.",
                        "rules": [
                            {"when": [], "action": {"kind": "check_rock", "rock_id": rock_id}}
                        ],
                    }
                )
                continue
            strategies.append(
                {
                    "name": f"approach-then-check-{rock_id}-{path}",
                    "description": f"Approach rock {rock_id} along a {path} path, then check it on the next step.",
                    "rules": [
                        {
                            "when": [{"kind": "step_at_least", "value": 1}],
                            "action": {"kind": "check_rock", "rock_id": rock_id},
                        },
                        {
                            "when": [],
                            "action": {"kind": "target_rock", "rock_id": rock_id, "path": path},
                        },
                    ],
                }
            )
        return [json.dumps({"strategies": strategies})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


def _choose_index(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _exhaustive_action_values(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    horizon: int,
) -> tuple[dict[str, float], int]:
    legal = model.legal_actions(position)
    scorer_units = 0
    values: dict[str, float] = {}
    for action in legal:
        scorer_units += 1
        value = model.expected_information_gain(position, belief, action)
        if horizon > 1:
            next_position = model.next_position(position, action)
            continuation = 0.0
            for outcome in model.outcomes(action):
                probability = model.outcome_probability(position, belief, action, outcome)
                if probability <= EPSILON:
                    continue
                posterior = model.posterior(position, belief, action, outcome)
                future_legal = model.legal_actions(next_position)
                future_scores = [
                    model.expected_information_gain(next_position, posterior, future_action)
                    for future_action in future_legal
                ]
                scorer_units += len(future_scores)
                continuation += probability * max(future_scores)
            value += continuation
        values[action] = value
    return values, scorer_units


def _anchor_value(model: RockDiagnosisModel, state: PolicyState, horizon: int) -> float:
    values, _units = _exhaustive_action_values(
        model, position=state.position, belief=state.belief, horizon=horizon
    )
    return max(values.values())


def _strategy_selection(
    model: RockDiagnosisModel,
    provider: LLMRockStrategyProvider,
    state: PolicyState,
    *,
    map_name: str,
    trial_index: int,
    horizon: int,
) -> Selection:
    cell = provider.propose_strategies(
        model,
        map_name=map_name,
        trial_index=trial_index,
        position=state.position,
        belief=state.belief,
        history=state.history,
        horizon=horizon,
    )
    scores = list(cell.exact_scores)
    index = _choose_index([score.eig for score in scores])
    selected = scores[index]
    exhaustive_value = _anchor_value(model, state, horizon)
    fraction = selected.eig / exhaustive_value if exhaustive_value > EPSILON else 1.0
    return Selection(
        action=str(selected.root_action),
        planning_score=selected.eig,
        immediate_eig=model.expected_information_gain(state.position, state.belief, str(selected.root_action)),
        candidate_roots=tuple(str(score.root_action) for score in scores),
        candidate_scores=tuple(score.eig for score in scores),
        selected_strategy=cell.strategies[index].raw_text,
        candidate_strategies=tuple(strategy.raw_text for strategy in cell.strategies),
        scorer_units=sum(score.expanded_decision_nodes for score in scores),
        distinct_scored_candidates=len(scores),
        exhaustive_value=exhaustive_value,
        exhaustive_fraction=fraction,
        logical_llm_calls=1,
    )


def _shared_d1_selection(
    model: RockDiagnosisModel,
    provider: LLMRockStrategyProvider,
    state: PolicyState,
    *,
    map_name: str,
    trial_index: int,
    horizon: int,
) -> Selection:
    cell = provider.propose_strategies(
        model,
        map_name=map_name,
        trial_index=trial_index,
        position=state.position,
        belief=state.belief,
        history=state.history,
        horizon=horizon,
    )
    executor = RockStrategyExecutor(model)
    roots = [
        executor.choose_action(
            strategy,
            position=state.position,
            belief=state.belief,
            history=state.history,
            strategy_step=0,
        )
        for strategy in cell.strategies
    ]
    scores = [model.expected_information_gain(state.position, state.belief, action) for action in roots]
    index = _choose_index(scores)
    exhaustive_value = _anchor_value(model, state, horizon)
    return Selection(
        action=roots[index],
        planning_score=scores[index],
        immediate_eig=scores[index],
        candidate_roots=tuple(roots),
        candidate_scores=tuple(scores),
        selected_strategy=cell.strategies[index].raw_text,
        candidate_strategies=tuple(strategy.raw_text for strategy in cell.strategies),
        scorer_units=len(scores),
        distinct_scored_candidates=len(set(roots)),
        exhaustive_value=exhaustive_value,
        exhaustive_fraction=scores[index] / exhaustive_value if exhaustive_value > EPSILON else 1.0,
        logical_llm_calls=1,
    )


def _random_branch_strategies(
    model: RockDiagnosisModel,
    rng: np.random.Generator,
    *,
    position: tuple[int, int],
    horizon: int,
    count: int,
) -> tuple[RockStrategy, ...]:
    legal_roots = model.legal_actions(position)
    move_roots = tuple(action for action in legal_roots if model.is_move(action))
    check_roots = tuple(action for action in legal_roots if model.check_id(action) is not None)
    if horizon > 1 and (not move_roots or not check_roots):
        raise AssertionError("branch-strategy root-mix control requires legal move and check actions")
    required_roots: list[str] = []
    if horizon > 1:
        move_count = _required_move_policy_count(
            model,
            position,
            horizon=horizon,
            total_count=count,
        )
        selected_move_indices = rng.choice(len(move_roots), size=move_count, replace=False)
        required_roots.extend(move_roots[int(index)] for index in selected_move_indices)
        required_roots.extend(
            check_roots[int(rng.integers(len(check_roots)))] for _ in range(count - move_count)
        )
    policies: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    for attempt in range(10_000):
        if len(policies) < len(required_roots):
            root_action = required_roots[len(policies)]
        else:
            root_action = legal_roots[int(rng.integers(len(legal_roots)))]
        outcome_keys = _branch_outcome_keys(model, root_action, horizon)
        child_legal = model.legal_actions(model.next_position(position, root_action))
        if model.is_move(root_action):
            child_choices = tuple(action for action in child_legal if model.check_id(action) is not None)
        else:
            child_choices = child_legal
        followups = {
            outcome: child_choices[int(rng.integers(len(child_choices)))] for outcome in outcome_keys
        }
        signature = (root_action, *(followups[outcome] for outcome in outcome_keys))
        if signature in signatures:
            continue
        signatures.add(signature)
        policies.append(
            {
                "name": f"random-branch-{len(policies)}",
                "description": "A policy sampled uniformly from the legal branch-policy grammar.",
                "root_action": root_action,
                "followups": followups,
            }
        )
        if len(policies) == count:
            response = json.dumps({"strategies": policies}, separators=(",", ":"))
            return parse_branch_strategy_cell(
                response,
                model=model,
                position=position,
                horizon=horizon,
                expected_count=count,
            )
    raise AssertionError("random branch-strategy sampler could not fill a distinct cell")


def _random_strategy_selection(
    model: RockDiagnosisModel,
    state: PolicyState,
    config: L1Config,
    *,
    map_name: str,
    trial_index: int,
    round_index: int,
    horizon: int,
) -> Selection:
    rng = np.random.default_rng(_stable_seed(config.seed, map_name, "random-strategy", trial_index, round_index))
    if config.strategy_schema == "branch_policy_v2":
        strategies = list(
            _random_branch_strategies(
                model,
                rng,
                position=state.position,
                horizon=horizon,
                count=config.num_strategies,
            )
        )
        scores = [
            score_rock_strategy_exact(
                model,
                strategy,
                position=state.position,
                belief=state.belief,
                history=state.history,
                horizon=horizon,
            )
            for strategy in strategies
        ]
    else:
        strategies = []
        scores = []
        for _cell_attempt in range(100):
            strategies = [
                parse_rock_strategy(random_rock_strategy_text(model, rng, index=index), model)
                for index in range(config.num_strategies)
            ]
            scores = [
                score_rock_strategy_exact(
                    model,
                    strategy,
                    position=state.position,
                    belief=state.belief,
                    history=state.history,
                    horizon=horizon,
                )
                for strategy in strategies
            ]
            if horizon <= 1:
                break
            roots = [str(score.root_action) for score in scores]
            if any(model.is_move(action) for action in roots) and any(
                model.check_id(action) is not None for action in roots
            ):
                break
        else:
            raise AssertionError("random-strategy rejection sampler could not produce a mixed root cell")
    index = _choose_index([score.eig for score in scores])
    selected = scores[index]
    exhaustive_value = _anchor_value(model, state, horizon)
    return Selection(
        action=str(selected.root_action),
        planning_score=selected.eig,
        immediate_eig=model.expected_information_gain(state.position, state.belief, str(selected.root_action)),
        candidate_roots=tuple(str(score.root_action) for score in scores),
        candidate_scores=tuple(score.eig for score in scores),
        selected_strategy=strategies[index].raw_text,
        candidate_strategies=tuple(strategy.raw_text for strategy in strategies),
        scorer_units=sum(score.expanded_decision_nodes for score in scores),
        distinct_scored_candidates=len(scores),
        exhaustive_value=exhaustive_value,
        exhaustive_fraction=selected.eig / exhaustive_value if exhaustive_value > EPSILON else 1.0,
        logical_llm_calls=0,
    )


def _width_selection(
    model: RockDiagnosisModel,
    provider: LLMRockStrategyProvider,
    state: PolicyState,
    *,
    map_name: str,
    trial_index: int,
    horizon: int,
    scorer_budget: int,
) -> Selection:
    if scorer_budget <= 0:
        raise ValueError("width scorer budget must be positive")
    cell = provider.propose_width_order(
        model,
        map_name=map_name,
        trial_index=trial_index,
        position=state.position,
        belief=state.belief,
        history=state.history,
    )
    distinct_count = min(scorer_budget, len(cell.action_ids))
    candidates = cell.action_ids[:distinct_count]
    score_by_action: dict[str, float] = {}
    evaluated: list[float] = []
    for evaluation_index in range(scorer_budget):
        action = cell.action_ids[evaluation_index % len(cell.action_ids)]
        score = model.expected_information_gain(state.position, state.belief, action)
        score_by_action.setdefault(action, score)
        if evaluation_index < distinct_count:
            evaluated.append(score)
    index = _choose_index(evaluated)
    action = candidates[index]
    exhaustive_value = _anchor_value(model, state, horizon)
    return Selection(
        action=action,
        planning_score=evaluated[index],
        immediate_eig=evaluated[index],
        candidate_roots=candidates,
        candidate_scores=tuple(evaluated),
        selected_strategy=None,
        candidate_strategies=(),
        scorer_units=scorer_budget,
        distinct_scored_candidates=distinct_count,
        exhaustive_value=exhaustive_value,
        exhaustive_fraction=evaluated[index] / exhaustive_value if exhaustive_value > EPSILON else 1.0,
        logical_llm_calls=1,
    )


def _exhaustive_selection(model: RockDiagnosisModel, state: PolicyState, *, horizon: int) -> Selection:
    values, scorer_units = _exhaustive_action_values(
        model, position=state.position, belief=state.belief, horizon=horizon
    )
    candidates = tuple(values)
    scores = [values[action] for action in candidates]
    index = _choose_index(scores)
    action = candidates[index]
    return Selection(
        action=action,
        planning_score=scores[index],
        immediate_eig=model.expected_information_gain(state.position, state.belief, action),
        candidate_roots=candidates,
        candidate_scores=tuple(scores),
        selected_strategy=None,
        candidate_strategies=(),
        scorer_units=scorer_units,
        distinct_scored_candidates=len(candidates),
        exhaustive_value=scores[index],
        exhaustive_fraction=1.0,
        logical_llm_calls=0,
    )


def _observe(
    model: RockDiagnosisModel,
    state: PolicyState,
    *,
    truth_index: int,
    seed: int,
    map_name: str,
    trial_index: int,
    action: str,
) -> str | None:
    check_id = model.check_id(action)
    if check_id is None:
        return None
    key = (state.position, check_id)
    repeat_index = state.check_counts.get(key, 0)
    state.check_counts[key] = repeat_index + 1
    probability_good = float(model.likelihood_vector(state.position, action, "good")[truth_index])
    return (
        "good"
        if _uniform(seed, map_name, "observation", trial_index, state.position, check_id, repeat_index)
        < probability_good
        else "bad"
    )


def _serialize_history(history: History) -> list[dict[str, Any]]:
    return [{"action": action, "observation": outcome} for action, outcome in history]


def _apply_selection(
    model: RockDiagnosisModel,
    state: PolicyState,
    selection: Selection,
    *,
    arm: ArmName,
    truth_index: int,
    config: L1Config,
    map_name: str,
    trial_index: int,
    round_index: int,
) -> None:
    if selection.action not in model.legal_actions(state.position):
        raise AssertionError(f"selected illegal action {selection.action} for arm {arm}")
    outcome = _observe(
        model,
        state,
        truth_index=truth_index,
        seed=config.seed,
        map_name=map_name,
        trial_index=trial_index,
        action=selection.action,
    )
    entropy_before = model.entropy(state.belief)
    posterior = model.posterior(state.position, state.belief, selection.action, outcome)
    state.steps.append(
        {
            "round": round_index,
            "position_before": list(state.position),
            "action": selection.action,
            "observation": outcome,
            "entropy_before": entropy_before,
            "entropy_after": model.entropy(posterior),
            "realized_entropy_drop": entropy_before - model.entropy(posterior),
            "map_correct": float(model.decode_map_index(posterior) == truth_index),
            "truth_log_probability": float(
                math.log(max(float(posterior[truth_index]), np.finfo(float).tiny))
            ),
            **asdict(selection),
        }
    )
    state.history = state.history + ((selection.action, outcome),)
    state.position = model.next_position(state.position, selection.action)
    state.belief = posterior


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means.append(np.mean(values[indices], axis=1))
    lower, upper = np.quantile(np.concatenate(means), [0.025, 0.975])
    return float(lower), float(upper)


def _arm_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    entropy = np.asarray([[step["entropy_after"] for step in trace["steps"]] for trace in traces])
    map_accuracy = np.asarray([[step["map_correct"] for step in trace["steps"]] for trace in traces])
    truth_log = np.asarray([[step["truth_log_probability"] for step in trace["steps"]] for trace in traces])
    return {
        "num_trials": len(traces),
        "entropy_auc_mean": float(np.mean(entropy)),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "final_map_accuracy_mean": float(np.mean(map_accuracy[:, -1])),
        "final_truth_log_probability_mean": float(np.mean(truth_log[:, -1])),
        "round_entropy_mean": [float(value) for value in np.mean(entropy, axis=0)],
        "round_map_accuracy_mean": [float(value) for value in np.mean(map_accuracy, axis=0)],
        "round_truth_log_probability_mean": [float(value) for value in np.mean(truth_log, axis=0)],
        "mean_scorer_units_per_decision": float(
            np.mean([step["scorer_units"] for trace in traces for step in trace["steps"]])
        ),
        "mean_logical_llm_calls_per_decision": float(
            np.mean([step["logical_llm_calls"] for trace in traces for step in trace["steps"]])
        ),
        "mean_exhaustive_fraction": float(
            np.mean([step["exhaustive_fraction"] for trace in traces for step in trace["steps"]])
        ),
    }


def _paired_comparison(
    strategy_traces: list[dict[str, Any]],
    baseline_traces: list[dict[str, Any]],
    *,
    config: L1Config,
    label: str,
) -> dict[str, Any]:
    strategy_entropy_traces = np.asarray(
        [[step["entropy_after"] for step in trace["steps"]] for trace in strategy_traces]
    )
    baseline_entropy_traces = np.asarray(
        [[step["entropy_after"] for step in trace["steps"]] for trace in baseline_traces]
    )
    strategy_entropy = strategy_entropy_traces[:, -1]
    baseline_entropy = baseline_entropy_traces[:, -1]
    entropy_gain = baseline_entropy - strategy_entropy
    entropy_auc_gain = np.mean(baseline_entropy_traces, axis=1) - np.mean(
        strategy_entropy_traces, axis=1
    )
    strategy_truth_traces = np.asarray(
        [[step["truth_log_probability"] for step in trace["steps"]] for trace in strategy_traces]
    )
    baseline_truth_traces = np.asarray(
        [[step["truth_log_probability"] for step in trace["steps"]] for trace in baseline_traces]
    )
    strategy_truth = strategy_truth_traces[:, -1]
    baseline_truth = baseline_truth_traces[:, -1]
    truth_auc_gain = np.mean(strategy_truth_traces, axis=1) - np.mean(
        baseline_truth_traces, axis=1
    )
    strategy_map = np.asarray([trace["steps"][-1]["map_correct"] for trace in strategy_traces])
    baseline_map = np.asarray([trace["steps"][-1]["map_correct"] for trace in baseline_traces])
    ci = _bootstrap_mean_ci(
        entropy_gain,
        seed=_stable_seed(config.seed, "l1-bootstrap", label),
        replicates=config.bootstrap_replicates,
    )
    entropy_auc_ci = _bootstrap_mean_ci(
        entropy_auc_gain,
        seed=_stable_seed(config.seed, "l1-bootstrap", label, "entropy-auc"),
        replicates=config.bootstrap_replicates,
    )
    truth_auc_ci = _bootstrap_mean_ci(
        truth_auc_gain,
        seed=_stable_seed(config.seed, "l1-bootstrap", label, "truth-log-auc"),
        replicates=config.bootstrap_replicates,
    )
    return {
        "entropy_auc_gain_mean": float(np.mean(entropy_auc_gain)),
        "entropy_auc_gain_ci95": [entropy_auc_ci[0], entropy_auc_ci[1]],
        "truth_log_probability_auc_gain_mean": float(np.mean(truth_auc_gain)),
        "truth_log_probability_auc_gain_ci95": [truth_auc_ci[0], truth_auc_ci[1]],
        "final_entropy_gain_mean": float(np.mean(entropy_gain)),
        "final_entropy_gain_ci95": [ci[0], ci[1]],
        "final_truth_log_probability_gain_mean": float(np.mean(strategy_truth - baseline_truth)),
        "final_map_accuracy_gain_mean": float(np.mean(strategy_map - baseline_map)),
        "wins_ties_losses": [
            int(np.count_nonzero(entropy_gain > 0.0)),
            int(np.count_nonzero(entropy_gain == 0.0)),
            int(np.count_nonzero(entropy_gain < 0.0)),
        ],
        "entropy_auc_wins_ties_losses": [
            int(np.count_nonzero(entropy_auc_gain > 0.0)),
            int(np.count_nonzero(entropy_auc_gain == 0.0)),
            int(np.count_nonzero(entropy_auc_gain < 0.0)),
        ],
        "paired_values": [float(value) for value in entropy_gain],
        "entropy_auc_paired_values": [float(value) for value in entropy_auc_gain],
        "truth_log_probability_auc_paired_values": [float(value) for value in truth_auc_gain],
    }


def _trace_payload(
    *,
    map_name: str,
    trial_index: int,
    truth_index: int,
    arm: ArmName,
    state: PolicyState,
) -> dict[str, Any]:
    return {
        "map_name": map_name,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "arm": arm,
        "final_position": list(state.position),
        "steps": state.steps,
    }


def _run_l1_trial(
    provider: LLMRockStrategyProvider,
    config: L1Config,
    *,
    map_name: str,
    trial_index: int,
) -> dict[str, Any]:
    model = RockDiagnosisModel(get_paper_map(map_name))
    truth_rng = np.random.default_rng(_stable_seed(config.seed, map_name, "truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    states = {
        arm: PolicyState(model.initial_belief.copy(), model.map_spec.start_position) for arm in ARMS
    }
    initial_strategy_texts: dict[ArmName, tuple[str, ...]] = {}
    all_actions_legal = True
    width_calls_match = True
    width_scorer_units_match = True
    random_cells_complete = True

    for round_index in range(config.num_rounds):
        horizon = min(config.planning_horizon, config.num_rounds - round_index)
        strategy_selection = _strategy_selection(
            model,
            provider,
            states["strategy_eig"],
            map_name=map_name,
            trial_index=trial_index,
            horizon=horizon,
        )
        selections: dict[ArmName, Selection] = {
            "strategy_eig": strategy_selection,
            "exhaustive_d2": _exhaustive_selection(
                model, states["exhaustive_d2"], horizon=horizon
            ),
            "shared_d1": _shared_d1_selection(
                model,
                provider,
                states["shared_d1"],
                map_name=map_name,
                trial_index=trial_index,
                horizon=horizon,
            ),
            "width": _width_selection(
                model,
                provider,
                states["width"],
                map_name=map_name,
                trial_index=trial_index,
                horizon=horizon,
                scorer_budget=strategy_selection.scorer_units,
            ),
            "random_strategy": _random_strategy_selection(
                model,
                states["random_strategy"],
                config,
                map_name=map_name,
                trial_index=trial_index,
                round_index=round_index,
                horizon=horizon,
            ),
        }
        if round_index == 0:
            initial_strategy_texts["strategy_eig"] = strategy_selection.candidate_strategies
            initial_strategy_texts["shared_d1"] = selections["shared_d1"].candidate_strategies
        width_calls_match = width_calls_match and (
            selections["width"].logical_llm_calls == strategy_selection.logical_llm_calls
        )
        width_scorer_units_match = width_scorer_units_match and (
            selections["width"].scorer_units == strategy_selection.scorer_units
        )
        random_cells_complete = random_cells_complete and (
            len(selections["random_strategy"].candidate_strategies) == config.num_strategies
        )
        for arm, selection in selections.items():
            all_actions_legal = all_actions_legal and selection.action in model.legal_actions(
                states[arm].position
            )
            _apply_selection(
                model,
                states[arm],
                selection,
                arm=arm,
                truth_index=truth_index,
                config=config,
                map_name=map_name,
                trial_index=trial_index,
                round_index=round_index,
            )

    return {
        "map_name": map_name,
        "trial_index": trial_index,
        "traces": {
            arm: _trace_payload(
                map_name=map_name,
                trial_index=trial_index,
                truth_index=truth_index,
                arm=arm,
                state=states[arm],
            )
            for arm in ARMS
        },
        "all_actions_legal": all_actions_legal,
        "initial_strategy_cells_shared": (
            initial_strategy_texts["strategy_eig"] == initial_strategy_texts["shared_d1"]
        ),
        "width_calls_match": width_calls_match,
        "width_scorer_units_match": width_scorer_units_match,
        "random_cells_complete": random_cells_complete,
    }


def run_l1_anchor(provider: LLMRockStrategyProvider, config: L1Config) -> dict[str, Any]:
    config.validate()
    traces: dict[str, dict[ArmName, list[dict[str, Any]]]] = {
        map_name: {arm: [] for arm in ARMS} for map_name in config.map_names
    }
    all_actions_legal = True
    initial_strategy_cells_shared = True
    width_calls_match = True
    width_scorer_units_match = True
    random_cells_complete = True

    jobs = [
        (map_name, trial_index)
        for map_name in config.map_names
        for trial_index in range(config.num_trials_per_map)
    ]
    worker_count = min(config.trial_concurrency, len(jobs))
    if worker_count == 1:
        trial_results = [
            _run_l1_trial(
                provider, config, map_name=map_name, trial_index=trial_index
            )
            for map_name, trial_index in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _run_l1_trial,
                    provider,
                    config,
                    map_name=map_name,
                    trial_index=trial_index,
                )
                for map_name, trial_index in jobs
            ]
            trial_results = [future.result() for future in futures]

    for trial_result in trial_results:
        map_name = str(trial_result["map_name"])
        for arm in ARMS:
            traces[map_name][arm].append(trial_result["traces"][arm])
        all_actions_legal = all_actions_legal and bool(trial_result["all_actions_legal"])
        initial_strategy_cells_shared = initial_strategy_cells_shared and bool(
            trial_result["initial_strategy_cells_shared"]
        )
        width_calls_match = width_calls_match and bool(trial_result["width_calls_match"])
        width_scorer_units_match = width_scorer_units_match and bool(
            trial_result["width_scorer_units_match"]
        )
        random_cells_complete = random_cells_complete and bool(
            trial_result["random_cells_complete"]
        )

    map_results: dict[str, Any] = {}
    required_baselines: tuple[ArmName, ...] = ("shared_d1", "width", "random_strategy")
    for map_name in config.map_names:
        comparisons = {
            f"strategy_eig_minus_{baseline}": _paired_comparison(
                traces[map_name]["strategy_eig"],
                traces[map_name][baseline],
                config=config,
                label=f"{map_name}-{baseline}",
            )
            for baseline in (*required_baselines, "exhaustive_d2")
        }
        map_results[map_name] = {
            "summary": {arm: _arm_summary(traces[map_name][arm]) for arm in ARMS},
            "paired": comparisons,
            "gate_passed": all(
                comparisons[f"strategy_eig_minus_{baseline}"][
                    "entropy_auc_gain_ci95"
                    if config.primary_endpoint == "entropy_auc"
                    else "final_entropy_gain_ci95"
                ][0]
                > 0.0
                for baseline in required_baselines
            ),
        }

    mechanics = {
        "terminal_cell_failures": 0,
        "raw_rejected_responses": len(provider.invalid_responses),
        "all_selected_actions_legal": all_actions_legal,
        "initial_strategy_cells_shared_with_d1": initial_strategy_cells_shared,
        "width_logical_llm_calls_match_strategy_eig": width_calls_match,
        "width_exact_scorer_units_match_strategy_eig": width_scorer_units_match,
        "random_strategy_cells_have_k_candidates": random_cells_complete,
        "rollout_scoring_llm_calls": 0,
        "physical_llm_requests": len(provider.physical_requests) + len(provider.invalid_responses),
        "accepted_llm_cells": len(provider.physical_requests),
        "logical_strategy_requests": provider.logical_strategy_calls,
        "logical_width_requests": provider.logical_width_calls,
        "provider_cache_hits": provider.cache_hits,
    }
    return {
        "schema_version": 1,
        "stage": "L1",
        "config": asdict(config),
        "maps": map_results,
        "mechanics": mechanics,
        "gate_passed": all(result["gate_passed"] for result in map_results.values())
        and all(
            bool(value)
            for key, value in mechanics.items()
            if key
            in {
                "all_selected_actions_legal",
                "initial_strategy_cells_shared_with_d1",
                "width_logical_llm_calls_match_strategy_eig",
                "width_exact_scorer_units_match_strategy_eig",
                "random_strategy_cells_have_k_candidates",
            }
        ),
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
        "traces": traces,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Rock Strategy-Prior L1 Anchor",
        "",
        "Strategy text is generated by the LLM; all rollout scoring, posteriors, observations, and decoding are exact.",
        "",
    ]
    for map_name, result in summary["maps"].items():
        lines.extend(
            [
                f"## Map {map_name}",
                "",
                "| Arm | Entropy AUC | Final entropy | Final MAP | Final truth log p | Exact units / decision | LLM calls / decision | Exhaustive fraction |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm in ARMS:
            row = result["summary"][arm]
            lines.append(
                f"| {arm} | {row['entropy_auc_mean']:.4f} | {row['final_entropy_mean']:.4f} | "
                f"{row['final_map_accuracy_mean']:.4f} | {row['final_truth_log_probability_mean']:.4f} | "
                f"{row['mean_scorer_units_per_decision']:.2f} | {row['mean_logical_llm_calls_per_decision']:.2f} | "
                f"{row['mean_exhaustive_fraction']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"Positive paired gains favor StrategyEIG. Registered primary endpoint: `{summary['config']['primary_endpoint']}`.",
                "",
                "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | Final entropy gain [95% CI] | AUC W / T / L |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for label, comparison in result["paired"].items():
            auc_ci = comparison["entropy_auc_gain_ci95"]
            truth_ci = comparison["truth_log_probability_auc_gain_ci95"]
            final_ci = comparison["final_entropy_gain_ci95"]
            wtl = comparison["entropy_auc_wins_ties_losses"]
            lines.append(
                f"| {label} | {comparison['entropy_auc_gain_mean']:+.4f} "
                f"[{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}] | "
                f"{comparison['truth_log_probability_auc_gain_mean']:+.4f} "
                f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | "
                f"{comparison['final_entropy_gain_mean']:+.4f} "
                f"[{final_ci[0]:+.4f}, {final_ci[1]:+.4f}] | "
                f"{wtl[0]} / {wtl[1]} / {wtl[2]} |"
            )
        lines.extend(["", f"Map gate passed: `{result['gate_passed']}`.", ""])
    lines.extend(["## Mechanics", ""])
    for key, value in summary["mechanics"].items():
        lines.append(f"- {key}: `{value}`.")
    lines.extend(["", f"**L1 gate passed: `{summary['gate_passed']}`.**", ""])
    return "\n".join(lines)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_rock_strategy_l1_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/rock_strategy_l1/20260716"),
    )
    parser.add_argument("--run-id", default="nonmyopic-rock-strategy-l1-20260716")
    parser.add_argument("--num-trials-per-map", type=int, default=30)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--num-strategies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12_032)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=32)
    parser.add_argument(
        "--strategy-schema",
        choices=("reactive_rules_v1", "branch_policy_v2"),
        default="reactive_rules_v1",
    )
    parser.add_argument(
        "--primary-endpoint",
        choices=("final_entropy", "entropy_auc"),
        default="final_entropy",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = L1Config(
        num_trials_per_map=args.num_trials_per_map,
        num_rounds=args.num_rounds,
        num_strategies=args.num_strategies,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
        strategy_schema=args.strategy_schema,
        primary_endpoint=args.primary_endpoint,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicStrategyModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        if not runtime_config.model_pairs:
            raise ValueError("L1 config requires a questioner model pair")
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = LLMRockStrategyProvider(chat_model, config)
    try:
        summary = run_l1_anchor(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "L1",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "L1_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    summary["usage"] = _usage_snapshot(chat_model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "L1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "L1.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"gate_passed": summary["gate_passed"], "mechanics": summary["mechanics"]}, indent=2))


if __name__ == "__main__":
    main()
