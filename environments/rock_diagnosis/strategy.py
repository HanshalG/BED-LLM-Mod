"""Compact strategy policies and exact rollout scoring for Rock Diagnosis.

Strategies are JSON documents with a natural-language name/description and an
ordered list of reactive rules.  The structured rules are the executable part of
the strategy; exact scoring never asks an LLM to choose a rollout action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal

import numpy as np

from .core import EPSILON, RockDiagnosisModel


History = tuple[tuple[str, str | None], ...]
PredicateKind = Literal[
    "step_at_least",
    "step_at_most",
    "last_observation",
    "ever_observed",
    "good_probability_at_least",
    "good_probability_at_most",
    "distance_at_most",
    "at_rock",
]
ActionKind = Literal["target_rock", "check_rock", "move"]


class RockStrategyParseError(ValueError):
    """A strategy document is not in the registered grammar."""


class RockStrategyExecutionError(RuntimeError):
    """A parsed strategy cannot produce a legal action in a reached state."""


@dataclass(frozen=True)
class RockStrategyPredicate:
    kind: PredicateKind
    rock_id: int | None = None
    outcome: str | None = None
    value: int | None = None
    threshold: float | None = None
    distance: int | None = None


@dataclass(frozen=True)
class RockStrategyAction:
    kind: ActionKind
    rock_id: int | None = None
    path: Literal["x_first", "y_first"] | None = None
    direction: Literal["NORTH", "EAST", "SOUTH", "WEST"] | None = None


@dataclass(frozen=True)
class RockStrategyRule:
    predicates: tuple[RockStrategyPredicate, ...]
    action: RockStrategyAction


@dataclass(frozen=True)
class RockStrategy:
    name: str
    description: str
    rules: tuple[RockStrategyRule, ...]
    raw_text: str


@dataclass(frozen=True)
class ExactRockStrategyScore:
    eig: float
    start_entropy: float
    expected_final_entropy: float
    root_action: str | None
    expanded_decision_nodes: int
    leaf_nodes: int
    action_counts: dict[str, int]


def _normalize_json_text(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("```json\n"):
        if not normalized.endswith("\n```"):
            raise RockStrategyParseError("strategy has an incomplete JSON fence")
        normalized = normalized[len("```json\n") : -len("\n```")]
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RockStrategyParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        raise RockStrategyParseError(
            f"{context} must contain exactly {sorted(expected)}; got {sorted(payload)}"
        )


def _require_rock_id(value: Any, model: RockDiagnosisModel, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RockStrategyParseError(f"{context} rock_id must be an integer")
    if not 0 <= value < model.num_rocks:
        raise RockStrategyParseError(f"{context} rock_id {value} is out of range")
    return value


def _parse_predicate(payload: Any, model: RockDiagnosisModel) -> RockStrategyPredicate:
    if not isinstance(payload, dict) or not isinstance(payload.get("kind"), str):
        raise RockStrategyParseError("each rule predicate must be an object with a string kind")
    kind = payload["kind"]
    if kind in {"step_at_least", "step_at_most"}:
        _require_exact_keys(payload, {"kind", "value"}, f"predicate {kind}")
        value = payload["value"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RockStrategyParseError(f"predicate {kind} value must be a non-negative integer")
        return RockStrategyPredicate(kind=kind, value=value)
    if kind in {"last_observation", "ever_observed"}:
        _require_exact_keys(payload, {"kind", "rock_id", "outcome"}, f"predicate {kind}")
        rock_id = _require_rock_id(payload["rock_id"], model, f"predicate {kind}")
        outcome = payload["outcome"]
        if outcome not in {"good", "bad"}:
            raise RockStrategyParseError(f"predicate {kind} outcome must be good or bad")
        return RockStrategyPredicate(kind=kind, rock_id=rock_id, outcome=outcome)
    if kind in {"good_probability_at_least", "good_probability_at_most"}:
        _require_exact_keys(payload, {"kind", "rock_id", "threshold"}, f"predicate {kind}")
        rock_id = _require_rock_id(payload["rock_id"], model, f"predicate {kind}")
        threshold = payload["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise RockStrategyParseError(f"predicate {kind} threshold must be numeric")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RockStrategyParseError(f"predicate {kind} threshold must be in [0, 1]")
        return RockStrategyPredicate(kind=kind, rock_id=rock_id, threshold=threshold)
    if kind == "distance_at_most":
        _require_exact_keys(payload, {"kind", "rock_id", "distance"}, "predicate distance_at_most")
        rock_id = _require_rock_id(payload["rock_id"], model, "predicate distance_at_most")
        distance = payload["distance"]
        if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
            raise RockStrategyParseError("predicate distance_at_most distance must be a non-negative integer")
        return RockStrategyPredicate(kind=kind, rock_id=rock_id, distance=distance)
    if kind == "at_rock":
        _require_exact_keys(payload, {"kind", "rock_id"}, "predicate at_rock")
        return RockStrategyPredicate(
            kind=kind,
            rock_id=_require_rock_id(payload["rock_id"], model, "predicate at_rock"),
        )
    raise RockStrategyParseError(f"unknown predicate kind: {kind!r}")


def _parse_action(payload: Any, model: RockDiagnosisModel) -> RockStrategyAction:
    if not isinstance(payload, dict) or not isinstance(payload.get("kind"), str):
        raise RockStrategyParseError("each rule action must be an object with a string kind")
    kind = payload["kind"]
    if kind == "target_rock":
        _require_exact_keys(payload, {"kind", "rock_id", "path"}, "action target_rock")
        rock_id = _require_rock_id(payload["rock_id"], model, "action target_rock")
        path = payload["path"]
        if path not in {"x_first", "y_first"}:
            raise RockStrategyParseError("action target_rock path must be x_first or y_first")
        return RockStrategyAction(kind=kind, rock_id=rock_id, path=path)
    if kind == "check_rock":
        _require_exact_keys(payload, {"kind", "rock_id"}, "action check_rock")
        return RockStrategyAction(
            kind=kind,
            rock_id=_require_rock_id(payload["rock_id"], model, "action check_rock"),
        )
    if kind == "move":
        _require_exact_keys(payload, {"kind", "direction"}, "action move")
        direction = payload["direction"]
        if direction not in {"NORTH", "EAST", "SOUTH", "WEST"}:
            raise RockStrategyParseError("action move direction must be NORTH, EAST, SOUTH, or WEST")
        return RockStrategyAction(kind=kind, direction=direction)
    raise RockStrategyParseError(f"unknown action kind: {kind!r}")


def parse_rock_strategy(text: str, model: RockDiagnosisModel) -> RockStrategy:
    """Parse one strategy strictly; malformed or incomplete policies fail closed."""

    normalized = _normalize_json_text(text)
    try:
        payload = json.loads(normalized, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise RockStrategyParseError("strategy is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RockStrategyParseError("strategy must be a JSON object")
    _require_exact_keys(payload, {"name", "description", "rules"}, "strategy")
    name = payload["name"]
    description = payload["description"]
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise RockStrategyParseError("strategy name must be a non-empty string of at most 80 characters")
    if not isinstance(description, str) or not description.strip() or len(description) > 800:
        raise RockStrategyParseError(
            "strategy description must be a non-empty string of at most 800 characters"
        )
    rules_payload = payload["rules"]
    if not isinstance(rules_payload, list) or not 1 <= len(rules_payload) <= 16:
        raise RockStrategyParseError("strategy rules must be a list containing 1 to 16 rules")
    rules: list[RockStrategyRule] = []
    for index, rule_payload in enumerate(rules_payload):
        if not isinstance(rule_payload, dict):
            raise RockStrategyParseError(f"rule {index} must be an object")
        _require_exact_keys(rule_payload, {"when", "action"}, f"rule {index}")
        when = rule_payload["when"]
        if not isinstance(when, list) or len(when) > 4:
            raise RockStrategyParseError(f"rule {index} when must be a list of at most four predicates")
        predicates = tuple(_parse_predicate(item, model) for item in when)
        if not predicates and index != len(rules_payload) - 1:
            raise RockStrategyParseError("only the final fallback rule may have an empty when list")
        rules.append(RockStrategyRule(predicates=predicates, action=_parse_action(rule_payload["action"], model)))
    if rules[-1].predicates:
        raise RockStrategyParseError("the final rule must be an unconditional fallback with when: []")
    return RockStrategy(name=name.strip(), description=description.strip(), rules=tuple(rules), raw_text=text)


def _rock_good_probability(model: RockDiagnosisModel, belief: np.ndarray, rock_id: int) -> float:
    return float(
        sum(
            probability
            for state, probability in zip(model.hidden_states, belief)
            if str(state[rock_id]).lower() == "good"
        )
    )


def _last_informative_observation(history: History) -> tuple[str, str] | None:
    for action, outcome in reversed(history):
        if outcome is not None:
            return action, outcome
    return None


class RockStrategyExecutor:
    """Compile one parsed reactive strategy into concrete legal Rock actions."""

    def __init__(self, model: RockDiagnosisModel) -> None:
        self.model = model

    def _predicate_matches(
        self,
        predicate: RockStrategyPredicate,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        strategy_step: int,
    ) -> bool:
        if predicate.kind == "step_at_least":
            return strategy_step >= int(predicate.value)
        if predicate.kind == "step_at_most":
            return strategy_step <= int(predicate.value)
        if predicate.kind == "last_observation":
            last = _last_informative_observation(history)
            return last == (f"check-{predicate.rock_id}", predicate.outcome)
        if predicate.kind == "ever_observed":
            return (f"check-{predicate.rock_id}", predicate.outcome) in history
        if predicate.kind in {"good_probability_at_least", "good_probability_at_most"}:
            probability = _rock_good_probability(self.model, belief, int(predicate.rock_id))
            if predicate.kind == "good_probability_at_least":
                return probability >= float(predicate.threshold)
            return probability <= float(predicate.threshold)
        rock_position = self.model.map_spec.rock_positions[int(predicate.rock_id)]
        distance = abs(position[0] - rock_position[0]) + abs(position[1] - rock_position[1])
        if predicate.kind == "distance_at_most":
            return distance <= int(predicate.distance)
        if predicate.kind == "at_rock":
            return distance == 0
        raise AssertionError(f"unhandled predicate kind: {predicate.kind}")

    def _compile_action(self, action: RockStrategyAction, position: tuple[int, int]) -> str:
        if action.kind == "check_rock":
            return f"check-{action.rock_id}"
        if action.kind == "move":
            return f"move-{action.direction}"
        target = self.model.map_spec.rock_positions[int(action.rock_id)]
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if dx == 0 and dy == 0:
            return f"check-{action.rock_id}"
        if action.path == "x_first":
            if dx:
                return "move-EAST" if dx > 0 else "move-WEST"
            return "move-SOUTH" if dy > 0 else "move-NORTH"
        if dy:
            return "move-SOUTH" if dy > 0 else "move-NORTH"
        return "move-EAST" if dx > 0 else "move-WEST"

    def choose_action(
        self,
        strategy: RockStrategy,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        strategy_step: int,
    ) -> str:
        for rule in strategy.rules:
            if all(
                self._predicate_matches(
                    predicate,
                    position=position,
                    belief=belief,
                    history=history,
                    strategy_step=strategy_step,
                )
                for predicate in rule.predicates
            ):
                action = self._compile_action(rule.action, position)
                if action not in self.model.legal_actions(position):
                    raise RockStrategyExecutionError(
                        f"strategy {strategy.name!r} compiled illegal action {action!r} at {position}"
                    )
                return action
        raise RockStrategyExecutionError(f"strategy {strategy.name!r} has no matching fallback rule")


def score_rock_strategy_exact(
    model: RockDiagnosisModel,
    strategy: RockStrategy,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History = (),
    horizon: int,
) -> ExactRockStrategyScore:
    """Return exact total EIG by enumerating every reachable observation branch."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    belief = np.asarray(belief, dtype=float)
    if belief.shape != model.initial_belief.shape or not np.all(np.isfinite(belief)):
        raise ValueError("belief has the wrong shape or contains non-finite values")
    if float(np.sum(belief)) <= EPSILON:
        raise ValueError("belief must have positive mass")
    belief = belief / float(np.sum(belief))
    executor = RockStrategyExecutor(model)

    def recurse(
        node_position: tuple[int, int],
        node_belief: np.ndarray,
        node_history: History,
        step: int,
    ) -> tuple[float, int, int, dict[str, int]]:
        if step >= horizon:
            return model.entropy(node_belief), 0, 1, {}
        action = executor.choose_action(
            strategy,
            position=node_position,
            belief=node_belief,
            history=node_history,
            strategy_step=step,
        )
        next_position = model.next_position(node_position, action)
        expected_entropy = 0.0
        expanded_nodes = 1
        leaf_nodes = 0
        action_counts = {action: 1}
        for outcome in model.outcomes(action):
            probability = model.outcome_probability(node_position, node_belief, action, outcome)
            if probability <= EPSILON:
                continue
            posterior = model.posterior(node_position, node_belief, action, outcome)
            child_entropy, child_nodes, child_leaves, child_counts = recurse(
                next_position,
                posterior,
                node_history + ((action, outcome),),
                step + 1,
            )
            expected_entropy += probability * child_entropy
            expanded_nodes += child_nodes
            leaf_nodes += child_leaves
            for child_action, count in child_counts.items():
                action_counts[child_action] = action_counts.get(child_action, 0) + count
        return expected_entropy, expanded_nodes, leaf_nodes, action_counts

    start_entropy = model.entropy(belief)
    if horizon == 0:
        return ExactRockStrategyScore(0.0, start_entropy, start_entropy, None, 0, 1, {})
    root_action = executor.choose_action(
        strategy,
        position=position,
        belief=belief,
        history=history,
        strategy_step=0,
    )
    final_entropy, expanded_nodes, leaf_nodes, action_counts = recurse(position, belief, history, 0)
    eig = start_entropy - final_entropy
    if eig < -1e-10:
        raise AssertionError(f"exact strategy EIG became negative: {eig}")
    return ExactRockStrategyScore(
        eig=max(0.0, eig),
        start_entropy=start_entropy,
        expected_final_entropy=final_entropy,
        root_action=root_action,
        expanded_decision_nodes=expanded_nodes,
        leaf_nodes=leaf_nodes,
        action_counts=action_counts,
    )


def random_rock_strategy_text(
    model: RockDiagnosisModel,
    rng: np.random.Generator,
    *,
    index: int,
) -> str:
    """Sample a valid policy from the registered grammar for the random control."""

    def random_action() -> dict[str, Any]:
        rock_id = int(rng.integers(model.num_rocks))
        if float(rng.random()) < 0.7:
            return {
                "kind": "target_rock",
                "rock_id": rock_id,
                "path": "x_first" if float(rng.random()) < 0.5 else "y_first",
            }
        return {"kind": "check_rock", "rock_id": rock_id}

    rules: list[dict[str, Any]] = []
    for _ in range(2):
        rock_id = int(rng.integers(model.num_rocks))
        choice = int(rng.integers(4))
        if choice == 0:
            predicate = {
                "kind": "last_observation",
                "rock_id": rock_id,
                "outcome": "good" if float(rng.random()) < 0.5 else "bad",
            }
        elif choice == 1:
            predicate = {
                "kind": "good_probability_at_least",
                "rock_id": rock_id,
                "threshold": round(float(rng.uniform(0.55, 0.85)), 3),
            }
        elif choice == 2:
            predicate = {
                "kind": "distance_at_most",
                "rock_id": rock_id,
                "distance": int(rng.integers(0, 3)),
            }
        else:
            predicate = {"kind": "step_at_least", "value": int(rng.integers(1, 4))}
        rules.append({"when": [predicate], "action": random_action()})
    rules.append({"when": [], "action": random_action()})
    payload = {
        "name": f"random-strategy-{index}",
        "description": "A policy sampled independently from the registered Rock strategy grammar.",
        "rules": rules,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    parse_rock_strategy(text, model)
    return text
