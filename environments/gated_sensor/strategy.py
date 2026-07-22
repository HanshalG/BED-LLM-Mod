"""Strict two-step branch policies and exact scoring for gated diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .model import EPSILON, GatedSensorModel, SensorState


class GatedStrategyParseError(ValueError):
    """An LLM proposal does not satisfy the registered branch grammar."""


@dataclass(frozen=True)
class GatedBranchStrategy:
    name: str
    description: str
    root_action: str
    followups: dict[str, str]
    raw_text: str


@dataclass(frozen=True)
class ExactGatedStrategyScore:
    eig: float
    start_entropy: float
    expected_final_entropy: float
    root_action: str
    expanded_decision_nodes: int
    leaf_nodes: int


def _normalize_json_response(response: str) -> str:
    normalized = response.strip()
    if "```json\n" in normalized:
        start = normalized.rfind("```json\n") + len("```json\n")
        end = normalized.find("\n```", start)
        if end < 0:
            raise GatedStrategyParseError("response has an incomplete JSON fence")
        normalized = normalized[start:end]
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise GatedStrategyParseError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _outcome_keys(model: GatedSensorModel, action: str, horizon: int) -> tuple[str, ...]:
    if horizon <= 1:
        return ()
    return tuple("none" if outcome is None else outcome for outcome in model.outcomes(action))


def parse_gated_strategy_cell(
    response: str,
    *,
    model: GatedSensorModel,
    state: SensorState,
    horizon: int,
    expected_count: int,
    required_activation_roots: tuple[str, ...] = (),
) -> tuple[GatedBranchStrategy, ...]:
    """Parse a complete candidate cell and validate every reachable action."""

    if horizon not in (1, 2):
        raise ValueError("gated branch policies support horizon one or two")
    try:
        payload = json.loads(_normalize_json_response(response), object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise GatedStrategyParseError("strategy response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"strategies"}:
        raise GatedStrategyParseError("response must contain exactly the strategies key")
    items = payload["strategies"]
    if not isinstance(items, list) or len(items) != expected_count:
        raise GatedStrategyParseError(f"expected exactly {expected_count} strategies")
    legal_roots = model.legal_actions(state)
    strategies: list[GatedBranchStrategy] = []
    signatures: set[tuple[Any, ...]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"name", "description", "root_action", "followups"}:
            raise GatedStrategyParseError(f"strategy {index} has incorrect fields")
        name = item["name"]
        description = item["description"]
        root = item["root_action"]
        followups = item["followups"]
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise GatedStrategyParseError(f"strategy {index} name must be 1 to 80 characters")
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise GatedStrategyParseError(f"strategy {index} description must be 1 to 500 characters")
        if not isinstance(root, str) or root not in legal_roots:
            raise GatedStrategyParseError(f"strategy {index} has illegal root_action {root!r}")
        if not isinstance(followups, dict):
            raise GatedStrategyParseError(f"strategy {index} followups must be an object")
        expected_keys = _outcome_keys(model, root, horizon)
        if set(followups) != set(expected_keys):
            raise GatedStrategyParseError(
                f"strategy {index} followups must contain exactly {list(expected_keys)}"
            )
        child_state = model.next_state(state, root)
        child_legal = model.legal_actions(child_state)
        for outcome in expected_keys:
            action = followups[outcome]
            if not isinstance(action, str) or action not in child_legal:
                raise GatedStrategyParseError(
                    f"strategy {index} followup {outcome!r} is illegal: {action!r}"
                )
        if model.action_kind(root) == "activate" and horizon > 1:
            followup = followups["none"]
            if not followup.startswith("precise:"):
                raise GatedStrategyParseError("activation roots must be followed by a precise test")
        signature = (root, *(followups[key] for key in expected_keys))
        if signature in signatures:
            raise GatedStrategyParseError("strategies must be behaviorally distinct")
        signatures.add(signature)
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":"))
        strategies.append(
            GatedBranchStrategy(name.strip(), description.strip(), root, dict(followups), canonical)
        )
    roots = tuple(strategy.root_action for strategy in strategies)
    if roots[: len(required_activation_roots)] != required_activation_roots:
        raise GatedStrategyParseError(
            f"first activation roots must be {list(required_activation_roots)} in order; got {list(roots)}"
        )
    if any(root.startswith("activate:") for root in roots[len(required_activation_roots) :]):
        raise GatedStrategyParseError("non-activation slots must use measurement roots")
    measurement_roots = roots[len(required_activation_roots) :]
    if len(set(measurement_roots)) != len(measurement_roots):
        raise GatedStrategyParseError("measurement-root slots must use distinct root actions")
    return tuple(strategies)


def score_gated_strategy_exact(
    model: GatedSensorModel,
    strategy: GatedBranchStrategy,
    *,
    state: SensorState,
    belief: np.ndarray,
    horizon: int,
) -> ExactGatedStrategyScore:
    """Enumerate a branch policy's exact expected information gain."""

    if horizon not in (1, 2):
        raise ValueError("gated branch policies support horizon one or two")
    start_entropy = model.entropy(belief)
    expected_final_entropy = 0.0
    leaf_nodes = 0
    expanded = 1
    child_state = model.next_state(state, strategy.root_action)
    for outcome in model.outcomes(strategy.root_action):
        probability = model.outcome_probability(belief, strategy.root_action, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(belief, strategy.root_action, outcome)
        if horizon == 1:
            expected_final_entropy += probability * model.entropy(posterior)
            leaf_nodes += 1
            continue
        key = "none" if outcome is None else outcome
        followup = strategy.followups[key]
        expanded += 1
        for final_outcome in model.outcomes(followup):
            conditional_probability = model.outcome_probability(posterior, followup, final_outcome)
            if conditional_probability <= EPSILON:
                continue
            final_posterior = model.posterior(posterior, followup, final_outcome)
            expected_final_entropy += probability * conditional_probability * model.entropy(final_posterior)
            leaf_nodes += 1
    return ExactGatedStrategyScore(
        eig=start_entropy - expected_final_entropy,
        start_entropy=start_entropy,
        expected_final_entropy=expected_final_entropy,
        root_action=strategy.root_action,
        expanded_decision_nodes=expanded,
        leaf_nodes=leaf_nodes,
    )
