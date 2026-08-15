"""Exact zero-call policy ladder for compositional NeuronBench worlds."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np


MECHANISM_NAMES = (
    "z_rebound",
    "h_sag",
    "na_fatigue",
    "ca_rebound",
    "d_type",
    "textbook_M",
)
TIE_TOLERANCE = 1e-12
PRACTICAL_TIE_TOLERANCE = 1e-9


def candidate_masks() -> tuple[int, ...]:
    singles = tuple(1 << index for index in range(len(MECHANISM_NAMES)))
    pairs = tuple(
        (1 << left) | (1 << right)
        for left in range(len(MECHANISM_NAMES))
        for right in range(left + 1, len(MECHANISM_NAMES))
    )
    return (0,) + singles + pairs


def truth_masks() -> tuple[int, ...]:
    return candidate_masks()[1 + len(MECHANISM_NAMES) :]


def mask_name(mask: int) -> str:
    if mask == 0:
        return "plain"
    return "+".join(
        name for index, name in enumerate(MECHANISM_NAMES) if mask & (1 << index)
    )


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("log weights must be a finite nonempty vector")
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        raise FloatingPointError("weights have no finite mass")
    return weights / total


@dataclass(frozen=True)
class CompositionalState:
    history: tuple[tuple[int, int], ...]
    support: tuple[int, ...]
    truth_weights: tuple[float, ...]

    def weights(self) -> np.ndarray:
        values = np.asarray(self.truth_weights, dtype=float)
        if (
            values.ndim != 1
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0)
        ):
            raise FloatingPointError("truth weights are not normalized")
        return values


class CompositionalBank:
    """Response bank and deterministic dynamic-support transition model."""

    def __init__(
        self,
        action_counts: np.ndarray,
        query_counts: np.ndarray,
        action_names: Sequence[str],
        query_names: Sequence[str],
        *,
        observation_sd: float = 1.0,
    ) -> None:
        actions = np.asarray(action_counts, dtype=float)
        queries = np.asarray(query_counts, dtype=float)
        masks = candidate_masks()
        truths = truth_masks()
        if actions.ndim != 2 or actions.shape[0] != len(masks):
            raise ValueError("action counts must have shape (22, actions)")
        if queries.ndim != 2 or queries.shape[0] != len(masks) or queries.shape[1] == 0:
            raise ValueError("query counts must have shape (22, nonempty queries)")
        if actions.shape[1] != len(action_names) or queries.shape[1] != len(query_names):
            raise ValueError("response arrays and names do not align")
        if not np.isfinite(actions).all() or not np.isfinite(queries).all():
            raise ValueError("response arrays must be finite")
        if observation_sd <= 0 or not math.isfinite(observation_sd):
            raise ValueError("observation_sd must be positive and finite")
        self.action_counts = actions
        self.query_counts = queries
        self.action_names = tuple(str(item) for item in action_names)
        self.query_names = tuple(str(item) for item in query_names)
        self.masks = masks
        self.truths = truths
        self.mask_to_index = {mask: index for index, mask in enumerate(masks)}
        self.truth_indices = tuple(self.mask_to_index[mask] for mask in truths)
        self.observation_sd = float(observation_sd)

    @property
    def num_actions(self) -> int:
        return self.action_counts.shape[1]

    def _history_log_likelihood(self, mask: int, history: Sequence[tuple[int, int]]) -> float:
        model = self.mask_to_index[mask]
        scale = -0.5 / (self.observation_sd**2)
        return float(
            sum(
                scale * (float(observation) - self.action_counts[model, action]) ** 2
                for action, observation in history
            )
        )

    def inference_weights(self, state: CompositionalState) -> np.ndarray:
        scores = np.asarray(
            [self._history_log_likelihood(mask, state.history) for mask in state.support],
            dtype=float,
        )
        return _normalize_log_weights(scores)

    def forecast(self, state: CompositionalState) -> np.ndarray:
        models = np.asarray([self.mask_to_index[mask] for mask in state.support], dtype=int)
        return self.inference_weights(state) @ self.query_counts[models]

    def truth_loss(self, state: CompositionalState, truth_mask: int) -> float:
        truth_index = self.mask_to_index[truth_mask]
        return float(np.mean((self.forecast(state) - self.query_counts[truth_index]) ** 2))

    def leaf_risk(self, state: CompositionalState) -> float:
        losses = np.asarray([self.truth_loss(state, mask) for mask in self.truths], dtype=float)
        return float(state.weights() @ losses)

    def initial_state(self, support: Sequence[int] = (0,)) -> CompositionalState:
        discovered = tuple(sorted(set(int(mask) for mask in support)))
        if not discovered or any(mask not in self.mask_to_index for mask in discovered):
            raise ValueError("initial support is invalid")
        count = len(self.truths)
        return CompositionalState(
            history=(),
            support=discovered,
            truth_weights=tuple(1.0 / count for _ in range(count)),
        )

    def proposal_candidates(self, support: Sequence[int]) -> tuple[int, ...]:
        discovered = set(int(mask) for mask in support)
        proposals: set[int] = set()
        for mask in discovered:
            for bit in range(len(MECHANISM_NAMES)):
                candidate = mask | (1 << bit)
                if (
                    candidate != mask
                    and candidate.bit_count() <= 2
                    and candidate in self.mask_to_index
                    and candidate not in discovered
                ):
                    proposals.add(candidate)
        return tuple(sorted(proposals))

    def oracle_proposal(
        self, support: Sequence[int], history: Sequence[tuple[int, int]]
    ) -> int | None:
        candidates = self.proposal_candidates(support)
        if not candidates:
            return None
        best = candidates[0]
        best_score = self._history_log_likelihood(best, history)
        for candidate in candidates[1:]:
            score = self._history_log_likelihood(candidate, history)
            if score > best_score + TIE_TOLERANCE:
                best, best_score = candidate, score
        return best

    def _updated_truth_weights(
        self, state: CompositionalState, action: int, observation: int
    ) -> tuple[float, ...]:
        truth_models = np.asarray(self.truth_indices, dtype=int)
        residual = float(observation) - self.action_counts[truth_models, action]
        log_prior = np.log(np.maximum(state.weights(), np.finfo(float).tiny))
        posterior = _normalize_log_weights(
            log_prior - 0.5 * residual**2 / (self.observation_sd**2)
        )
        return tuple(float(value) for value in posterior)

    def transition(
        self,
        state: CompositionalState,
        action: int,
        observation: int,
        *,
        proposal_mode: str = "oracle",
    ) -> CompositionalState:
        if action < 0 or action >= self.num_actions or not math.isfinite(float(observation)):
            raise ValueError("transition action or observation is invalid")
        history = state.history + ((int(action), int(observation)),)
        support = state.support
        if proposal_mode == "oracle":
            proposal = self.oracle_proposal(support, history)
        elif proposal_mode == "history_blind":
            candidates = self.proposal_candidates(support)
            proposal = candidates[len(state.history) % len(candidates)] if candidates else None
        elif proposal_mode == "fixed":
            proposal = None
        else:
            raise ValueError(f"unknown proposal mode: {proposal_mode}")
        if proposal is not None:
            support = tuple(sorted(support + (proposal,)))
        return CompositionalState(
            history=history,
            support=support,
            truth_weights=self._updated_truth_weights(state, action, observation),
        )

    def branches(
        self, state: CompositionalState, action: int
    ) -> tuple[tuple[int, float], ...]:
        grouped: dict[int, float] = {}
        for weight, model in zip(state.weights(), self.truth_indices):
            observation = int(round(float(self.action_counts[model, action])))
            grouped[observation] = grouped.get(observation, 0.0) + float(weight)
        branches = tuple(sorted((outcome, probability) for outcome, probability in grouped.items()))
        if not math.isclose(sum(probability for _, probability in branches), 1.0, abs_tol=1e-12):
            raise FloatingPointError("branch probabilities are not normalized")
        return branches


class CompositionalPlanner:
    """Finite-budget policy improvement with an explicit dynamic-support transition."""

    def __init__(self, bank: CompositionalBank, *, proposal_mode: str = "oracle") -> None:
        if proposal_mode not in {"oracle", "history_blind", "fixed"}:
            raise ValueError("invalid proposal mode")
        self.bank = bank
        self.proposal_mode = proposal_mode

    @lru_cache(maxsize=None)
    def transition(
        self, state: CompositionalState, action: int, observation: int
    ) -> CompositionalState:
        return self.bank.transition(
            state, action, observation, proposal_mode=self.proposal_mode
        )

    @lru_cache(maxsize=None)
    def action_value(
        self,
        state: CompositionalState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
    ) -> float:
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for observation, probability in self.bank.branches(state, action):
            child = self.transition(state, action, observation)
            if level == 1 or remaining == 1:
                child_value = self.bank.leaf_risk(child)
            else:
                child_value = self.policy_value(child, remainder, remaining - 1, level - 1)
            value += probability * child_value
        return value

    @lru_cache(maxsize=None)
    def policy_action(
        self,
        state: CompositionalState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        if remaining <= 0 or not available:
            return -1
        best_action = -1
        best_value = math.inf
        for action in available:
            value = self.action_value(state, available, remaining, level, action)
            if value < best_value - TIE_TOLERANCE:
                best_action, best_value = action, value
        if best_action < 0:
            raise AssertionError("policy failed to select an action")
        return best_action

    @lru_cache(maxsize=None)
    def policy_value(
        self,
        state: CompositionalState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.bank.leaf_risk(state)
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        return sum(
            probability
            * self.policy_value(
                self.transition(state, action, observation),
                remainder,
                remaining - 1,
                level,
            )
            for observation, probability in self.bank.branches(state, action)
        )

    def replay_truth(
        self,
        truth_mask: int,
        level: int,
        *,
        execution_budget: int = 4,
        initial_support: Sequence[int] = (0,),
    ) -> tuple[float, tuple[int, ...], CompositionalState]:
        state = self.bank.initial_state(initial_support)
        available = tuple(range(self.bank.num_actions))
        actions: list[int] = []
        for remaining in range(execution_budget, 0, -1):
            action = self.policy_action(state, available, remaining, level)
            actions.append(action)
            truth_index = self.bank.mask_to_index[truth_mask]
            observation = int(round(float(self.bank.action_counts[truth_index, action])))
            state = self.transition(state, action, observation)
            available = tuple(item for item in available if item != action)
        return self.bank.truth_loss(state, truth_mask), tuple(actions), state

    def _branch_replay(
        self,
        state: CompositionalState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        truth_weight_pairs: tuple[tuple[int, float], ...],
    ) -> float:
        if remaining <= 0 or not available:
            return sum(
                weight * self.bank.truth_loss(state, truth_mask)
                for truth_mask, weight in truth_weight_pairs
            )
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        groups: dict[int, list[tuple[int, float]]] = {}
        for truth_mask, weight in truth_weight_pairs:
            model = self.bank.mask_to_index[truth_mask]
            observation = int(round(float(self.bank.action_counts[model, action])))
            groups.setdefault(observation, []).append((truth_mask, weight))
        return sum(
            self._branch_replay(
                self.transition(state, action, observation),
                remainder,
                remaining - 1,
                level,
                tuple(items),
            )
            for observation, items in sorted(groups.items())
        )

    def reachable_states(
        self,
        level: int,
        *,
        execution_budget: int = 4,
        initial_support: Sequence[int] = (0,),
    ) -> tuple[CompositionalState, ...]:
        root = self.bank.initial_state(initial_support)
        frontier: dict[tuple[tuple[int, int], ...], CompositionalState] = {(): root}
        reached: dict[tuple[tuple[int, int], ...], CompositionalState] = {}
        for remaining in range(execution_budget, 0, -1):
            next_frontier: dict[tuple[tuple[int, int], ...], CompositionalState] = {}
            for state in frontier.values():
                reached[state.history] = state
                available = tuple(
                    action
                    for action in range(self.bank.num_actions)
                    if action not in {item[0] for item in state.history}
                )
                action = self.policy_action(state, available, remaining, level)
                for observation, _ in self.bank.branches(state, action):
                    child = self.transition(state, action, observation)
                    next_frontier[child.history] = child
            frontier = next_frontier
        return tuple(reached[key] for key in sorted(reached))

    def evaluate(
        self,
        level: int,
        *,
        execution_budget: int = 4,
        initial_support: Sequence[int] = (0,),
    ) -> dict[str, Any]:
        root = self.bank.initial_state(initial_support)
        available = tuple(range(self.bank.num_actions))
        root_action = self.policy_action(root, available, execution_budget, level)
        truth_results = [
            self.replay_truth(
                truth_mask,
                level,
                execution_budget=execution_budget,
                initial_support=initial_support,
            )
            for truth_mask in self.bank.truths
        ]
        truth_losses = [item[0] for item in truth_results]
        prior = 1.0 / len(self.bank.truths)
        planned_replay = self._branch_replay(
            root,
            available,
            execution_budget,
            level,
            tuple((mask, prior) for mask in self.bank.truths),
        )
        expected = float(np.mean(truth_losses))
        return {
            "policy_level": level,
            "root_action_index": root_action,
            "root_action": self.bank.action_names[root_action],
            "root_surrogate_value": self.action_value(
                root, available, execution_budget, level, root_action
            ),
            "planned_value": float(planned_replay),
            "expected_terminal_mse": expected,
            "truth_losses": [float(value) for value in truth_losses],
            "truth_actions": [list(item[1]) for item in truth_results],
            "final_supports": [list(item[2].support) for item in truth_results],
            "reachable_histories": len(self.reachable_states(
                level,
                execution_budget=execution_budget,
                initial_support=initial_support,
            )),
        }


def policy_divergence(
    bank: CompositionalBank,
    lower: CompositionalPlanner,
    lower_level: int,
    upper: CompositionalPlanner,
    upper_level: int,
    *,
    execution_budget: int = 4,
) -> dict[str, Any]:
    states: dict[tuple[tuple[int, int], ...], CompositionalState] = {}
    for planner, level in ((lower, lower_level), (upper, upper_level)):
        for state in planner.reachable_states(level, execution_budget=execution_budget):
            states.setdefault(state.history, state)
    changed = 0
    comparable = 0
    rows = []
    for history, state in sorted(states.items()):
        remaining = execution_budget - len(history)
        if remaining <= 0:
            continue
        available = tuple(
            action
            for action in range(bank.num_actions)
            if action not in {item[0] for item in history}
        )
        lower_action = lower.policy_action(state, available, remaining, lower_level)
        upper_action = upper.policy_action(state, available, remaining, upper_level)
        different = lower_action != upper_action
        comparable += 1
        changed += int(different)
        rows.append(
            {
                "history": [list(item) for item in history],
                "lower_action": lower_action,
                "upper_action": upper_action,
                "different": different,
            }
        )
    root_changed = bool(rows and rows[0]["different"] and rows[0]["history"] == [])
    return {
        "root_changed": root_changed,
        "changed": changed,
        "comparable": comparable,
        "fraction_changed": changed / comparable if comparable else 0.0,
        "rows": rows,
    }


def deterministic_random_action(
    state: CompositionalState, available: Sequence[int], seed: int
) -> int:
    payload = f"{seed}|{state.history}|{state.support}".encode()
    index = int(hashlib.sha256(payload).hexdigest()[:16], 16) % len(available)
    return int(tuple(available)[index])


def compare_losses(
    baseline: Sequence[float], challenger: Sequence[float], tolerance: float = PRACTICAL_TIE_TOLERANCE
) -> dict[str, float | int]:
    before = np.asarray(baseline, dtype=float)
    after = np.asarray(challenger, dtype=float)
    if before.shape != after.shape or before.ndim != 1 or before.size == 0:
        raise ValueError("loss vectors must be aligned and nonempty")
    difference = before - after
    return {
        "baseline_mean": float(before.mean()),
        "challenger_mean": float(after.mean()),
        "relative_reduction": float((before.mean() - after.mean()) / before.mean())
        if before.mean() > 0
        else 0.0,
        "wins": int(np.sum(difference > tolerance)),
        "ties": int(np.sum(np.abs(difference) <= tolerance)),
        "losses": int(np.sum(difference < -tolerance)),
    }
