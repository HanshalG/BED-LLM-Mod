"""Cost-aware policy improvement for stochastic compositional ChemBench."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from .compositional import AuditedCompositionalPolicyPlanner
from .mechanics import SpeculativeState, TIE_TOLERANCE


@dataclass(frozen=True)
class CostedAction:
    name: str
    base_name: str
    base_index: int
    repeats: int

    @property
    def cost(self) -> int:
        return self.repeats


class CostedCompositionalPolicyPlanner(AuditedCompositionalPolicyPlanner):
    """Receding policy ladder whose actions consume a shared well budget."""

    def __init__(
        self,
        *args: Any,
        actions: Sequence[CostedAction],
        well_budget: int,
        cost_aware: bool = True,
        shortlist_per_objective: int = 4,
        shortlist_cap: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.actions = tuple(actions)
        if len(self.actions) != self.bank.num_actions:
            raise ValueError("costed actions must match the model bank")
        if well_budget <= 0:
            raise ValueError("well budget must be positive")
        if shortlist_per_objective <= 0 or shortlist_cap <= 0:
            raise ValueError("shortlist sizes must be positive")
        if any(action.repeats not in {1, 2, 4} for action in self.actions):
            raise ValueError("costed repeat count is outside the frozen set")
        identities = {(action.base_index, action.repeats) for action in self.actions}
        if len(identities) != len(self.actions):
            raise ValueError("costed action identities must be unique")
        self.well_budget = int(well_budget)
        self.cost_aware = bool(cost_aware)
        self.shortlist_per_objective = int(shortlist_per_objective)
        self.shortlist_cap = int(shortlist_cap)
        self.last_execution_policy_records: dict[
            tuple[SpeculativeState, tuple[int, ...], int, int], int
        ] = {}
        self.last_scenario_losses: np.ndarray | None = None

    @lru_cache(maxsize=None)
    def transition(
        self, state: SpeculativeState, action: int, outcome: int
    ) -> SpeculativeState:
        return super().transition(state, action, outcome)

    def feasible_actions(
        self, available: tuple[int, ...], remaining_wells: int
    ) -> tuple[int, ...]:
        return tuple(
            action
            for action in available
            if self.actions[action].cost <= remaining_wells
        )

    def remaining_actions(self, available: tuple[int, ...], chosen: int) -> tuple[int, ...]:
        base_index = self.actions[chosen].base_index
        return tuple(
            action
            for action in available
            if self.actions[action].base_index != base_index
        )

    def planning_cost(self, action: int) -> int:
        return self.actions[action].cost if self.cost_aware else 1

    @lru_cache(maxsize=None)
    def one_step_risk(self, state: SpeculativeState, action: int) -> float:
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            value += probability * self.leaf_risk(self.transition(state, action, outcome))
        return value

    @lru_cache(maxsize=None)
    def candidate_actions(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
    ) -> tuple[int, ...]:
        feasible = self.feasible_actions(available, remaining_wells)
        if not feasible:
            return ()
        current = self.leaf_risk(state)
        rows = []
        for action in feasible:
            gain = current - self.one_step_risk(state, action)
            rows.append((action, gain, gain / self.actions[action].cost))
        by_absolute = sorted(rows, key=lambda row: (-row[1], row[0]))
        by_efficiency = sorted(rows, key=lambda row: (-row[2], row[0]))
        ordered = []
        for row in (
            by_absolute[: self.shortlist_per_objective]
            + by_efficiency[: self.shortlist_per_objective]
        ):
            action = row[0]
            if action not in ordered:
                ordered.append(action)
            if len(ordered) == self.shortlist_cap:
                break
        return tuple(ordered)

    @lru_cache(maxsize=None)
    def policy_action_value(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
        action: int,
    ) -> float:
        if self.actions[action].cost > remaining_wells:
            return math.inf
        remainder = self.remaining_actions(available, action)
        planned_remaining = remaining_wells - self.planning_cost(action)
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            child_value = (
                self.leaf_risk(child)
                if level == 1
                or not self.feasible_actions(remainder, max(planned_remaining, 0))
                else self.policy_value(
                    child,
                    remainder,
                    planned_remaining,
                    level - 1,
                )
            )
            value += probability * child_value
        return value

    @lru_cache(maxsize=None)
    def policy_action(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
    ) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        candidates = self.candidate_actions(state, available, remaining_wells)
        if not candidates:
            return -1
        return min(
            (
                self.policy_action_value(
                    state, available, remaining_wells, level, action
                ),
                action,
            )
            for action in candidates
        )[1]

    @lru_cache(maxsize=None)
    def policy_value(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
    ) -> float:
        action = self.policy_action(state, available, remaining_wells, level)
        if action < 0:
            return self.leaf_risk(state)
        remainder = self.remaining_actions(available, action)
        planned_remaining = remaining_wells - self.planning_cost(action)
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            value += probability * self.policy_value(
                child,
                remainder,
                max(planned_remaining, 0),
                level,
            )
        return value

    @lru_cache(maxsize=None)
    def expected_policy_truth_losses(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
    ) -> tuple[float, ...]:
        action = self.policy_action(state, available, remaining_wells, level)
        if action < 0:
            forecast = self.forecast(state)
            targets = self.bank.target_features[np.asarray(self.particle_indices)]
            return tuple(float(value) for value in np.mean((targets - forecast) ** 2, axis=1))
        remainder = self.remaining_actions(available, action)
        actual_remaining = remaining_wells - self.actions[action].cost
        values = np.zeros(len(self.particle_indices), dtype=float)
        truth_likelihoods = self.bank.likelihoods[
            np.asarray(self.particle_indices), action, :
        ]
        for outcome in range(3):
            probabilities = truth_likelihoods[:, outcome]
            if float(np.max(probabilities)) <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            child_values = np.asarray(
                self.expected_policy_truth_losses(
                    child,
                    remainder,
                    actual_remaining,
                    level,
                ),
                dtype=float,
            )
            values += probabilities * child_values
        return tuple(float(value) for value in values)

    def simulate_truth_losses(
        self,
        level: int,
        scenario_uniforms: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        uniforms = np.asarray(scenario_uniforms, dtype=float)
        if (
            uniforms.ndim != 3
            or uniforms.shape[0] != len(self.particle_indices)
            or uniforms.shape[2] < self.well_budget
            or not np.isfinite(uniforms).all()
            or np.any((uniforms < 0.0) | (uniforms >= 1.0))
        ):
            raise ValueError("execution scenario uniforms are invalid")
        losses = np.empty(uniforms.shape[:2], dtype=float)
        maximum_spend = 0
        maximum_decisions = 0
        reused_base_count = 0
        outcome_digest = hashlib.sha256()
        targets = self.bank.target_features[np.asarray(self.particle_indices)]
        policy_records: dict[
            tuple[SpeculativeState, tuple[int, ...], int, int], int
        ] = {}
        for truth_position, truth in enumerate(self.particle_indices):
            for scenario in range(uniforms.shape[1]):
                state = self.initial_state()
                available = tuple(range(self.bank.num_actions))
                remaining = self.well_budget
                used_bases: set[int] = set()
                decisions = 0
                while True:
                    action = self.policy_action(state, available, remaining, level)
                    policy_key = (state, available, remaining, level)
                    previous_action = policy_records.get(policy_key)
                    if previous_action is not None and previous_action != action:
                        raise AssertionError("execution policy is not deterministic")
                    policy_records[policy_key] = action
                    if action < 0:
                        break
                    metadata = self.actions[action]
                    if metadata.cost > remaining:
                        raise AssertionError("selected action exceeds remaining well budget")
                    if metadata.base_index in used_bases:
                        reused_base_count += 1
                        raise AssertionError("selected policy reused a base assay")
                    used_bases.add(metadata.base_index)
                    probabilities = self.bank.likelihoods[truth, action]
                    cumulative = np.cumsum(probabilities)
                    cumulative[-1] = 1.0
                    outcome = int(
                        np.searchsorted(
                            cumulative,
                            uniforms[truth_position, scenario, decisions],
                            side="right",
                        )
                    )
                    outcome_digest.update(
                        np.asarray(
                            (truth_position, scenario, decisions, action, outcome),
                            dtype=np.int64,
                        ).tobytes()
                    )
                    state = self.transition(state, action, outcome)
                    available = self.remaining_actions(available, action)
                    remaining -= metadata.cost
                    decisions += 1
                    if decisions >= uniforms.shape[2]:
                        if self.feasible_actions(available, remaining):
                            raise AssertionError("execution scenarios are too short")
                        break
                maximum_spend = max(maximum_spend, self.well_budget - remaining)
                maximum_decisions = max(maximum_decisions, decisions)
                forecast = self.forecast(state)
                losses[truth_position, scenario] = float(
                    np.mean((forecast - targets[truth_position]) ** 2)
                )
        self.last_execution_policy_records = policy_records
        self.last_scenario_losses = losses.copy()
        return losses, {
            "num_scenarios": int(uniforms.shape[1]),
            "maximum_wells_spent": maximum_spend,
            "maximum_decisions": maximum_decisions,
            "base_assay_reuse_count": reused_base_count,
            "within_budget_no_base_reuse": maximum_spend <= self.well_budget
            and reused_base_count == 0,
            "outcome_action_sha256": outcome_digest.hexdigest(),
        }

    def evaluate_policy_level(
        self,
        level: int,
        *,
        execution_budget: int | None = None,
        scenario_uniforms: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if execution_budget is not None and execution_budget != self.well_budget:
            raise ValueError("execution budget differs from frozen well budget")
        if scenario_uniforms is None:
            raise ValueError("costed policy evaluation requires frozen CRN scenarios")
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        root_action = self.policy_action(state, available, self.well_budget, level)
        root_lookahead_value = self.policy_action_value(
            state,
            available,
            self.well_budget,
            level,
            root_action,
        )
        scenario_losses, execution_audit = self.simulate_truth_losses(
            level, scenario_uniforms
        )
        truth_losses = np.mean(scenario_losses, axis=1)
        expected = float(np.mean(truth_losses))
        scenario_means = np.mean(scenario_losses, axis=0)
        standard_error = float(np.std(scenario_means, ddof=1) / math.sqrt(len(scenario_means)))
        root = self.actions[root_action]
        return {
            "policy_level": int(level),
            "root_action_index": int(root_action),
            "root_action": root.name,
            "root_base_assay": root.base_name,
            "root_repeat_count": root.repeats,
            "root_lookahead_value": float(root_lookahead_value),
            "expected_terminal_mse": expected,
            "expected_terminal_rmsle": math.sqrt(max(expected, 0.0)),
            "scenario_standard_error": standard_error,
            "num_truths": len(self.particle_indices),
            "truth_losses": [float(value) for value in truth_losses],
            "scenario_losses_sha256": hashlib.sha256(
                np.asarray(scenario_losses, dtype=np.float64).tobytes(order="C")
            ).hexdigest(),
            "execution_audit": execution_audit,
        }


class RandomCostedCompositionalPolicyPlanner(CostedCompositionalPolicyPlanner):
    """State-hashed random feasible policy evaluated on frozen CRN scenarios."""

    @lru_cache(maxsize=None)
    def policy_action(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
    ) -> int:
        del level
        feasible = self.feasible_actions(available, remaining_wells)
        if not feasible:
            return -1
        payload = {
            "seed": self.seed,
            "state": state.inference.public_key(),
            "available": list(available),
            "remaining_wells": int(remaining_wells),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        return feasible[int.from_bytes(digest[:8], "big") % len(feasible)]


class TranscriptReplayCostedPolicyPlanner(CostedCompositionalPolicyPlanner):
    """Re-execute CRN trajectories from an immutable exact-policy transcript."""

    def __init__(
        self,
        *args: Any,
        policy_records: Mapping[
            tuple[SpeculativeState, tuple[int, ...], int, int], int
        ],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._replay_policy_records = dict(policy_records)
        self._consumed_policy_records: set[
            tuple[SpeculativeState, tuple[int, ...], int, int]
        ] = set()

    def policy_action(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining_wells: int,
        level: int,
    ) -> int:
        key = (state, available, remaining_wells, level)
        if key not in self._replay_policy_records:
            raise KeyError("execution state is absent from the immutable policy transcript")
        self._consumed_policy_records.add(key)
        return self._replay_policy_records[key]

    @property
    def unused_policy_record_count(self) -> int:
        return len(self._replay_policy_records.keys() - self._consumed_policy_records)
