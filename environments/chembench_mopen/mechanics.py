"""Zero-call dynamic-support planning mechanics for ChemBench."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


TIE_TOLERANCE = 1e-12
MAX_PROPOSALS = 4


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class DynamicState:
    history: tuple[tuple[int, int], ...]
    discovered: tuple[int, ...]
    live: tuple[int, ...]
    reserve: tuple[int, ...]
    represented_mass: tuple[float, ...]
    represented_weight: tuple[float, ...]
    outside_mass: float

    @property
    def represented_models(self) -> tuple[int, ...]:
        return self.live + self.reserve

    @property
    def represented_weights(self) -> np.ndarray:
        values = np.asarray(self.represented_weight, dtype=float)
        if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise FloatingPointError("conditional represented weights are not normalized")
        return values

    def public_key(self) -> dict[str, Any]:
        return {
            "history": [list(item) for item in self.history],
            "discovered": list(self.discovered),
            "live": list(self.live),
            "reserve": list(self.reserve),
        }


class Proposer(Protocol):
    mode: str

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]: ...


class ModelBank:
    def __init__(
        self,
        likelihoods: np.ndarray,
        target_features: np.ndarray,
        model_names: Sequence[str],
        action_names: Sequence[str],
        action_groups: Sequence[str | None],
        initial_support: Sequence[int],
        *,
        outside_prior: float = 0.35,
        live_cap: int = 12,
        reserve_cap: int = 12,
    ) -> None:
        likelihoods = np.asarray(likelihoods, dtype=float)
        target_features = np.asarray(target_features, dtype=float)
        if likelihoods.ndim != 3 or likelihoods.shape[2] != 3:
            raise ValueError("likelihoods must have shape (models, actions, 3)")
        if target_features.ndim != 2 or target_features.shape[0] != likelihoods.shape[0]:
            raise ValueError("target features must match model count")
        if len(model_names) != likelihoods.shape[0]:
            raise ValueError("model names must match model count")
        if len(action_names) != likelihoods.shape[1] or len(action_groups) != likelihoods.shape[1]:
            raise ValueError("action metadata must match action count")
        if not np.isfinite(likelihoods).all() or not np.isfinite(target_features).all():
            raise ValueError("model bank arrays must be finite")
        if not np.allclose(likelihoods.sum(axis=2), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("likelihood rows must be normalized")
        initial = tuple(dict.fromkeys(int(item) for item in initial_support))
        if not initial or any(item < 0 or item >= likelihoods.shape[0] for item in initial):
            raise ValueError("initial support is invalid")
        if not 0 < outside_prior < 1:
            raise ValueError("outside prior must be between zero and one")
        if live_cap <= 0 or reserve_cap < 0:
            raise ValueError("support caps are invalid")
        self.likelihoods = likelihoods
        self.target_features = target_features
        self.model_names = tuple(model_names)
        self.action_names = tuple(action_names)
        self.action_groups = tuple(action_groups)
        self.initial_support = initial
        self.outside_prior = float(outside_prior)
        self.live_cap = int(live_cap)
        self.reserve_cap = int(reserve_cap)
        initial_features = target_features[np.asarray(initial)]
        self.outside_penalty = float(np.mean(np.var(initial_features, axis=0)))
        if self.outside_penalty <= 0:
            self.outside_penalty = 1e-6

    @property
    def num_models(self) -> int:
        return self.likelihoods.shape[0]

    @property
    def num_actions(self) -> int:
        return self.likelihoods.shape[1]

    def log_likelihood(self, model: int, history: Sequence[tuple[int, int]]) -> float:
        result = 0.0
        for action, outcome in history:
            result += math.log(max(float(self.likelihoods[model, action, outcome]), 1e-300))
        return result

    def state(self, history: Sequence[tuple[int, int]], discovered: Sequence[int]) -> DynamicState:
        history_tuple = tuple((int(action), int(outcome)) for action, outcome in history)
        unique = tuple(sorted(set(int(item) for item in discovered)))
        if not unique or any(item < 0 or item >= self.num_models for item in unique):
            raise ValueError("discovered support is invalid")
        known_scores = np.asarray(
            [
                math.log(1.0 - self.outside_prior)
                - math.log(len(unique))
                + self.log_likelihood(model, history_tuple)
                for model in unique
            ],
            dtype=float,
        )
        ranking = sorted(range(len(unique)), key=lambda index: (-known_scores[index], unique[index]))
        kept = ranking[: self.live_cap + self.reserve_cap]
        kept_models = tuple(unique[index] for index in kept)
        kept_scores = np.asarray([known_scores[index] for index in kept], dtype=float)
        live_count = min(self.live_cap, len(kept_models))
        live = kept_models[:live_count]
        reserve = kept_models[live_count:]
        represented_scores = kept_scores
        outside_score = math.log(self.outside_prior) - len(history_tuple) * math.log(3.0)
        known_normalizer = _logsumexp(represented_scores)
        represented_weight_values = np.exp(represented_scores - known_normalizer)
        difference = outside_score - known_normalizer
        if difference >= 0:
            ratio = math.exp(-difference) if difference < 746.0 else 0.0
            represented_mass = ratio / (1.0 + ratio)
        else:
            ratio = math.exp(difference) if difference > -746.0 else 0.0
            represented_mass = 1.0 / (1.0 + ratio)
        represented_mass = max(represented_mass, np.finfo(float).tiny)
        outside_mass = 1.0 - represented_mass
        represented_joint_mass = tuple(
            float(represented_mass * value) for value in represented_weight_values
        )
        represented_weight = tuple(float(value) for value in represented_weight_values)
        total = sum(represented_joint_mass) + outside_mass
        if not math.isfinite(total) or not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise FloatingPointError("dynamic belief did not normalize")
        return DynamicState(
            history=history_tuple,
            discovered=tuple(sorted(kept_models)),
            live=live,
            reserve=reserve,
            represented_mass=represented_joint_mass,
            represented_weight=represented_weight,
            outside_mass=outside_mass,
        )

    def initial_state(self) -> DynamicState:
        return self.state((), self.initial_support)

    def predictive(self, state: DynamicState, action: int) -> np.ndarray:
        known = np.asarray(state.represented_mass) @ self.likelihoods[
            np.asarray(state.represented_models), action, :
        ]
        result = known + state.outside_mass / 3.0
        result /= result.sum()
        return result

    def leaf_risk(self, state: DynamicState) -> float:
        weights = state.represented_weights
        features = self.target_features[np.asarray(state.represented_models)]
        forecast = weights @ features
        represented = float(np.mean(np.sum(weights[:, None] * (features - forecast) ** 2, axis=0)))
        return represented + state.outside_mass * self.outside_penalty

    def truth_loss(self, state: DynamicState, truth: int) -> float:
        weights = state.represented_weights
        forecast = weights @ self.target_features[np.asarray(state.represented_models)]
        return float(np.mean((forecast - self.target_features[truth]) ** 2))

    def predictive_variance(self, state: DynamicState, action: int) -> float:
        weights = state.represented_weights
        means = self.likelihoods[
            np.asarray(state.represented_models), action, :
        ] @ np.arange(3, dtype=float)
        mean = float(weights @ means)
        return float(weights @ (means - mean) ** 2)

    def transition(
        self,
        state: DynamicState,
        action: int,
        outcome: int,
        proposal: Sequence[int],
    ) -> DynamicState:
        if action < 0 or action >= self.num_actions or outcome not in (0, 1, 2):
            raise ValueError("transition action or outcome is invalid")
        accepted: list[int] = []
        seen = set(state.discovered)
        for item in proposal:
            if not isinstance(item, (int, np.integer)):
                continue
            candidate = int(item)
            if candidate < 0 or candidate >= self.num_models or candidate in seen:
                continue
            accepted.append(candidate)
            seen.add(candidate)
            if len(accepted) == MAX_PROPOSALS:
                break
        history = state.history + ((action, outcome),)
        return self.state(history, state.discovered + tuple(accepted))


class OracleProposer:
    mode = "registry_oracle"

    def __init__(self, bank: ModelBank) -> None:
        self.bank = bank

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        del seed
        history = state.history + ((action, outcome),)
        missing = [item for item in range(self.bank.num_models) if item not in state.discovered]
        missing.sort(key=lambda item: (-self.bank.log_likelihood(item, history), item))
        return tuple(missing[:MAX_PROPOSALS])


class FixedProposer:
    mode = "fixed_support"

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        del state, action, outcome, seed
        return ()


class HistoryBlindProposer:
    mode = "history_blind"

    def __init__(self, candidate_order: Sequence[int]) -> None:
        self.candidate_order = tuple(dict.fromkeys(int(item) for item in candidate_order))

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        del action, outcome, seed
        available = [item for item in self.candidate_order if item not in state.discovered]
        if not available:
            return ()
        offset = (len(state.history) * MAX_PROPOSALS) % len(available)
        ordered = available[offset:] + available[:offset]
        return tuple(ordered[:MAX_PROPOSALS])


class ScriptedResidualProposer:
    mode = "scripted_residual"

    def __init__(self, bank: ModelBank, dictionary: Mapping[tuple[str, int], Sequence[int]]) -> None:
        self.bank = bank
        self.dictionary = {key: tuple(value) for key, value in dictionary.items()}

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        del seed
        group = self.bank.action_groups[action]
        if group is None:
            return ()
        candidates = self.dictionary.get((group, outcome), ())
        return tuple(item for item in candidates if item not in state.discovered)[:MAX_PROPOSALS]


def _proposal_outcome_value(outcome: int | float) -> int | float:
    if isinstance(outcome, (int, np.integer)):
        return int(outcome)
    value = float(outcome)
    if not math.isfinite(value):
        raise ValueError("proposal outcome must be finite")
    return value


def _canonical_public_json(state: Any) -> str:
    attribute = "_proposal_key_canonical_public_json"
    cached = getattr(state, attribute, None)
    if cached is not None:
        return str(cached)
    encoded = json.dumps(state.public_key(), sort_keys=True, separators=(",", ":"))
    try:
        object.__setattr__(state, attribute, encoded)
    except (AttributeError, TypeError):
        pass
    return encoded


def proposal_key(
    mode: str,
    state: DynamicState,
    action: int,
    outcome: int | float,
    seed: int,
) -> str:
    outcome_value = _proposal_outcome_value(outcome)
    payload = (
        '{"action":'
        + str(int(action))
        + ',"mode":'
        + json.dumps(mode, separators=(",", ":"))
        + ',"outcome":'
        + json.dumps(outcome_value, separators=(",", ":"))
        + ',"seed":'
        + str(int(seed))
        + ',"state":'
        + _canonical_public_json(state)
        + "}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class BankedProposer:
    mode = "banked"

    def __init__(self, source_mode: str, records: Mapping[str, Sequence[int]]) -> None:
        self.source_mode = source_mode
        self.records = {key: tuple(int(item) for item in value) for key, value in records.items()}

    def propose(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        key = proposal_key(self.source_mode, state, action, outcome, seed)
        if key not in self.records:
            raise KeyError(f"transition is absent from immutable bank: {key}")
        return self.records[key]


class ProposalCache:
    def __init__(self, proposer: Proposer, *, source_mode: str | None = None) -> None:
        self.proposer = proposer
        self.source_mode = source_mode or proposer.mode
        self._cache: dict[str, tuple[int, ...]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, state: DynamicState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        key = proposal_key(self.source_mode, state, action, outcome, seed)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        proposal = tuple(self.proposer.propose(state, action, outcome, seed))
        self._cache[key] = proposal
        self.misses += 1
        return proposal

    @property
    def records(self) -> dict[str, list[int]]:
        return {key: list(value) for key, value in sorted(self._cache.items())}

    @property
    def frozen_records(self) -> Mapping[str, tuple[int, ...]]:
        """Read-only zero-copy view used by large deterministic replay audits."""
        return MappingProxyType(self._cache)


class DynamicPlanner:
    def __init__(self, bank: ModelBank, proposal_cache: ProposalCache, *, seed: int = 2026081600) -> None:
        self.bank = bank
        self.proposal_cache = proposal_cache
        self.seed = int(seed)

    def _seed(self, state: DynamicState, action: int, outcome: int) -> int:
        key = proposal_key("seed", state, action, outcome, self.seed)
        return int(key[:16], 16) % (2**31 - 1)

    def transition(self, state: DynamicState, action: int, outcome: int) -> DynamicState:
        seed = self._seed(state, action, outcome)
        proposal = self.proposal_cache.get(state, action, outcome, seed)
        return self.bank.transition(state, action, outcome, proposal)

    def candidate_actions(self, state: DynamicState, available: tuple[int, ...], width: int) -> tuple[int, ...]:
        by_group: dict[str, tuple[float, int]] = {}
        for action in available:
            group = self.bank.action_groups[action]
            if group is None:
                continue
            score = self.bank.predictive_variance(state, action)
            current = by_group.get(group)
            if current is None or score > current[0] + TIE_TOLERANCE or (
                abs(score - current[0]) <= TIE_TOLERANCE and action < current[1]
            ):
                by_group[group] = (score, action)
        ranked = sorted(by_group.values(), key=lambda item: (-item[0], item[1]))
        return tuple(action for _, action in ranked[:width])

    @lru_cache(maxsize=None)
    def plan(self, state: DynamicState, available: tuple[int, ...], depth: int, level: int = 0) -> tuple[float, int]:
        if depth <= 0 or not available:
            return self.bank.leaf_risk(state), -1
        width = 6 if level == 0 else 3 if level == 1 else 2
        candidates = self.candidate_actions(state, available, width)
        if not candidates:
            return self.bank.leaf_risk(state), -1
        best_value = math.inf
        best_action = -1
        for action in candidates:
            probabilities = self.bank.predictive(state, action)
            remainder = tuple(item for item in available if item != action)
            value = 0.0
            for outcome, probability in enumerate(probabilities):
                probability = float(probability)
                if probability <= 1e-14:
                    continue
                child = self.transition(state, action, outcome)
                child_value, _ = self.plan(child, remainder, depth - 1, level + 1)
                value += probability * child_value
            if value < best_value - TIE_TOLERANCE:
                best_value = value
                best_action = action
        if best_action < 0:
            raise AssertionError("dynamic planner failed to select an action")
        return best_value, best_action

    @lru_cache(maxsize=None)
    def expected_truth_loss(
        self,
        state: DynamicState,
        available: tuple[int, ...],
        remaining: int,
        horizon: int,
        truth: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.bank.truth_loss(state, truth)
        _, action = self.plan(state, available, min(horizon, remaining), 0)
        if action < 0:
            return self.bank.truth_loss(state, truth)
        remainder = tuple(item for item in available if item != action)
        result = 0.0
        for outcome, probability in enumerate(self.bank.likelihoods[truth, action, :]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            result += probability * self.expected_truth_loss(
                child, remainder, remaining - 1, horizon, truth
            )
        return result

    def evaluate_horizon(
        self,
        horizon: int,
        *,
        execution_budget: int = 4,
        truth_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        state = self.bank.initial_state()
        available = tuple(range(self.bank.num_actions))
        planned_value, root_action = self.plan(state, available, min(horizon, execution_budget), 0)
        truths = tuple(range(self.bank.num_models)) if truth_indices is None else tuple(truth_indices)
        if not truths or any(truth < 0 or truth >= self.bank.num_models for truth in truths):
            raise ValueError("truth indices are invalid")
        truth_losses = [
            self.expected_truth_loss(state, available, execution_budget, horizon, truth)
            for truth in truths
        ]
        return {
            "horizon": int(horizon),
            "root_action_index": int(root_action),
            "root_action": self.bank.action_names[root_action],
            "planned_value": float(planned_value),
            "expected_terminal_mse": float(np.mean(truth_losses)),
            "expected_terminal_rmsle": math.sqrt(max(float(np.mean(truth_losses)), 0.0)),
            "num_truths": len(truths),
            "truth_losses": [float(value) for value in truth_losses],
        }


@dataclass(frozen=True)
class SpeculativeState:
    inference: DynamicState
    particle_weight: tuple[float, ...]

    def weights(self) -> np.ndarray:
        values = np.asarray(self.particle_weight, dtype=float)
        if not np.isfinite(values).all() or not math.isclose(
            float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise FloatingPointError("speculative particle weights are not normalized")
        return values


class SpeculativePlanner:
    """Plan over possible executable worlds while inference support expands."""

    def __init__(
        self,
        bank: ModelBank,
        proposal_cache: ProposalCache,
        particle_indices: Sequence[int],
        *,
        seed: int = 2026081700,
    ) -> None:
        particles = tuple(dict.fromkeys(int(item) for item in particle_indices))
        if not particles or any(item < 0 or item >= bank.num_models for item in particles):
            raise ValueError("speculative particle indices are invalid")
        self.bank = bank
        self.proposal_cache = proposal_cache
        self.particle_indices = particles
        self.seed = int(seed)

    def initial_state(self) -> SpeculativeState:
        count = len(self.particle_indices)
        return SpeculativeState(
            inference=self.bank.initial_state(),
            particle_weight=tuple(1.0 / count for _ in range(count)),
        )

    def _seed(self, state: SpeculativeState, action: int, outcome: int) -> int:
        key = proposal_key("speculative-seed", state.inference, action, outcome, self.seed)
        return int(key[:16], 16) % (2**31 - 1)

    def predictive(self, state: SpeculativeState, action: int) -> np.ndarray:
        result = state.weights() @ self.bank.likelihoods[
            np.asarray(self.particle_indices), action, :
        ]
        total = float(result.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("speculative predictive has no mass")
        return result / total

    def transition(self, state: SpeculativeState, action: int, outcome: int) -> SpeculativeState:
        weights = state.weights()
        posterior = weights * self.bank.likelihoods[
            np.asarray(self.particle_indices), action, outcome
        ]
        total = float(posterior.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("speculative branch has zero posterior mass")
        posterior /= total
        seed = self._seed(state, action, outcome)
        proposal = self.proposal_cache.get(state.inference, action, outcome, seed)
        inference = self.bank.transition(state.inference, action, outcome, proposal)
        return SpeculativeState(
            inference=inference,
            particle_weight=tuple(float(value) for value in posterior),
        )

    def forecast(self, state: SpeculativeState) -> np.ndarray:
        weights = state.inference.represented_weights
        models = np.asarray(state.inference.represented_models)
        return weights @ self.bank.target_features[models]

    def leaf_risk(self, state: SpeculativeState) -> float:
        forecast = self.forecast(state)
        targets = self.bank.target_features[np.asarray(self.particle_indices)]
        losses = np.mean((targets - forecast) ** 2, axis=1)
        return float(state.weights() @ losses)

    def speculative_variance(self, state: SpeculativeState, action: int) -> float:
        means = self.bank.likelihoods[
            np.asarray(self.particle_indices), action, :
        ] @ np.arange(3, dtype=float)
        weights = state.weights()
        mean = float(weights @ means)
        return float(weights @ (means - mean) ** 2)

    def candidate_actions(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        width: int,
    ) -> tuple[int, ...]:
        by_group: dict[str, tuple[float, int]] = {}
        for action in available:
            group = self.bank.action_groups[action]
            if group is None:
                continue
            score = self.speculative_variance(state, action)
            current = by_group.get(group)
            if current is None or score > current[0] + TIE_TOLERANCE or (
                abs(score - current[0]) <= TIE_TOLERANCE and action < current[1]
            ):
                by_group[group] = (score, action)
        ranked = sorted(by_group.values(), key=lambda item: (-item[0], item[1]))
        return tuple(action for _, action in ranked[:width])

    @lru_cache(maxsize=None)
    def action_value(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        action: int,
        depth: int,
        level: int,
    ) -> float:
        probabilities = self.predictive(state, action)
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(probabilities):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            child_value, _ = self.plan(child, remainder, depth - 1, level + 1)
            value += probability * child_value
        return value

    @lru_cache(maxsize=None)
    def plan(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        depth: int,
        level: int = 0,
    ) -> tuple[float, int]:
        if depth <= 0 or not available:
            return self.leaf_risk(state), -1
        width = 6 if level == 0 else 3 if level == 1 else 2
        candidates = self.candidate_actions(state, available, width)
        if not candidates:
            return self.leaf_risk(state), -1
        best_value = math.inf
        best_action = -1
        for action in candidates:
            value = self.action_value(state, available, action, depth, level)
            if value < best_value - TIE_TOLERANCE:
                best_value = value
                best_action = action
        if best_action < 0:
            raise AssertionError("speculative planner failed to select an action")
        return best_value, best_action

    @lru_cache(maxsize=None)
    def expected_truth_loss(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        horizon: int,
        truth: int,
    ) -> float:
        if remaining <= 0 or not available:
            forecast = self.forecast(state)
            return float(np.mean((forecast - self.bank.target_features[truth]) ** 2))
        _, action = self.plan(state, available, min(horizon, remaining), 0)
        if action < 0:
            forecast = self.forecast(state)
            return float(np.mean((forecast - self.bank.target_features[truth]) ** 2))
        remainder = tuple(item for item in available if item != action)
        result = 0.0
        for outcome, probability in enumerate(self.bank.likelihoods[truth, action, :]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            result += probability * self.expected_truth_loss(
                child,
                remainder,
                remaining - 1,
                horizon,
                truth,
            )
        return result

    def forced_root_risk(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        action: int,
        *,
        execution_budget: int,
        horizon: int,
    ) -> float:
        remainder = tuple(item for item in available if item != action)
        particle_prior = state.weights()
        truth_losses = []
        for truth in self.particle_indices:
            value = 0.0
            for outcome, probability in enumerate(self.bank.likelihoods[truth, action, :]):
                probability = float(probability)
                if probability <= 1e-14:
                    continue
                child = self.transition(state, action, outcome)
                value += probability * self.expected_truth_loss(
                    child,
                    remainder,
                    execution_budget - 1,
                    horizon,
                    truth,
                )
            truth_losses.append(value)
        return float(particle_prior @ np.asarray(truth_losses))

    def evaluate_horizon(self, horizon: int, *, execution_budget: int = 4) -> dict[str, Any]:
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        planned_value, root_action = self.plan(
            state,
            available,
            min(horizon, execution_budget),
            0,
        )
        truth_losses = [
            self.expected_truth_loss(
                state,
                available,
                execution_budget,
                horizon,
                truth,
            )
            for truth in self.particle_indices
        ]
        return {
            "horizon": int(horizon),
            "root_action_index": int(root_action),
            "root_action": self.bank.action_names[root_action],
            "planned_value": float(planned_value),
            "expected_terminal_mse": float(np.mean(truth_losses)),
            "expected_terminal_rmsle": math.sqrt(max(float(np.mean(truth_losses)), 0.0)),
            "num_truths": len(self.particle_indices),
            "truth_losses": [float(value) for value in truth_losses],
        }


class PolicyLadderPlanner(SpeculativePlanner):
    """Apply exact finite-budget policy improvement to dynamic support updates."""

    @lru_cache(maxsize=None)
    def policy_action_value(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
    ) -> float:
        if remaining <= 0:
            return self.leaf_risk(state)
        probabilities = self.predictive(state, action)
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(probabilities):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            if level == 1 or remaining == 1:
                child_value = self.leaf_risk(child)
            else:
                child_value = self.policy_value(
                    child,
                    remainder,
                    remaining - 1,
                    level - 1,
                )
            value += probability * child_value
        return value

    @lru_cache(maxsize=None)
    def policy_action(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        if remaining <= 0 or not available:
            return -1
        candidates = self.candidate_actions(state, available, 6)
        if not candidates:
            return -1
        return min(
            (
                self.policy_action_value(
                    state,
                    available,
                    remaining,
                    level,
                    action,
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
        remaining: int,
        level: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.leaf_risk(state)
        action = self.policy_action(state, available, remaining, level)
        if action < 0:
            return self.leaf_risk(state)
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            value += probability * self.policy_value(
                child,
                remainder,
                remaining - 1,
                level,
            )
        return value

    @lru_cache(maxsize=None)
    def expected_policy_truth_loss(
        self,
        state: SpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        truth: int,
    ) -> float:
        if remaining <= 0 or not available:
            forecast = self.forecast(state)
            return float(np.mean((forecast - self.bank.target_features[truth]) ** 2))
        action = self.policy_action(state, available, remaining, level)
        if action < 0:
            forecast = self.forecast(state)
            return float(np.mean((forecast - self.bank.target_features[truth]) ** 2))
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(self.bank.likelihoods[truth, action, :]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            value += probability * self.expected_policy_truth_loss(
                child,
                remainder,
                remaining - 1,
                level,
                truth,
            )
        return value

    def evaluate_policy_level(
        self,
        level: int,
        *,
        execution_budget: int = 4,
    ) -> dict[str, Any]:
        if level <= 0:
            raise ValueError("policy level must be positive")
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        root_action = self.policy_action(state, available, execution_budget, level)
        planned_value = self.policy_value(state, available, execution_budget, level)
        truth_losses = [
            self.expected_policy_truth_loss(
                state,
                available,
                execution_budget,
                level,
                truth,
            )
            for truth in self.particle_indices
        ]
        expected_terminal_mse = float(np.mean(truth_losses))
        return {
            "policy_level": int(level),
            "root_action_index": int(root_action),
            "root_action": self.bank.action_names[root_action],
            "planned_value": float(planned_value),
            "expected_terminal_mse": expected_terminal_mse,
            "expected_terminal_rmsle": math.sqrt(max(expected_terminal_mse, 0.0)),
            "num_truths": len(self.particle_indices),
            "truth_losses": [float(value) for value in truth_losses],
        }
