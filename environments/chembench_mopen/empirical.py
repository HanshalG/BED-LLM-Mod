"""Structure/parameter beliefs for ChemBench policy-improvement planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .mechanics import MAX_PROPOSALS, TIE_TOLERANCE, ProposalCache, proposal_key


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass(frozen=True)
class EmpiricalState:
    history: tuple[tuple[int, int], ...]
    discovered: tuple[int, ...]
    live: tuple[int, ...]
    reserve: tuple[int, ...]
    represented_mass: tuple[float, ...]
    represented_weight: tuple[float, ...]
    parameter_weight: tuple[tuple[float, ...], ...]
    outside_mass: float

    @property
    def represented_models(self) -> tuple[int, ...]:
        return self.live + self.reserve

    def weights(self) -> np.ndarray:
        values = np.asarray(self.represented_weight, dtype=float)
        if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise FloatingPointError("represented structure weights are not normalized")
        return values

    def public_key(self) -> dict[str, Any]:
        return {
            "history": [list(item) for item in self.history],
            "discovered": list(self.discovered),
            "live": list(self.live),
            "reserve": list(self.reserve),
        }


class EmpiricalParameterBank:
    def __init__(
        self,
        particle_observation_means: Sequence[np.ndarray],
        particle_target_features: Sequence[np.ndarray],
        model_names: Sequence[str],
        action_names: Sequence[str],
        action_groups: Sequence[str | None],
        action_inputs: np.ndarray,
        initial_support: Sequence[int],
        *,
        noise_level: float = 0.01,
        outside_prior: float = 0.35,
        live_cap: int = 12,
        reserve_cap: int = 12,
    ) -> None:
        if len(particle_observation_means) != len(model_names):
            raise ValueError("particle means must match model count")
        if len(particle_target_features) != len(model_names):
            raise ValueError("particle targets must match model count")
        if len(action_names) != len(action_groups):
            raise ValueError("action metadata lengths differ")
        if noise_level <= 0 or not 0 < outside_prior < 1:
            raise ValueError("noise level and outside prior are invalid")
        if live_cap <= 0 or reserve_cap < 0:
            raise ValueError("support caps are invalid")
        means = tuple(np.asarray(item, dtype=float) for item in particle_observation_means)
        targets = tuple(np.asarray(item, dtype=float) for item in particle_target_features)
        num_actions = len(action_names)
        target_width = targets[0].shape[1] if targets and targets[0].ndim == 2 else -1
        for model_means, model_targets in zip(means, targets, strict=True):
            if (
                model_means.ndim != 2
                or model_targets.ndim != 2
                or model_means.shape[0] != model_targets.shape[0]
                or model_means.shape[1] != num_actions
                or model_targets.shape[1] != target_width
                or model_means.shape[0] == 0
            ):
                raise ValueError("particle arrays have incompatible shapes")
            if not np.isfinite(model_means).all() or not np.isfinite(model_targets).all():
                raise ValueError("particle arrays must be finite")
        action_inputs = np.asarray(action_inputs, dtype=float)
        if action_inputs.shape != (num_actions, 7) or not np.isfinite(action_inputs).all():
            raise ValueError("action inputs must have shape (actions, 7)")
        initial = tuple(dict.fromkeys(int(item) for item in initial_support))
        if not initial or any(item < 0 or item >= len(model_names) for item in initial):
            raise ValueError("initial support is invalid")

        self.particle_observation_means = means
        self.particle_target_features = targets
        self.model_names = tuple(model_names)
        self.action_names = tuple(action_names)
        self.action_groups = tuple(action_groups)
        self.action_inputs = action_inputs
        self.initial_support = initial
        self.noise_level = float(noise_level)
        self.outside_prior = float(outside_prior)
        self.live_cap = int(live_cap)
        self.reserve_cap = int(reserve_cap)
        self.particle_log_means = tuple(np.log1p(np.maximum(item, 0.0)) for item in means)
        pooled = np.concatenate(self.particle_log_means, axis=0)
        self.bin_thresholds = np.quantile(pooled, (1.0 / 3.0, 2.0 / 3.0), axis=0).T
        self.particle_likelihoods = tuple(
            self.likelihoods_for_means(item) for item in self.particle_observation_means
        )
        self.representative_log_observations = self._representative_log_observations()

    @property
    def num_models(self) -> int:
        return len(self.model_names)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    def likelihoods_for_means(self, means: np.ndarray) -> np.ndarray:
        means = np.asarray(means, dtype=float)
        if means.ndim != 2 or means.shape[1] != self.num_actions:
            raise ValueError("means must have shape (particles, actions)")
        result = np.empty((means.shape[0], self.num_actions, 3), dtype=float)
        log_means = np.log1p(np.maximum(means, 0.0))
        sigma = np.maximum(
            self.noise_level * np.maximum(means, 0.0) / (1.0 + np.maximum(means, 0.0)),
            1e-6,
        )
        for particle in range(means.shape[0]):
            for action in range(self.num_actions):
                low, high = self.bin_thresholds[action]
                low_mass = _normal_cdf(
                    (float(low) - float(log_means[particle, action]))
                    / float(sigma[particle, action])
                )
                high_mass = 1.0 - _normal_cdf(
                    (float(high) - float(log_means[particle, action]))
                    / float(sigma[particle, action])
                )
                result[particle, action] = (
                    low_mass,
                    max(0.0, 1.0 - low_mass - high_mass),
                    high_mass,
                )
                result[particle, action] = np.maximum(result[particle, action], 1e-300)
                result[particle, action] /= result[particle, action].sum()
        return result

    def _representative_log_observations(self) -> np.ndarray:
        representatives = np.empty((self.num_actions, 3), dtype=float)
        for action in range(self.num_actions):
            values = []
            probabilities = []
            for model, log_means in enumerate(self.particle_log_means):
                for particle in range(log_means.shape[0]):
                    values.append(float(log_means[particle, action]))
                    probabilities.append(self.particle_likelihoods[model][particle, action])
            value_array = np.asarray(values, dtype=float)
            probability_array = np.asarray(probabilities, dtype=float)
            for outcome in range(3):
                weights = probability_array[:, outcome]
                total = float(weights.sum())
                if total <= 0:
                    raise FloatingPointError("outcome bin has no representative mass")
                representatives[action, outcome] = float(weights @ value_array / total)
        return representatives

    def particle_log_likelihoods(
        self, model: int, history: Sequence[tuple[int, int]]
    ) -> np.ndarray:
        result = np.zeros(self.particle_likelihoods[model].shape[0], dtype=float)
        for action, outcome in history:
            result += np.log(
                np.maximum(self.particle_likelihoods[model][:, action, outcome], 1e-300)
            )
        return result

    def structure_log_evidence(
        self, model: int, history: Sequence[tuple[int, int]]
    ) -> float:
        values = self.particle_log_likelihoods(model, history)
        return _logsumexp(values) - math.log(len(values))

    def state(self, history: Sequence[tuple[int, int]], discovered: Sequence[int]) -> EmpiricalState:
        history_tuple = tuple((int(action), int(outcome)) for action, outcome in history)
        unique = tuple(sorted(set(int(item) for item in discovered)))
        return self._state(history_tuple, unique)

    @lru_cache(maxsize=None)
    def _state(
        self,
        history_tuple: tuple[tuple[int, int], ...],
        unique: tuple[int, ...],
    ) -> EmpiricalState:
        if not unique or any(item < 0 or item >= self.num_models for item in unique):
            raise ValueError("discovered support is invalid")
        evidences = np.asarray(
            [self.structure_log_evidence(model, history_tuple) for model in unique], dtype=float
        )
        known_scores = math.log(1.0 - self.outside_prior) - math.log(len(unique)) + evidences
        ranking = sorted(range(len(unique)), key=lambda index: (-known_scores[index], unique[index]))
        kept_positions = ranking[: self.live_cap + self.reserve_cap]
        kept_models = tuple(unique[index] for index in kept_positions)
        kept_scores = np.asarray([known_scores[index] for index in kept_positions], dtype=float)
        live_count = min(self.live_cap, len(kept_models))
        live = kept_models[:live_count]
        reserve = kept_models[live_count:]
        represented_normalizer = _logsumexp(kept_scores)
        represented_weights = np.exp(kept_scores - represented_normalizer)
        outside_score = math.log(self.outside_prior) - len(history_tuple) * math.log(3.0)
        total_normalizer = _logsumexp(np.append(kept_scores, outside_score))
        represented_joint = np.exp(kept_scores - total_normalizer)
        outside_mass = math.exp(outside_score - total_normalizer)
        parameter_weights = []
        for model in kept_models:
            values = self.particle_log_likelihoods(model, history_tuple)
            parameter_weights.append(tuple(float(value) for value in np.exp(values - _logsumexp(values))))
        if not math.isclose(
            float(represented_joint.sum()) + outside_mass,
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise FloatingPointError("empirical belief did not normalize")
        return EmpiricalState(
            history=history_tuple,
            discovered=tuple(sorted(kept_models)),
            live=live,
            reserve=reserve,
            represented_mass=tuple(float(value) for value in represented_joint),
            represented_weight=tuple(float(value) for value in represented_weights),
            parameter_weight=tuple(parameter_weights),
            outside_mass=float(outside_mass),
        )

    def initial_state(self) -> EmpiricalState:
        return self.state((), self.initial_support)

    def structure_predictive(self, state: EmpiricalState, position: int, action: int) -> np.ndarray:
        model = state.represented_models[position]
        parameter_weights = np.asarray(state.parameter_weight[position], dtype=float)
        return parameter_weights @ self.particle_likelihoods[model][:, action, :]

    def predictive(self, state: EmpiricalState, action: int) -> np.ndarray:
        result = np.full(3, state.outside_mass / 3.0, dtype=float)
        for position, mass in enumerate(state.represented_mass):
            result += float(mass) * self.structure_predictive(state, position, action)
        return result / result.sum()

    def transition(
        self,
        state: EmpiricalState,
        action: int,
        outcome: int,
        proposal: Sequence[int],
    ) -> EmpiricalState:
        if action < 0 or action >= self.num_actions or outcome not in (0, 1, 2):
            raise ValueError("transition action or outcome is invalid")
        accepted = []
        seen = set(state.discovered)
        for item in proposal:
            if not isinstance(item, (int, np.integer)):
                continue
            model = int(item)
            if model < 0 or model >= self.num_models or model in seen:
                continue
            accepted.append(model)
            seen.add(model)
            if len(accepted) == MAX_PROPOSALS:
                break
        return self.state(
            state.history + ((action, outcome),),
            state.discovered + tuple(accepted),
        )

    def forecast(self, state: EmpiricalState) -> np.ndarray:
        result = np.zeros(self.particle_target_features[0].shape[1], dtype=float)
        for position, structure_weight in enumerate(state.weights()):
            model = state.represented_models[position]
            parameter_weights = np.asarray(state.parameter_weight[position], dtype=float)
            result += float(structure_weight) * (
                parameter_weights @ self.particle_target_features[model]
            )
        return result

    def predicted_log_rate(self, state: EmpiricalState, action: int) -> float:
        value = 0.0
        for position, structure_weight in enumerate(state.weights()):
            model = state.represented_models[position]
            parameter_weights = np.asarray(state.parameter_weight[position], dtype=float)
            value += float(structure_weight) * float(
                parameter_weights @ self.particle_log_means[model][:, action]
            )
        return value

    def residual_report(self, state: EmpiricalState) -> dict[str, Any]:
        observations = []
        residuals = []
        for index, (action, outcome) in enumerate(state.history):
            prefix = self.state(state.history[:index], state.discovered)
            prediction = self.predicted_log_rate(prefix, action)
            observed = float(self.representative_log_observations[action, outcome])
            residual = observed - prediction
            residuals.append(residual)
            observations.append(
                {
                    "action": self.action_names[action],
                    "observed_log_rate": observed,
                    "predicted_log_rate": prediction,
                    "signed_residual": residual,
                }
            )
        correlations = {}
        if len(residuals) >= 2 and float(np.std(residuals)) > 0:
            action_indices = [item[0] for item in state.history]
            for column, name in enumerate(("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")):
                values = self.action_inputs[action_indices, column]
                if float(np.std(values)) > 0:
                    correlations[name] = float(np.corrcoef(values, residuals)[0, 1])
        return {
            "observations": observations,
            "latest_signed_residual": residuals[-1] if residuals else None,
            "residual_input_correlations": correlations,
            "top_models": [
                {
                    "name": self.model_names[model],
                    "weight": float(state.represented_weight[position]),
                }
                for position, model in enumerate(state.represented_models[:4])
            ],
        }


class EmpiricalOracleProposer:
    mode = "empirical_registry_oracle"

    def __init__(self, bank: EmpiricalParameterBank) -> None:
        self.bank = bank

    def propose(
        self, state: EmpiricalState, action: int, outcome: int, seed: int
    ) -> tuple[int, ...]:
        del seed
        history = state.history + ((action, outcome),)
        missing = [item for item in range(self.bank.num_models) if item not in state.discovered]
        missing.sort(key=lambda item: (-self.bank.structure_log_evidence(item, history), item))
        return tuple(missing[:MAX_PROPOSALS])


@dataclass(frozen=True)
class EmpiricalSpeculativeState:
    inference: EmpiricalState
    particle_weight: tuple[float, ...]

    def weights(self) -> np.ndarray:
        values = np.asarray(self.particle_weight, dtype=float)
        if not np.isfinite(values).all() or not math.isclose(
            float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise FloatingPointError("speculative weights are not normalized")
        return values


class EmpiricalPolicyLadderPlanner:
    def __init__(
        self,
        bank: EmpiricalParameterBank,
        proposal_cache: ProposalCache,
        speculative_models: Sequence[int],
        truth_observation_means: np.ndarray,
        truth_target_features: np.ndarray,
        *,
        seed: int = 2026081800,
    ) -> None:
        models = tuple(dict.fromkeys(int(item) for item in speculative_models))
        if not models or any(item < 0 or item >= bank.num_models for item in models):
            raise ValueError("speculative models are invalid")
        truth_means = np.asarray(truth_observation_means, dtype=float)
        truth_targets = np.asarray(truth_target_features, dtype=float)
        if (
            truth_means.shape != (len(models), bank.num_actions)
            or truth_targets.ndim != 2
            or truth_targets.shape[0] != len(models)
        ):
            raise ValueError("truth arrays do not match speculative structures")
        self.bank = bank
        self.proposal_cache = proposal_cache
        self.speculative_models = models
        self.speculative_pairs = tuple(
            (model, particle)
            for model in models
            for particle in range(bank.particle_observation_means[model].shape[0])
        )
        self._pair_likelihoods = np.asarray(
            [
                bank.particle_likelihoods[model][particle]
                for model, particle in self.speculative_pairs
            ],
            dtype=float,
        )
        self._pair_targets = np.asarray(
            [
                bank.particle_target_features[model][particle]
                for model, particle in self.speculative_pairs
            ],
            dtype=float,
        )
        self.truth_likelihoods = bank.likelihoods_for_means(truth_means)
        self.truth_target_features = truth_targets
        self.seed = int(seed)

    def initial_state(self) -> EmpiricalSpeculativeState:
        structure_mass = 1.0 / len(self.speculative_models)
        weights = []
        for model in self.speculative_models:
            count = self.bank.particle_observation_means[model].shape[0]
            weights.extend(structure_mass / count for _ in range(count))
        return EmpiricalSpeculativeState(
            inference=self.bank.initial_state(),
            particle_weight=tuple(weights),
        )

    def _seed(self, state: EmpiricalSpeculativeState, action: int, outcome: int) -> int:
        key = proposal_key("empirical-speculative-seed", state.inference, action, outcome, self.seed)
        return int(key[:16], 16) % (2**31 - 1)

    def pair_likelihoods(self, action: int, outcome: int | None = None) -> np.ndarray:
        values = self._pair_likelihoods[:, action]
        return values if outcome is None else values[:, outcome]

    @lru_cache(maxsize=None)
    def predictive(self, state: EmpiricalSpeculativeState, action: int) -> np.ndarray:
        result = state.weights() @ self.pair_likelihoods(action)
        return result / result.sum()

    @lru_cache(maxsize=None)
    def transition(
        self, state: EmpiricalSpeculativeState, action: int, outcome: int
    ) -> EmpiricalSpeculativeState:
        posterior = state.weights() * self.pair_likelihoods(action, outcome)
        total = float(posterior.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("speculative branch has no mass")
        posterior /= total
        seed = self._seed(state, action, outcome)
        proposal = self.proposal_cache.get(state.inference, action, outcome, seed)
        inference = self.bank.transition(state.inference, action, outcome, proposal)
        return EmpiricalSpeculativeState(
            inference=inference,
            particle_weight=tuple(float(value) for value in posterior),
        )

    @lru_cache(maxsize=None)
    def forecast(self, state: EmpiricalSpeculativeState) -> np.ndarray:
        return self.bank.forecast(state.inference)

    @lru_cache(maxsize=None)
    def leaf_risk(self, state: EmpiricalSpeculativeState) -> float:
        forecast = self.forecast(state)
        losses = np.mean((self._pair_targets - forecast) ** 2, axis=1)
        return float(state.weights() @ losses)

    def candidate_actions(
        self,
        state: EmpiricalSpeculativeState,
        available: tuple[int, ...],
        width: int,
    ) -> tuple[int, ...]:
        by_group: dict[str, tuple[float, int]] = {}
        weights = state.weights()
        for action in available:
            group = self.bank.action_groups[action]
            if group is None:
                continue
            means = self.pair_likelihoods(action) @ np.arange(3, dtype=float)
            mean = float(weights @ means)
            score = float(weights @ (means - mean) ** 2)
            current = by_group.get(group)
            if current is None or score > current[0] + TIE_TOLERANCE or (
                abs(score - current[0]) <= TIE_TOLERANCE and action < current[1]
            ):
                by_group[group] = (score, action)
        ranked = sorted(by_group.values(), key=lambda item: (-item[0], item[1]))
        return tuple(action for _, action in ranked[:width])

    @lru_cache(maxsize=None)
    def policy_action_value(
        self,
        state: EmpiricalSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
    ) -> float:
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = self.transition(state, action, outcome)
            child_value = (
                self.leaf_risk(child)
                if level == 1 or remaining == 1
                else self.policy_value(child, remainder, remaining - 1, level - 1)
            )
            value += probability * child_value
        return value

    @lru_cache(maxsize=None)
    def policy_action(
        self,
        state: EmpiricalSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        if remaining <= 0 or not available:
            return -1
        candidates = self.candidate_actions(state, available, 6)
        return min(
            (
                self.policy_action_value(state, available, remaining, level, action),
                action,
            )
            for action in candidates
        )[1]

    @lru_cache(maxsize=None)
    def policy_value(
        self,
        state: EmpiricalSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.leaf_risk(state)
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(self.predictive(state, action)):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            value += probability * self.policy_value(
                self.transition(state, action, outcome),
                remainder,
                remaining - 1,
                level,
            )
        return value

    @lru_cache(maxsize=None)
    def truth_value(
        self,
        state: EmpiricalSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        truth_position: int,
    ) -> float:
        if remaining <= 0 or not available:
            return float(
                np.mean(
                    (self.forecast(state) - self.truth_target_features[truth_position]) ** 2
                )
            )
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for outcome, probability in enumerate(self.truth_likelihoods[truth_position, action]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            value += probability * self.truth_value(
                self.transition(state, action, outcome),
                remainder,
                remaining - 1,
                level,
                truth_position,
            )
        return value

    def evaluate_policy_level(self, level: int, *, execution_budget: int = 4) -> dict[str, Any]:
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        root_action = self.policy_action(state, available, execution_budget, level)
        planned_value = self.policy_value(state, available, execution_budget, level)
        truth_losses = [
            self.truth_value(state, available, execution_budget, level, truth_position)
            for truth_position in range(len(self.speculative_models))
        ]
        truth_mean = float(np.mean(truth_losses))
        return {
            "policy_level": int(level),
            "root_action_index": int(root_action),
            "root_action": self.bank.action_names[root_action],
            "planned_particle_risk": float(planned_value),
            "expected_truth_mse": truth_mean,
            "expected_truth_rmsle": math.sqrt(max(truth_mean, 0.0)),
            "truth_losses": [float(value) for value in truth_losses],
        }
