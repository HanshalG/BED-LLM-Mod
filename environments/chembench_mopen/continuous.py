"""Continuous-rate structure/parameter planning for ChemBench M-open BED."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .mechanics import MAX_PROPOSALS, TIE_TOLERANCE, ProposalCache, proposal_key


LOG_2PI = math.log(2.0 * math.pi)
CACHE_SIZE = 8_192


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _normal_logpdf(value: float, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    residual = (float(value) - means) / sigmas
    return -0.5 * (residual**2 + LOG_2PI) - np.log(sigmas)


@dataclass(frozen=True)
class PredictiveBranch:
    probability: float
    observation: float


def weighted_quantile_branches(
    values: np.ndarray,
    weights: np.ndarray,
    num_branches: int,
) -> tuple[PredictiveBranch, ...]:
    """Compress a weighted discrete predictive into equal-mass quantile branches."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or values.size == 0
        or num_branches <= 0
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("weighted predictive inputs are invalid")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weighted predictive has no mass")
    weights = weights / total
    positive = weights > 1e-14
    values = values[positive]
    weights = weights[positive]
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    midpoints = cumulative - 0.5 * ordered_weights
    assignments = np.minimum((midpoints * num_branches).astype(int), num_branches - 1)
    branches = []
    for branch_index in range(num_branches):
        members = np.flatnonzero(assignments == branch_index)
        if members.size == 0:
            continue
        member_weights = ordered_weights[members]
        mass = float(member_weights.sum())
        conditional_mean = float(member_weights @ ordered_values[members] / mass)
        representative = int(
            members[np.argmin(np.abs(ordered_values[members] - conditional_mean))]
        )
        branches.append(PredictiveBranch(mass, float(ordered_values[representative])))
    normalization = sum(branch.probability for branch in branches)
    return tuple(
        PredictiveBranch(branch.probability / normalization, branch.observation)
        for branch in branches
    )


@dataclass(frozen=True)
class ContinuousState:
    history: tuple[tuple[int, float], ...]
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
            raise FloatingPointError("continuous structure weights are not normalized")
        return values

    def public_key(self) -> dict[str, Any]:
        return {
            "history": [[action, observation] for action, observation in self.history],
            "discovered": list(self.discovered),
            "live": list(self.live),
            "reserve": list(self.reserve),
        }


class ContinuousParameterBank:
    """Finite structure/parameter posterior under raw Gaussian rate observations."""

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
        absolute_noise_floor: float = 1e-8,
        absolute_log_noise_floor: float = 1e-6,
        parameter_kernel_scale: float = 0.0,
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
        if (
            noise_level <= 0
            or absolute_noise_floor <= 0
            or absolute_log_noise_floor <= 0
            or parameter_kernel_scale < 0
            or not 0 < outside_prior < 1
        ):
            raise ValueError("continuous noise or outside prior is invalid")
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
            if (
                not np.isfinite(model_means).all()
                or not np.isfinite(model_targets).all()
                or np.any(model_means < 0)
            ):
                raise ValueError("particle arrays must be finite with nonnegative rates")
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
        self.absolute_noise_floor = float(absolute_noise_floor)
        self.absolute_log_noise_floor = float(absolute_log_noise_floor)
        self.parameter_kernel_scale = float(parameter_kernel_scale)
        self.outside_prior = float(outside_prior)
        self.live_cap = int(live_cap)
        self.reserve_cap = int(reserve_cap)
        self.particle_log_means = tuple(np.log1p(item) for item in means)
        self.particle_log_sigmas = tuple(
            self.log_sigmas_for_model(item) for item in means
        )
        particle_counts = {item.shape[0] for item in means}
        self.stacked_particle_log_means = (
            np.stack(self.particle_log_means) if len(particle_counts) == 1 else None
        )
        self.stacked_particle_log_sigmas = (
            np.stack(self.particle_log_sigmas)
            if self.stacked_particle_log_means is not None
            else None
        )
        pooled_log_means = np.concatenate(self.particle_log_means, axis=0)
        self.outside_log_mean = np.mean(pooled_log_means, axis=0)
        self.outside_log_sigma = np.maximum(np.std(pooled_log_means, axis=0), 0.5)

    @property
    def num_models(self) -> int:
        return len(self.model_names)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    def raw_sigmas_for_means(self, means: np.ndarray) -> np.ndarray:
        values = np.asarray(means, dtype=float)
        return np.maximum(self.noise_level * values, self.absolute_noise_floor)

    def measurement_log_sigmas_for_means(self, means: np.ndarray) -> np.ndarray:
        values = np.asarray(means, dtype=float)
        return np.maximum(
            self.noise_level * values / (1.0 + values),
            self.absolute_log_noise_floor,
        )

    def log_sigmas_for_model(self, means: np.ndarray) -> np.ndarray:
        values = np.asarray(means, dtype=float)
        measurement = self.measurement_log_sigmas_for_means(values)
        if self.parameter_kernel_scale == 0 or values.shape[0] <= 1:
            return measurement
        log_means = np.log1p(values)
        bandwidth = (
            self.parameter_kernel_scale
            * np.std(log_means, axis=0, ddof=1)
            * values.shape[0] ** (-0.2)
        )
        return np.sqrt(measurement**2 + bandwidth[None, :] ** 2)

    def particle_log_likelihoods(
        self,
        model: int,
        history: Sequence[tuple[int, float]],
    ) -> np.ndarray:
        history_tuple = tuple(history)
        if self.stacked_particle_log_means is not None:
            return self._stacked_particle_log_likelihoods(history_tuple)[model]
        result = np.zeros(self.particle_observation_means[model].shape[0], dtype=float)
        for action, observation in history_tuple:
            result += _normal_logpdf(
                math.log1p(observation),
                self.particle_log_means[model][:, action],
                self.particle_log_sigmas[model][:, action],
            )
        return result

    def structure_log_evidence(
        self,
        model: int,
        history: Sequence[tuple[int, float]],
    ) -> float:
        if self.stacked_particle_log_means is None:
            values = self.particle_log_likelihoods(model, history)
            return _logsumexp(values) - math.log(len(values))
        return float(self.all_structure_log_evidences(history)[model])

    @lru_cache(maxsize=CACHE_SIZE)
    def _stacked_particle_log_likelihoods(
        self,
        history: tuple[tuple[int, float], ...],
    ) -> np.ndarray:
        if (
            self.stacked_particle_log_means is None
            or self.stacked_particle_log_sigmas is None
        ):
            raise ValueError("stacked likelihoods require equal particle counts")
        values = np.zeros(self.stacked_particle_log_means.shape[:2], dtype=float)
        for action, observation in history:
            values += _normal_logpdf(
                math.log1p(observation),
                self.stacked_particle_log_means[:, :, action],
                self.stacked_particle_log_sigmas[:, :, action],
            )
        values.setflags(write=False)
        return values

    def all_structure_log_evidences(
        self,
        history: Sequence[tuple[int, float]],
    ) -> np.ndarray:
        return self._all_structure_log_evidences(tuple(history))

    @lru_cache(maxsize=CACHE_SIZE)
    def _all_structure_log_evidences(
        self,
        history: tuple[tuple[int, float], ...],
    ) -> np.ndarray:
        if (
            self.stacked_particle_log_means is None
            or self.stacked_particle_log_sigmas is None
        ):
            return np.asarray(
                [self.structure_log_evidence(model, history) for model in range(self.num_models)],
                dtype=float,
            )
        values = self._stacked_particle_log_likelihoods(history)
        maximum = np.max(values, axis=1)
        result = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)) - math.log(
            values.shape[1]
        )
        result.setflags(write=False)
        return result

    def outside_log_likelihood(self, history: Sequence[tuple[int, float]]) -> float:
        value = 0.0
        for action, observation in history:
            transformed = math.log1p(max(float(observation), 0.0))
            sigma = float(self.outside_log_sigma[action])
            residual = (transformed - float(self.outside_log_mean[action])) / sigma
            value += -0.5 * (residual**2 + LOG_2PI) - math.log(sigma)
        return value

    def state(
        self,
        history: Sequence[tuple[int, float]],
        discovered: Sequence[int],
    ) -> ContinuousState:
        history_tuple = tuple((int(action), float(observation)) for action, observation in history)
        if any(
            action < 0
            or action >= self.num_actions
            or not math.isfinite(observation)
            or observation < 0
            for action, observation in history_tuple
        ):
            raise ValueError("continuous history is invalid")
        unique = tuple(sorted(set(int(item) for item in discovered)))
        return self._state(history_tuple, unique)

    @lru_cache(maxsize=CACHE_SIZE)
    def _state(
        self,
        history: tuple[tuple[int, float], ...],
        unique: tuple[int, ...],
    ) -> ContinuousState:
        if not unique or any(item < 0 or item >= self.num_models for item in unique):
            raise ValueError("discovered support is invalid")
        all_evidences = self.all_structure_log_evidences(history)
        evidences = all_evidences[np.asarray(unique)]
        known_scores = math.log(1.0 - self.outside_prior) - math.log(len(unique)) + evidences
        ranking = sorted(range(len(unique)), key=lambda index: (-known_scores[index], unique[index]))
        kept_positions = ranking[: self.live_cap + self.reserve_cap]
        kept_models = tuple(unique[index] for index in kept_positions)
        kept_scores = np.asarray([known_scores[index] for index in kept_positions], dtype=float)
        live_count = min(self.live_cap, len(kept_models))
        live = kept_models[:live_count]
        reserve = kept_models[live_count:]
        represented_weights = np.exp(kept_scores - _logsumexp(kept_scores))
        outside_score = math.log(self.outside_prior) + self.outside_log_likelihood(history)
        total_normalizer = _logsumexp(np.append(kept_scores, outside_score))
        represented_joint = np.exp(kept_scores - total_normalizer)
        outside_mass = math.exp(outside_score - total_normalizer)
        parameter_weights = []
        stacked_values = (
            self._stacked_particle_log_likelihoods(history)
            if self.stacked_particle_log_means is not None
            else None
        )
        for model in kept_models:
            values = (
                stacked_values[model]
                if stacked_values is not None
                else self.particle_log_likelihoods(model, history)
            )
            parameter_weights.append(tuple(float(item) for item in np.exp(values - _logsumexp(values))))
        if not math.isclose(
            float(represented_joint.sum()) + outside_mass,
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise FloatingPointError("continuous belief did not normalize")
        return ContinuousState(
            history=history,
            discovered=tuple(sorted(kept_models)),
            live=live,
            reserve=reserve,
            represented_mass=tuple(float(item) for item in represented_joint),
            represented_weight=tuple(float(item) for item in represented_weights),
            parameter_weight=tuple(parameter_weights),
            outside_mass=float(outside_mass),
        )

    def initial_state(self) -> ContinuousState:
        return self.state((), self.initial_support)

    def transition(
        self,
        state: ContinuousState,
        action: int,
        observation: float,
        proposal: Sequence[int],
    ) -> ContinuousState:
        if action < 0 or action >= self.num_actions or observation < 0 or not math.isfinite(observation):
            raise ValueError("continuous transition is invalid")
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
            state.history + ((action, float(observation)),),
            state.discovered + tuple(accepted),
        )

    def forecast(self, state: ContinuousState) -> np.ndarray:
        result = np.zeros(self.particle_target_features[0].shape[1], dtype=float)
        for position, structure_weight in enumerate(state.weights()):
            model = state.represented_models[position]
            parameter_weights = np.asarray(state.parameter_weight[position], dtype=float)
            active = parameter_weights > 1e-14
            result += float(structure_weight) * (
                parameter_weights[active] @ self.particle_target_features[model][active]
            )
        return result

    def predicted_rate(self, state: ContinuousState, action: int) -> float:
        result = 0.0
        for position, structure_weight in enumerate(state.weights()):
            model = state.represented_models[position]
            parameter_weights = np.asarray(state.parameter_weight[position], dtype=float)
            active = parameter_weights > 1e-14
            result += float(structure_weight) * float(
                parameter_weights[active]
                @ self.particle_observation_means[model][active, action]
            )
        return result

    def residual_report(self, state: ContinuousState) -> dict[str, Any]:
        observations = []
        residuals = []
        for index, (action, observation) in enumerate(state.history):
            prefix = self.state(state.history[:index], state.discovered)
            prediction = self.predicted_rate(prefix, action)
            residual = math.log1p(observation) - math.log1p(max(prediction, 0.0))
            residuals.append(residual)
            observations.append(
                {
                    "action": self.action_names[action],
                    "observed_rate": observation,
                    "predicted_rate": prediction,
                    "signed_log_rate_residual": residual,
                }
            )
        correlations = {}
        if len(residuals) >= 2 and float(np.std(residuals)) > 0:
            action_indices = [item[0] for item in state.history]
            names = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
            for column, name in enumerate(names):
                values = self.action_inputs[action_indices, column]
                if float(np.std(values)) > 0:
                    correlations[name] = float(np.corrcoef(values, residuals)[0, 1])
        return {
            "observations": observations,
            "latest_signed_log_rate_residual": residuals[-1] if residuals else None,
            "residual_input_correlations": correlations,
            "top_models": [
                {
                    "name": self.model_names[model],
                    "weight": float(state.represented_weight[position]),
                }
                for position, model in enumerate(state.represented_models[:4])
            ],
        }


class ContinuousOracleProposer:
    mode = "continuous_registry_oracle"

    def __init__(self, bank: ContinuousParameterBank) -> None:
        self.bank = bank

    def propose(
        self,
        state: ContinuousState,
        action: int,
        observation: float,
        seed: int,
    ) -> tuple[int, ...]:
        del seed
        history = state.history + ((action, float(observation)),)
        missing = [item for item in range(self.bank.num_models) if item not in state.discovered]
        evidences = self.bank.all_structure_log_evidences(history)
        missing.sort(key=lambda item: (-evidences[item], item))
        return tuple(missing[:MAX_PROPOSALS])


@dataclass(frozen=True)
class ContinuousSpeculativeState:
    inference: ContinuousState
    particle_weight: tuple[float, ...]

    def weights(self) -> np.ndarray:
        values = np.asarray(self.particle_weight, dtype=float)
        if not np.isfinite(values).all() or not math.isclose(
            float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise FloatingPointError("continuous speculative weights are not normalized")
        return values


class ContinuousPolicyLadderPlanner:
    def __init__(
        self,
        bank: ContinuousParameterBank,
        proposal_cache: ProposalCache,
        speculative_models: Sequence[int],
        truth_observation_means: np.ndarray,
        truth_target_features: np.ndarray,
        *,
        speculative_particle_observation_means: Sequence[np.ndarray] | None = None,
        speculative_particle_target_features: Sequence[np.ndarray] | None = None,
        num_branches: int = 5,
        branch_counts_by_remaining: Sequence[int] | None = None,
        predictive_noise_order: int = 3,
        truth_quadrature_order: int = 3,
        seed: int = 2026081900,
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
            or np.any(truth_means < 0)
        ):
            raise ValueError("truth arrays do not match speculative structures")
        branch_counts = (
            tuple(int(item) for item in branch_counts_by_remaining)
            if branch_counts_by_remaining is not None
            else (int(num_branches),)
        )
        if (
            num_branches <= 0
            or not branch_counts
            or any(item <= 0 for item in branch_counts)
            or predictive_noise_order <= 0
            or truth_quadrature_order <= 0
        ):
            raise ValueError("continuous quadrature settings must be positive")
        self.bank = bank
        self.proposal_cache = proposal_cache
        self.speculative_models = models
        speculative_means = (
            tuple(np.asarray(item, dtype=float) for item in speculative_particle_observation_means)
            if speculative_particle_observation_means is not None
            else bank.particle_observation_means
        )
        speculative_targets = (
            tuple(np.asarray(item, dtype=float) for item in speculative_particle_target_features)
            if speculative_particle_target_features is not None
            else bank.particle_target_features
        )
        if len(speculative_means) != bank.num_models or len(speculative_targets) != bank.num_models:
            raise ValueError("speculative particle arrays must match the model bank")
        for model in models:
            if (
                speculative_means[model].ndim != 2
                or speculative_targets[model].ndim != 2
                or speculative_means[model].shape[0] != speculative_targets[model].shape[0]
                or speculative_means[model].shape[1] != bank.num_actions
                or speculative_targets[model].shape[1] != truth_targets.shape[1]
                or speculative_means[model].shape[0] == 0
                or not np.isfinite(speculative_means[model]).all()
                or not np.isfinite(speculative_targets[model]).all()
                or np.any(speculative_means[model] < 0)
            ):
                raise ValueError("speculative particle arrays are invalid")
        self.speculative_particle_observation_means = speculative_means
        self.speculative_particle_target_features = speculative_targets
        self.speculative_pairs = tuple(
            (model, particle)
            for model in models
            for particle in range(speculative_means[model].shape[0])
        )
        self._pair_raw_means = np.asarray(
            [speculative_means[model][particle] for model, particle in self.speculative_pairs],
            dtype=float,
        )
        self._pair_log_means = np.log1p(self._pair_raw_means)
        self._pair_log_sigmas = bank.measurement_log_sigmas_for_means(self._pair_raw_means)
        self._pair_targets = np.asarray(
            [speculative_targets[model][particle] for model, particle in self.speculative_pairs],
            dtype=float,
        )
        self._pair_target_squared_norm = np.mean(self._pair_targets**2, axis=1)
        self.truth_observation_means = truth_means
        self.truth_observation_sigmas = bank.raw_sigmas_for_means(truth_means)
        self.truth_target_features = truth_targets
        self.num_branches = int(num_branches)
        self.branch_counts_by_remaining = branch_counts
        predictive_nodes, predictive_weights = np.polynomial.hermite.hermgauss(
            predictive_noise_order
        )
        self.predictive_nodes = np.sqrt(2.0) * predictive_nodes
        self.predictive_weights = predictive_weights / math.sqrt(math.pi)
        nodes, weights = np.polynomial.hermite.hermgauss(truth_quadrature_order)
        self.truth_nodes = np.sqrt(2.0) * nodes
        self.truth_weights = weights / math.sqrt(math.pi)
        self.seed = int(seed)

    def initial_state(self) -> ContinuousSpeculativeState:
        structure_mass = 1.0 / len(self.speculative_models)
        weights = []
        for model in self.speculative_models:
            count = self.speculative_particle_observation_means[model].shape[0]
            weights.extend(structure_mass / count for _ in range(count))
        return ContinuousSpeculativeState(
            inference=self.bank.initial_state(),
            particle_weight=tuple(weights),
        )

    def _seed(self, state: ContinuousSpeculativeState, action: int, observation: float) -> int:
        key = proposal_key(
            "continuous-speculative-seed",
            state.inference,
            action,
            observation,
            self.seed,
        )
        return int(key[:16], 16) % (2**31 - 1)

    def pair_log_likelihood(self, action: int, observation: float) -> np.ndarray:
        return _normal_logpdf(
            math.log1p(observation),
            self._pair_log_means[:, action],
            self._pair_log_sigmas[:, action],
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def predictive_branches(
        self,
        state: ContinuousSpeculativeState,
        action: int,
        num_branches: int | None = None,
    ) -> tuple[PredictiveBranch, ...]:
        observations = np.maximum(
            self._pair_log_means[:, action, None]
            + self._pair_log_sigmas[:, action, None] * self.predictive_nodes[None, :],
            0.0,
        )
        weights = state.weights()[:, None] * self.predictive_weights[None, :]
        return weighted_quantile_branches(
            np.expm1(observations.reshape(-1)),
            weights.reshape(-1),
            self.num_branches if num_branches is None else int(num_branches),
        )

    def branch_count(self, remaining: int) -> int:
        if remaining <= 0:
            raise ValueError("remaining budget must be positive")
        index = min(remaining, len(self.branch_counts_by_remaining)) - 1
        return self.branch_counts_by_remaining[index]

    @lru_cache(maxsize=CACHE_SIZE)
    def transition(
        self,
        state: ContinuousSpeculativeState,
        action: int,
        observation: float,
    ) -> ContinuousSpeculativeState:
        log_weights = np.log(np.maximum(state.weights(), 1e-300))
        log_weights += self.pair_log_likelihood(action, observation)
        posterior = np.exp(log_weights - _logsumexp(log_weights))
        seed = self._seed(state, action, observation)
        proposal = self.proposal_cache.get(state.inference, action, observation, seed)
        inference = self.bank.transition(state.inference, action, observation, proposal)
        return ContinuousSpeculativeState(
            inference=inference,
            particle_weight=tuple(float(item) for item in posterior),
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def forecast(self, state: ContinuousSpeculativeState) -> np.ndarray:
        return self.bank.forecast(state.inference)

    @lru_cache(maxsize=CACHE_SIZE)
    def leaf_risk(self, state: ContinuousSpeculativeState) -> float:
        forecast = self.forecast(state)
        weights = state.weights()
        active = weights > 1e-14
        active_weights = weights[active]
        target_mean = active_weights @ self._pair_targets[active]
        expected_squared_norm = float(active_weights @ self._pair_target_squared_norm[active])
        return expected_squared_norm - 2.0 * float(target_mean @ forecast) / len(forecast) + float(
            forecast @ forecast
        ) / len(forecast)

    def candidate_actions(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        width: int,
    ) -> tuple[int, ...]:
        by_group: dict[str, tuple[float, int]] = {}
        weights = state.weights()
        log_means = self._pair_log_means
        for action in available:
            group = self.bank.action_groups[action]
            if group is None:
                continue
            mean = float(weights @ log_means[:, action])
            score = float(weights @ (log_means[:, action] - mean) ** 2)
            current = by_group.get(group)
            if current is None or score > current[0] + TIE_TOLERANCE or (
                abs(score - current[0]) <= TIE_TOLERANCE and action < current[1]
            ):
                by_group[group] = (score, action)
        ranked = sorted(by_group.values(), key=lambda item: (-item[0], item[1]))
        return tuple(action for _, action in ranked[:width])

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_action_value(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
    ) -> float:
        remainder = tuple(item for item in available if item != action)
        value = 0.0
        for branch in self.predictive_branches(
            state, action, self.branch_count(remaining)
        ):
            child = self.transition(state, action, branch.observation)
            child_value = (
                self.leaf_risk(child)
                if level == 1 or remaining == 1
                else self.policy_value(child, remainder, remaining - 1, level - 1)
            )
            value += branch.probability * child_value
        return value

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_action(
        self,
        state: ContinuousSpeculativeState,
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

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_value(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.leaf_risk(state)
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        return sum(
            branch.probability
            * self.policy_value(
                self.transition(state, action, branch.observation),
                remainder,
                remaining - 1,
                level,
            )
            for branch in self.predictive_branches(
                state, action, self.branch_count(remaining)
            )
        )

    def truth_observation_branches(
        self,
        truth_position: int,
        action: int,
    ) -> tuple[PredictiveBranch, ...]:
        mean = float(self.truth_observation_means[truth_position, action])
        sigma = float(self.truth_observation_sigmas[truth_position, action])
        if mean == 0.0:
            return (PredictiveBranch(1.0, 0.0),)
        return tuple(
            PredictiveBranch(float(weight), max(0.0, mean + sigma * float(node)))
            for node, weight in zip(self.truth_nodes, self.truth_weights, strict=True)
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def truth_value(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        truth_position: int,
    ) -> float:
        if remaining <= 0 or not available:
            return float(
                np.mean((self.forecast(state) - self.truth_target_features[truth_position]) ** 2)
            )
        action = self.policy_action(state, available, remaining, level)
        remainder = tuple(item for item in available if item != action)
        return sum(
            branch.probability
            * self.truth_value(
                self.transition(state, action, branch.observation),
                remainder,
                remaining - 1,
                level,
                truth_position,
            )
            for branch in self.truth_observation_branches(truth_position, action)
        )

    def truth_value_after_action(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        truth_position: int,
        action: int,
    ) -> float:
        remainder = tuple(item for item in available if item != action)
        return sum(
            branch.probability
            * self.truth_value(
                self.transition(state, action, branch.observation),
                remainder,
                remaining - 1,
                level,
                truth_position,
            )
            for branch in self.truth_observation_branches(truth_position, action)
        )

    def clear_runtime_caches(self, *, clear_policy_table: bool = False) -> None:
        del clear_policy_table
        self.predictive_branches.cache_clear()
        self.transition.cache_clear()
        self.forecast.cache_clear()
        self.leaf_risk.cache_clear()
        self.policy_action_value.cache_clear()
        self.policy_action.cache_clear()
        self.policy_value.cache_clear()
        self.truth_value.cache_clear()
        self.bank._state.cache_clear()
        self.bank._stacked_particle_log_likelihoods.cache_clear()
        self.bank._all_structure_log_evidences.cache_clear()

    def monte_carlo_one_step_value(
        self,
        state: ContinuousSpeculativeState,
        action: int,
        *,
        num_samples: int,
        seed: int,
    ) -> tuple[float, float]:
        if num_samples <= 1:
            raise ValueError("Monte Carlo sample count must exceed one")
        rng = np.random.default_rng(seed)
        pair_indices = rng.choice(len(self.speculative_pairs), size=num_samples, p=state.weights())
        means = self._pair_raw_means[pair_indices, action]
        sigmas = self.bank.raw_sigmas_for_means(means)
        observations = np.maximum(rng.normal(means, sigmas), 0.0)
        values = np.asarray(
            [self.leaf_risk(self.transition(state, action, float(item))) for item in observations],
            dtype=float,
        )
        return float(values.mean()), float(values.std(ddof=1) / math.sqrt(num_samples))

    def evaluate_policy_level(
        self,
        level: int,
        *,
        execution_budget: int = 4,
        evaluate_execution_model_risk: bool = True,
        clear_before_truth: bool = False,
    ) -> dict[str, Any]:
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        root_action = self.policy_action(state, available, execution_budget, level)
        root_improvement_value = self.policy_action_value(
            state,
            available,
            execution_budget,
            level,
            root_action,
        )
        execution_model_risk = (
            self.policy_value(state, available, execution_budget, level)
            if evaluate_execution_model_risk
            else None
        )
        if clear_before_truth:
            self.clear_runtime_caches()
        truth_losses = [
            self.truth_value_after_action(
                state,
                available,
                execution_budget,
                level,
                truth_position,
                root_action,
            )
            for truth_position in range(len(self.speculative_models))
        ]
        truth_mean = float(np.mean(truth_losses))
        return {
            "policy_level": int(level),
            "root_action_index": int(root_action),
            "root_action": self.bank.action_names[root_action],
            "root_improvement_value": float(root_improvement_value),
            "planned_particle_risk": (
                float(execution_model_risk) if execution_model_risk is not None else None
            ),
            "expected_truth_mse": truth_mean,
            "expected_truth_rmsle": math.sqrt(max(truth_mean, 0.0)),
            "truth_losses": [float(item) for item in truth_losses],
        }


class ScenarioPolicyLadderPlanner(ContinuousPolicyLadderPlanner):
    """Common-random-number rollout evaluator for the continuous policy ladder."""

    def __init__(
        self,
        *args: Any,
        scenario_counts_by_remaining: Sequence[int] = (3, 6, 12, 96),
        action_widths_by_remaining: Sequence[int] = (2, 3, 4, 6),
        policy_improvement_replicates: int = 3,
        minimum_improvement_fraction: float = 0.05,
        use_policy_abstraction: bool = True,
        policy_signature_mode: str = "predictive",
        policy_signature_top_k: int = 3,
        policy_weight_resolution: float = 0.1,
        policy_predictive_resolution: float = 0.25,
        policy_risk_resolution: float = 0.1,
        policy_table_max_size: int = 10_000,
        freeze_predecessor_on_miss: bool = False,
        policy_cover_paths_by_level: Sequence[int] = (0,),
        policy_nearest_max_distance: float = 0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        counts = tuple(int(item) for item in scenario_counts_by_remaining)
        widths = tuple(int(item) for item in action_widths_by_remaining)
        cover_paths = tuple(int(item) for item in policy_cover_paths_by_level)
        if not counts or any(item <= 1 for item in counts):
            raise ValueError("scenario counts must all exceed one")
        if not widths or any(item <= 0 for item in widths):
            raise ValueError("scenario action widths must be positive")
        if not cover_paths or any(item < 0 for item in cover_paths):
            raise ValueError("policy cover path counts must be nonnegative")
        if policy_improvement_replicates <= 0 or not 0 <= minimum_improvement_fraction < 1:
            raise ValueError("conservative policy-improvement settings are invalid")
        if policy_signature_mode not in {"model_mass", "predictive"}:
            raise ValueError("policy signature mode is invalid")
        if (
            policy_signature_top_k <= 0
            or not 0 < policy_weight_resolution <= 1
            or policy_predictive_resolution <= 0
            or policy_risk_resolution <= 0
            or policy_table_max_size <= 0
            or policy_nearest_max_distance <= 0
        ):
            raise ValueError("policy abstraction settings are invalid")
        self.scenario_counts_by_remaining = counts
        self.action_widths_by_remaining = widths
        self.policy_improvement_replicates = int(policy_improvement_replicates)
        self.minimum_improvement_fraction = float(minimum_improvement_fraction)
        self.use_policy_abstraction = bool(use_policy_abstraction)
        self.policy_signature_mode = str(policy_signature_mode)
        self.policy_signature_top_k = int(policy_signature_top_k)
        self.policy_weight_resolution = float(policy_weight_resolution)
        self.policy_predictive_resolution = float(policy_predictive_resolution)
        self.policy_risk_resolution = float(policy_risk_resolution)
        self.policy_table_max_size = int(policy_table_max_size)
        self.freeze_predecessor_on_miss = bool(freeze_predecessor_on_miss)
        self.policy_cover_paths_by_level = cover_paths
        self.policy_nearest_max_distance = float(policy_nearest_max_distance)
        self._frozen_policy_level = 0
        self._signature_log_center = np.mean(self._pair_log_means, axis=0)
        self._signature_log_scale = np.maximum(np.std(self._pair_log_means, axis=0), 1e-6)
        self._initial_leaf_risk = max(self.leaf_risk(self.initial_state()), 1e-12)
        self.abstract_policy_actions: OrderedDict[tuple[Any, ...], int] = OrderedDict()
        self.abstract_policy_neighbors: dict[
            tuple[int, int],
            list[tuple[tuple[Any, ...], tuple[float, ...], int]],
        ] = {}
        self.abstract_policy_hits = 0
        self.abstract_policy_misses = 0
        self.abstract_policy_evictions = 0
        self.abstract_policy_predecessor_fallbacks = 0
        self.abstract_policy_nearest_hits = 0
        self.abstract_policy_nearest_distance_sum = 0.0
        self.abstract_policy_nearest_distance_max = 0.0

    @staticmethod
    def _scheduled_value(values: tuple[int, ...], remaining: int) -> int:
        if remaining <= 0:
            raise ValueError("remaining budget must be positive")
        return values[min(remaining, len(values)) - 1]

    def scenario_count(self, remaining: int) -> int:
        return self._scheduled_value(self.scenario_counts_by_remaining, remaining)

    def action_width(self, remaining: int) -> int:
        return self._scheduled_value(self.action_widths_by_remaining, remaining)

    def policy_cover_path_count(self, level: int) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        return self.policy_cover_paths_by_level[
            min(level, len(self.policy_cover_paths_by_level)) - 1
        ]

    def policy_signature(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> tuple[Any, ...]:
        resolution = self.policy_weight_resolution
        inference = state.inference
        weights = state.weights()
        ess = 1.0 / float(weights @ weights)
        if self.policy_signature_mode == "model_mass":
            represented = sorted(
                zip(inference.represented_weight, inference.represented_models, strict=True),
                reverse=True,
            )[: self.policy_signature_top_k]
            inference_signature: tuple[Any, ...] = tuple(
                (int(model), int(round(float(weight) / resolution)))
                for weight, model in represented
            )
            model_mass = np.zeros(self.bank.num_models, dtype=float)
            for weight, (model, _) in zip(weights, self.speculative_pairs, strict=True):
                model_mass[model] += float(weight)
            top_models = np.argsort(-model_mass, kind="stable")[: self.policy_signature_top_k]
            speculative_signature: tuple[Any, ...] = tuple(
                (int(model), int(round(float(model_mass[model]) / resolution)))
                for model in top_models
                if model_mass[model] > 1e-14
            )
        else:
            candidates = self.candidate_actions(
                state,
                available,
                self.action_width(remaining),
            )
            predictive = []
            inference_predictive = []
            predictive_resolution = self.policy_predictive_resolution
            for action in candidates:
                values = self._pair_log_means[:, action]
                mean = float(weights @ values)
                standard_deviation = math.sqrt(max(float(weights @ (values - mean) ** 2), 0.0))
                predictive.append(
                    (
                        int(action),
                        int(
                            round(
                                (mean - float(self._signature_log_center[action]))
                                / float(self._signature_log_scale[action])
                                / predictive_resolution
                            )
                        ),
                        int(
                            round(
                                standard_deviation
                                / float(self._signature_log_scale[action])
                                / predictive_resolution
                            )
                        ),
                    )
                )
                inference_log_rate = math.log1p(
                    max(self.bank.predicted_rate(inference, action), 0.0)
                )
                inference_predictive.append(
                    (
                        int(action),
                        int(
                            round(
                                (
                                    inference_log_rate
                                    - float(self._signature_log_center[action])
                                )
                                / float(self._signature_log_scale[action])
                                / predictive_resolution
                            )
                        ),
                    )
                )
            inference_weights = np.asarray(inference.represented_weight, dtype=float)
            inference_ess = 1.0 / float(inference_weights @ inference_weights)
            inference_signature = (
                tuple(inference_predictive),
                int(round(math.log2(max(inference_ess, 1.0)))),
                len(inference.live),
                len(inference.reserve),
                min(len(inference.discovered), self.bank.live_cap + self.bank.reserve_cap),
            )
            speculative_signature = (
                tuple(predictive),
                int(
                    round(
                        math.log1p(max(self.leaf_risk(state), 0.0) / self._initial_leaf_risk)
                        / self.policy_risk_resolution
                    )
                ),
            )
        return (
            tuple(available),
            int(remaining),
            int(level),
            inference_signature,
            speculative_signature,
            int(round(inference.outside_mass / resolution)),
            int(round(math.log2(max(ess, 1.0)))),
            tuple(action for action, _ in inference.history[-2:]),
        )

    def _abstract_policy_action(self, signature: tuple[Any, ...]) -> int | None:
        action = self.abstract_policy_actions.get(signature)
        if action is not None:
            self.abstract_policy_actions.move_to_end(signature)
        return action

    def _remember_abstract_policy_action(
        self,
        signature: tuple[Any, ...],
        action: int,
        *,
        state: ContinuousSpeculativeState | None = None,
        available: tuple[int, ...] | None = None,
        remaining: int | None = None,
        level: int | None = None,
    ) -> None:
        existing = self.abstract_policy_actions.get(signature)
        if existing is not None:
            self._remove_policy_neighbor(signature)
        self.abstract_policy_actions[signature] = int(action)
        self.abstract_policy_actions.move_to_end(signature)
        if (
            self.freeze_predecessor_on_miss
            and state is not None
            and available is not None
            and remaining is not None
            and level is not None
        ):
            context = (int(remaining), int(level))
            self.abstract_policy_neighbors.setdefault(context, []).append(
                (
                    signature,
                    self.policy_feature_vector(state, available),
                    int(action),
                )
            )
        while len(self.abstract_policy_actions) > self.policy_table_max_size:
            evicted_signature, _ = self.abstract_policy_actions.popitem(last=False)
            self._remove_policy_neighbor(evicted_signature)
            self.abstract_policy_evictions += 1

    def _remove_policy_neighbor(self, signature: tuple[Any, ...]) -> None:
        empty = []
        for context, entries in self.abstract_policy_neighbors.items():
            retained = [entry for entry in entries if entry[0] != signature]
            if retained:
                self.abstract_policy_neighbors[context] = retained
            else:
                empty.append(context)
        for context in empty:
            del self.abstract_policy_neighbors[context]

    def policy_feature_vector(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
    ) -> tuple[float, ...]:
        weights = state.weights()
        available_set = set(available)
        features = []
        for action in range(self.bank.num_actions):
            active = 1.0 if action in available_set else 0.0
            values = self._pair_log_means[:, action]
            mean = float(weights @ values)
            standard_deviation = math.sqrt(
                max(float(weights @ (values - mean) ** 2), 0.0)
            )
            scale = float(self._signature_log_scale[action])
            speculative_mean = (
                mean - float(self._signature_log_center[action])
            ) / scale
            speculative_standard_deviation = standard_deviation / scale
            inference_mean = (
                math.log1p(max(self.bank.predicted_rate(state.inference, action), 0.0))
                - float(self._signature_log_center[action])
            ) / scale
            features.extend(
                (
                    active,
                    active * float(np.clip(speculative_mean, -8.0, 8.0)),
                    active * float(np.clip(speculative_standard_deviation, 0.0, 8.0)),
                    active * float(np.clip(inference_mean, -8.0, 8.0)),
                )
            )
        inference_weights = np.asarray(state.inference.represented_weight, dtype=float)
        inference_ess = 1.0 / float(inference_weights @ inference_weights)
        features.extend(
            (
                state.inference.outside_mass,
                math.log2(max(1.0 / float(weights @ weights), 1.0)) / 6.0,
                math.log2(max(inference_ess, 1.0)) / 5.0,
                math.log1p(max(self.leaf_risk(state), 0.0) / self._initial_leaf_risk),
            )
        )
        return tuple(features)

    def _nearest_abstract_policy_action(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> int | None:
        entries = self.abstract_policy_neighbors.get((remaining, level), ())
        usable = [entry for entry in entries if entry[2] in available]
        if not usable:
            return None
        target = np.asarray(self.policy_feature_vector(state, available), dtype=float)
        matrix = np.asarray([entry[1] for entry in usable], dtype=float)
        distances = np.sqrt(np.mean((matrix - target[None, :]) ** 2, axis=1))
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance > self.policy_nearest_max_distance:
            return None
        self.abstract_policy_nearest_hits += 1
        self.abstract_policy_nearest_distance_sum += distance
        self.abstract_policy_nearest_distance_max = max(
            self.abstract_policy_nearest_distance_max,
            distance,
        )
        return int(usable[best][2])

    def policy_diagnostics(self) -> dict[str, int | float | str | bool]:
        return {
            "signature_mode": self.policy_signature_mode,
            "freeze_predecessor_on_miss": self.freeze_predecessor_on_miss,
            "table_size": len(self.abstract_policy_actions),
            "table_max_size": self.policy_table_max_size,
            "hits": self.abstract_policy_hits,
            "misses": self.abstract_policy_misses,
            "evictions": self.abstract_policy_evictions,
            "predecessor_fallbacks": self.abstract_policy_predecessor_fallbacks,
            "nearest_hits": self.abstract_policy_nearest_hits,
            "nearest_mean_distance": (
                self.abstract_policy_nearest_distance_sum / self.abstract_policy_nearest_hits
                if self.abstract_policy_nearest_hits
                else 0.0
            ),
            "nearest_max_distance": self.abstract_policy_nearest_distance_max,
            "nearest_distance_limit": self.policy_nearest_max_distance,
        }

    @lru_cache(maxsize=CACHE_SIZE)
    def scenario_bank(
        self,
        state: ContinuousSpeculativeState,
        remaining: int,
        level: int,
        replicate: int,
    ) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
        count = self.scenario_count(remaining)
        key = proposal_key(
            "continuous-scenario-bank",
            state.inference,
            remaining,
            float(level) + replicate / 1000.0,
            self.seed + replicate,
        )
        rng = np.random.default_rng(int(key[:16], 16))
        offset = float(rng.random()) / count
        quantiles = offset + np.arange(count, dtype=float) / count
        cumulative = np.cumsum(state.weights())
        pairs = np.searchsorted(cumulative, quantiles, side="right")
        pairs = np.minimum(pairs, len(self.speculative_pairs) - 1)
        noises = rng.standard_normal((count, remaining))
        return (
            tuple(int(item) for item in pairs),
            tuple(tuple(float(value) for value in row) for row in noises),
        )

    def scenario_observation(self, pair: int, action: int, noise: float) -> float:
        log_rate = (
            float(self._pair_log_means[pair, action])
            + float(self._pair_log_sigmas[pair, action]) * float(noise)
        )
        return math.expm1(max(log_rate, 0.0))

    def simulate_policy_path(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        pair: int,
        noises: tuple[float, ...],
    ) -> float:
        current = state
        current_available = available
        for step in range(remaining):
            action = self.policy_action(
                current,
                current_available,
                remaining - step,
                level,
            )
            observation = self.scenario_observation(pair, action, noises[step])
            current = self.transition(current, action, observation)
            current_available = tuple(item for item in current_available if item != action)
        return self.leaf_risk(current)

    def populate_policy_cover(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> dict[str, int]:
        path_count = self.policy_cover_path_count(level)
        if path_count == 0:
            return {
                "path_count": 0,
                "visited_states": 0,
                "table_entries_added": 0,
            }
        pairs, noises = self.scenario_bank(state, remaining, level, 100 + level)
        selected_positions = np.floor(
            (np.arange(path_count, dtype=float) + 0.5) * len(pairs) / path_count
        ).astype(int)
        selected_positions = np.minimum(selected_positions, len(pairs) - 1)
        table_size_before = len(self.abstract_policy_actions)
        visited_states = 0
        for position in selected_positions:
            pair = pairs[int(position)]
            noise_path = noises[int(position)]
            current = state
            current_available = available
            for step in range(remaining):
                action = self.policy_action(
                    current,
                    current_available,
                    remaining - step,
                    level,
                )
                observation = self.scenario_observation(pair, action, noise_path[step])
                current = self.transition(current, action, observation)
                current_available = tuple(
                    item for item in current_available if item != action
                )
                visited_states += 1
        return {
            "path_count": path_count,
            "visited_states": visited_states,
            "table_entries_added": max(
                0,
                len(self.abstract_policy_actions) - table_size_before,
            ),
        }

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_action_value_replicate(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
        replicate: int,
    ) -> float:
        pairs, noises = self.scenario_bank(state, remaining, level, replicate)
        remainder = tuple(item for item in available if item != action)
        values = []
        for pair, noise_path in zip(pairs, noises, strict=True):
            observation = self.scenario_observation(pair, action, noise_path[0])
            child = self.transition(state, action, observation)
            if remaining == 1 or level == 1:
                value = self.leaf_risk(child)
            else:
                value = self.simulate_policy_path(
                    child,
                    remainder,
                    remaining - 1,
                    level - 1,
                    pair,
                    noise_path[1:],
                )
            values.append(value)
        return float(np.mean(values))

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_action_value(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
        action: int,
    ) -> float:
        replicate_count = 1 if level == 1 else self.policy_improvement_replicates
        return float(
            np.mean(
                [
                    self.policy_action_value_replicate(
                        state,
                        available,
                        remaining,
                        level,
                        action,
                        replicate,
                    )
                    for replicate in range(replicate_count)
                ]
            )
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_action(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> int:
        if level <= 0:
            raise ValueError("policy level must be positive")
        if remaining <= 0 or not available:
            return -1
        signature = self.policy_signature(state, available, remaining, level)
        abstract_action = (
            self._abstract_policy_action(signature) if self.use_policy_abstraction else None
        )
        if abstract_action is not None and abstract_action in available:
            self.abstract_policy_hits += 1
            return abstract_action
        self.abstract_policy_misses += 1
        if (
            self.use_policy_abstraction
            and self.freeze_predecessor_on_miss
            and 1 < level <= self._frozen_policy_level
        ):
            nearest_action = self._nearest_abstract_policy_action(
                state,
                available,
                remaining,
                level,
            )
            if nearest_action is not None:
                return nearest_action
            self.abstract_policy_predecessor_fallbacks += 1
            return self.policy_action(state, available, remaining, level - 1)
        candidates = list(self.candidate_actions(
            state,
            available,
            self.action_width(remaining),
        ))
        if level == 1:
            selected = min(
                (
                    self.policy_action_value(state, available, remaining, level, action),
                    action,
                )
                for action in candidates
            )[1]
            if self.use_policy_abstraction:
                self._remember_abstract_policy_action(
                    signature,
                    selected,
                    state=state,
                    available=available,
                    remaining=remaining,
                    level=level,
                )
            return selected

        predecessor = self.policy_action(state, available, remaining, level - 1)
        if predecessor not in candidates:
            candidates.append(predecessor)
        values = {
            action: tuple(
                self.policy_action_value_replicate(
                    state,
                    available,
                    remaining,
                    level,
                    action,
                    replicate,
                )
                for replicate in range(self.policy_improvement_replicates)
            )
            for action in candidates
        }
        challenger = min((float(np.mean(value)), action) for action, value in values.items())[1]
        if challenger == predecessor:
            selected = predecessor
        else:
            predecessor_values = values[predecessor]
            challenger_values = values[challenger]
            predecessor_mean = float(np.mean(predecessor_values))
            challenger_mean = float(np.mean(challenger_values))
            material = (
                predecessor_mean > 0
                and (predecessor_mean - challenger_mean) / predecessor_mean
                >= self.minimum_improvement_fraction
            )
            unanimous = all(
                challenger_value < predecessor_value - TIE_TOLERANCE
                for challenger_value, predecessor_value in zip(
                    challenger_values,
                    predecessor_values,
                    strict=True,
                )
            )
            selected = challenger if material and unanimous else predecessor
        if self.use_policy_abstraction:
            self._remember_abstract_policy_action(
                signature,
                selected,
                state=state,
                available=available,
                remaining=remaining,
                level=level,
            )
        return selected

    @lru_cache(maxsize=CACHE_SIZE)
    def policy_value(
        self,
        state: ContinuousSpeculativeState,
        available: tuple[int, ...],
        remaining: int,
        level: int,
    ) -> float:
        if remaining <= 0 or not available:
            return self.leaf_risk(state)
        pairs, noises = self.scenario_bank(state, remaining, level, 0)
        values = [
            self.simulate_policy_path(
                state,
                available,
                remaining,
                level,
                pair,
                noise_path,
            )
            for pair, noise_path in zip(pairs, noises, strict=True)
        ]
        return float(np.mean(values))

    def clear_runtime_caches(self, *, clear_policy_table: bool = False) -> None:
        self.scenario_bank.cache_clear()
        self.policy_action_value_replicate.cache_clear()
        super().clear_runtime_caches()
        if clear_policy_table:
            self.abstract_policy_actions.clear()
            self.abstract_policy_neighbors.clear()
            self.abstract_policy_hits = 0
            self.abstract_policy_misses = 0
            self.abstract_policy_evictions = 0
            self.abstract_policy_predecessor_fallbacks = 0
            self.abstract_policy_nearest_hits = 0
            self.abstract_policy_nearest_distance_sum = 0.0
            self.abstract_policy_nearest_distance_max = 0.0

    def evaluate_policy_level(
        self,
        level: int,
        *,
        execution_budget: int = 4,
        evaluate_execution_model_risk: bool = True,
        clear_before_truth: bool = False,
    ) -> dict[str, Any]:
        state = self.initial_state()
        available = tuple(range(self.bank.num_actions))
        prior_frozen_level = self._frozen_policy_level
        try:
            self._frozen_policy_level = max(0, level - 1)
            root_action = self.policy_action(state, available, execution_budget, level)
            root_improvement_value = self.policy_action_value(
                state,
                available,
                execution_budget,
                level,
                root_action,
            )
            execution_model_risk = (
                self.policy_value(state, available, execution_budget, level)
                if evaluate_execution_model_risk
                else None
            )
            policy_cover = self.populate_policy_cover(
                state,
                available,
                execution_budget,
                level,
            )
            if clear_before_truth:
                self.clear_runtime_caches()
            self._frozen_policy_level = level
            truth_losses = [
                self.truth_value_after_action(
                    state,
                    available,
                    execution_budget,
                    level,
                    truth_position,
                    root_action,
                )
                for truth_position in range(len(self.speculative_models))
            ]
            truth_mean = float(np.mean(truth_losses))
            return {
                "policy_level": int(level),
                "root_action_index": int(root_action),
                "root_action": self.bank.action_names[root_action],
                "root_improvement_value": float(root_improvement_value),
                "planned_particle_risk": (
                    float(execution_model_risk) if execution_model_risk is not None else None
                ),
                "expected_truth_mse": truth_mean,
                "expected_truth_rmsle": math.sqrt(max(truth_mean, 0.0)),
                "truth_losses": [float(item) for item in truth_losses],
                "policy_cover": policy_cover,
                "policy_abstraction": self.policy_diagnostics(),
            }
        finally:
            self._frozen_policy_level = prior_frozen_level
