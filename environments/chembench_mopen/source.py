"""Pinned-source boundary for mixed-version ChemBench M-open mechanics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


INPUT_ORDER = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
LINEAR_PARAMETER_NAMES = {
    "alpha",
    "beta",
    "n",
    "n_inh",
    "n_met",
    "pKa",
    "pKa1",
    "pKa2",
}


@dataclass(frozen=True)
class MixedVersionResponses:
    observation_means: np.ndarray
    target_log_rates: np.ndarray
    query_values: np.ndarray
    version_by_model: tuple[str, ...]
    version_map_sha256: str
    truth_indices: tuple[int, ...]


@dataclass(frozen=True)
class EmpiricalParameterResponses:
    particle_observation_means: tuple[np.ndarray, ...]
    particle_target_log_rates: tuple[np.ndarray, ...]
    candidate_versions: tuple[tuple[str, ...], ...]
    truth_observation_means: np.ndarray
    truth_target_log_rates: np.ndarray
    truth_indices: tuple[int, ...]
    query_values: np.ndarray
    version_plan_sha256: str


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_broadened_parameter_particles(
    parameter_states: Sequence[dict[str, Any]],
    *,
    num_particles: int,
    seed: int,
    expansion_factor: float = 1.5,
    minimum_log_span: float = math.log(2.0),
    minimum_linear_span: float = 0.5,
) -> tuple[dict[str, float], ...]:
    """Sample a deterministic broad prior from observed parameter versions only."""

    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if expansion_factor < 0 or minimum_log_span <= 0 or minimum_linear_span <= 0:
        raise ValueError("parameter prior span settings are invalid")
    if not parameter_states:
        raise ValueError("parameter states must be nonempty")
    keys = tuple(sorted(parameter_states[0]))
    if not keys or any(tuple(sorted(state)) != keys for state in parameter_states):
        raise ValueError("parameter states must have identical nonempty keys")
    values_by_key = {
        key: np.asarray([float(state[key]) for state in parameter_states], dtype=float)
        for key in keys
    }
    if any(not np.isfinite(values).all() for values in values_by_key.values()):
        raise ValueError("parameter states must be finite")

    rng = np.random.default_rng(seed)
    sampled: dict[str, np.ndarray] = {}
    for key in keys:
        values = values_by_key[key]
        use_linear = key in LINEAR_PARAMETER_NAMES
        if not use_linear and np.any(values <= 0):
            raise ValueError(f"log-scale parameter {key} must be positive")
        transformed = values if use_linear else np.log(values)
        observed_low = float(np.min(transformed))
        observed_high = float(np.max(transformed))
        minimum_span = minimum_linear_span if use_linear else minimum_log_span
        padding = expansion_factor * max(observed_high - observed_low, minimum_span)
        low = observed_low - padding
        high = observed_high + padding
        strata = (np.arange(num_particles, dtype=float) + rng.random(num_particles))
        strata /= num_particles
        rng.shuffle(strata)
        draws = low + strata * (high - low)
        if not use_linear:
            draws = np.exp(draws)
        else:
            draws = np.maximum(draws, np.finfo(float).eps)
        sampled[key] = draws

    if "pKa1" in sampled and "pKa2" in sampled:
        low = np.minimum(sampled["pKa1"], sampled["pKa2"])
        high = np.maximum(sampled["pKa1"], sampled["pKa2"])
        sampled["pKa1"] = low
        sampled["pKa2"] = np.maximum(high, low + 1e-6)
    return tuple(
        {key: float(sampled[key][particle]) for key in keys}
        for particle in range(num_particles)
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_all_finite(item) for item in value)
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _query_assays(
    seed: int,
    bounds: dict[str, tuple[float, float]],
    log_variables: Iterable[str],
    count: int,
) -> np.ndarray:
    log_variables = set(log_variables)
    rng = np.random.default_rng(seed)
    values = np.empty((count, len(INPUT_ORDER)), dtype=float)
    for column, name in enumerate(INPUT_ORDER):
        low, high = bounds[name]
        if name in log_variables:
            if low <= 0:
                raise ValueError(f"log-uniform lower bound is not positive for {name}")
            values[:, column] = np.exp(rng.uniform(math.log(low), math.log(high), count))
        else:
            values[:, column] = rng.uniform(low, high, count)
    return values


def _version_plan(
    source: Any,
    domains: Sequence[str],
    initial_names: Sequence[str],
    difficulty: str,
    initial_version: str,
    truth_version: str,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    initial = set(initial_names)
    if len(initial) != len(initial_names):
        raise ValueError("initial support names must be unique")
    if not initial.issubset(domains):
        raise ValueError("initial support is absent from active domains")
    versions: list[str] = []
    truth_indices: list[int] = []
    parameters: list[dict[str, Any]] = []
    for index, domain in enumerate(domains):
        version = initial_version if domain in initial else truth_version
        try:
            params = source._PARAMS[domain][difficulty][version]
        except KeyError as exc:
            raise ValueError(
                f"missing frozen parameter state for {domain}/{difficulty}/{version}"
            ) from exc
        if not _all_finite(params):
            raise ValueError(f"parameter state is empty or non-finite for {domain}")
        if domain not in source._RATE_FNS or not callable(source._RATE_FNS[domain]):
            raise ValueError(f"rate function is missing for {domain}")
        versions.append(version)
        parameters.append(params)
        if domain not in initial:
            truth_indices.append(index)
    if len(truth_indices) != len(domains) - len(initial):
        raise AssertionError("outside-support truth cohort has the wrong size")
    return tuple(versions), tuple(truth_indices), tuple(parameters)


def build_mixed_version_responses(
    source: Any,
    domains: Sequence[str],
    initial_names: Sequence[str],
    *,
    difficulty: str,
    initial_version: str,
    truth_version: str,
    query_seed: int,
    assays: Sequence[Any],
    num_queries: int = 1_000,
) -> MixedVersionResponses:
    """Preflight all versions before constructing any source response."""

    versions, truth_indices, parameters = _version_plan(
        source,
        domains,
        initial_names,
        difficulty,
        initial_version,
        truth_version,
    )
    if len(truth_indices) != 48:
        raise ValueError(f"expected 48 outside-support truths, got {len(truth_indices)}")
    query_values = _query_assays(
        query_seed,
        source.CHEM_INPUT_BOUNDS,
        source.CHEM_LOG_VARS,
        num_queries,
    )
    observation_means = np.empty((len(domains), len(assays)), dtype=float)
    target_log_rates = np.empty((len(domains), num_queries), dtype=float)
    for domain_index, domain in enumerate(domains):
        params = parameters[domain_index]
        rate_fn = source._RATE_FNS[domain]
        for action_index, assay in enumerate(assays):
            value = float(rate_fn(params, *assay.values))
            value *= float(source._secondary_effects(assay.values[5], assay.values[6]))
            observation_means[domain_index, action_index] = value
        for query_index, query in enumerate(query_values):
            value = float(rate_fn(params, *query))
            target_log_rates[domain_index, query_index] = math.log1p(max(value, 0.0))
    if not np.isfinite(observation_means).all() or not np.isfinite(target_log_rates).all():
        raise ValueError("mixed-version source response contains non-finite values")
    version_payload = {domain: versions[index] for index, domain in enumerate(domains)}
    version_sha = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MixedVersionResponses(
        observation_means=observation_means,
        target_log_rates=target_log_rates,
        query_values=query_values,
        version_by_model=versions,
        version_map_sha256=version_sha,
        truth_indices=truth_indices,
    )


def build_empirical_parameter_responses(
    source: Any,
    domains: Sequence[str],
    initial_names: Sequence[str],
    *,
    difficulty: str,
    initial_candidate_versions: Sequence[str],
    outside_candidate_versions: Sequence[str],
    truth_version: str,
    query_seed: int,
    assays: Sequence[Any],
    num_queries: int = 1_000,
    broadened_num_particles: int | None = None,
    broadened_seed: int = 0,
    broadened_expansion_factor: float = 1.5,
) -> EmpiricalParameterResponses:
    """Build structure-level parameter particles before opening truth responses."""

    initial = set(initial_names)
    if len(initial) != len(initial_names) or not initial.issubset(domains):
        raise ValueError("initial support is invalid")
    initial_versions = tuple(dict.fromkeys(str(item) for item in initial_candidate_versions))
    outside_versions = tuple(dict.fromkeys(str(item) for item in outside_candidate_versions))
    if not initial_versions or not outside_versions or truth_version in outside_versions:
        raise ValueError("candidate and truth version plans must be nonempty and disjoint")

    candidate_versions: list[tuple[str, ...]] = []
    candidate_parameters: list[tuple[dict[str, Any], ...]] = []
    truth_parameters: list[dict[str, Any]] = []
    truth_indices: list[int] = []
    for index, domain in enumerate(domains):
        versions = initial_versions if domain in initial else outside_versions
        parameters = []
        for version in versions:
            try:
                params = source._PARAMS[domain][difficulty][version]
            except KeyError as exc:
                raise ValueError(
                    f"missing candidate parameter state for {domain}/{difficulty}/{version}"
                ) from exc
            if not _all_finite(params):
                raise ValueError(f"candidate parameter state is invalid for {domain}/{version}")
            parameters.append(params)
        if domain not in source._RATE_FNS or not callable(source._RATE_FNS[domain]):
            raise ValueError(f"rate function is missing for {domain}")
        candidate_versions.append(versions)
        if broadened_num_particles is None:
            candidate_parameters.append(tuple(parameters))
        else:
            candidate_parameters.append(
                sample_broadened_parameter_particles(
                    parameters,
                    num_particles=broadened_num_particles,
                    seed=_stable_seed(broadened_seed, difficulty, domain),
                    expansion_factor=broadened_expansion_factor,
                )
            )
        if domain not in initial:
            try:
                truth_params = source._PARAMS[domain][difficulty][truth_version]
            except KeyError as exc:
                raise ValueError(
                    f"missing truth parameter state for {domain}/{difficulty}/{truth_version}"
                ) from exc
            if not _all_finite(truth_params):
                raise ValueError(f"truth parameter state is invalid for {domain}/{truth_version}")
            truth_indices.append(index)
            truth_parameters.append(truth_params)
    if len(truth_indices) != 48:
        raise ValueError(f"expected 48 outside-support truths, got {len(truth_indices)}")

    query_values = _query_assays(
        query_seed,
        source.CHEM_INPUT_BOUNDS,
        source.CHEM_LOG_VARS,
        num_queries,
    )

    def evaluate(domain: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        rate_fn = source._RATE_FNS[domain]
        means = np.empty(len(assays), dtype=float)
        targets = np.empty(num_queries, dtype=float)
        for action_index, assay in enumerate(assays):
            value = float(rate_fn(params, *assay.values))
            value *= float(source._secondary_effects(assay.values[5], assay.values[6]))
            means[action_index] = value
        for query_index, query in enumerate(query_values):
            value = float(rate_fn(params, *query))
            targets[query_index] = math.log1p(max(value, 0.0))
        return means, targets

    particle_means: list[np.ndarray] = []
    particle_targets: list[np.ndarray] = []
    truth_means = np.empty((len(truth_indices), len(assays)), dtype=float)
    truth_targets = np.empty((len(truth_indices), num_queries), dtype=float)
    truth_position = 0
    for model_index, domain in enumerate(domains):
        model_means = []
        model_targets = []
        for params in candidate_parameters[model_index]:
            means, targets = evaluate(domain, params)
            model_means.append(means)
            model_targets.append(targets)
        particle_means.append(np.asarray(model_means, dtype=float))
        particle_targets.append(np.asarray(model_targets, dtype=float))
        if domain not in initial:
            means, targets = evaluate(domain, truth_parameters[truth_position])
            truth_means[truth_position] = means
            truth_targets[truth_position] = targets
            truth_position += 1

    arrays = [*particle_means, *particle_targets, truth_means, truth_targets]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("empirical parameter response contains non-finite values")
    version_payload = {
        "broadened_prior": {
            "num_particles": broadened_num_particles,
            "seed": broadened_seed if broadened_num_particles is not None else None,
            "expansion_factor": (
                broadened_expansion_factor if broadened_num_particles is not None else None
            ),
        },
        "models": {
            domain: {
                "candidate": list(candidate_versions[index]),
                "candidate_parameters": candidate_parameters[index],
                "truth": truth_version if domain not in initial else None,
            }
            for index, domain in enumerate(domains)
        },
    }
    version_sha = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EmpiricalParameterResponses(
        particle_observation_means=tuple(particle_means),
        particle_target_log_rates=tuple(particle_targets),
        candidate_versions=tuple(candidate_versions),
        truth_observation_means=truth_means,
        truth_target_log_rates=truth_targets,
        truth_indices=tuple(truth_indices),
        query_values=query_values,
        version_plan_sha256=version_sha,
    )
