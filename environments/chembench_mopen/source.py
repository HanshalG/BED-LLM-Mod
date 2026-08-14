"""Pinned-source boundary for mixed-version ChemBench M-open mechanics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


INPUT_ORDER = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")


@dataclass(frozen=True)
class MixedVersionResponses:
    observation_means: np.ndarray
    target_log_rates: np.ndarray
    query_values: np.ndarray
    version_by_model: tuple[str, ...]
    version_map_sha256: str
    truth_indices: tuple[int, ...]


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
