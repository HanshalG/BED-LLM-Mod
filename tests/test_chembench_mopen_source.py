from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from environments.chembench_mopen.source import (
    build_empirical_parameter_responses,
    build_mixed_version_responses,
    sample_broadened_parameter_particles,
)


INITIAL = tuple(f"simple_{index}" for index in range(9))
OUTSIDE = tuple(f"outside_{index}" for index in range(48))
DOMAINS = INITIAL + OUTSIDE


@dataclass(frozen=True)
class _Assay:
    values: tuple[float, float, float, float, float, float, float]


def _source(counter: dict[str, int]) -> SimpleNamespace:
    params = {}
    functions = {}
    for index, name in enumerate(DOMAINS):
        version = "v2" if name in INITIAL else "v3"
        params[name] = {"easy": {version: {"scale": float(index + 1)}}}

        def rate_fn(p, C_A, C_I, C_B, C_P, Enz, T, pH):
            del C_I, C_B, C_P, Enz, T, pH
            counter["calls"] += 1
            return p["scale"] * C_A

        functions[name] = rate_fn
    return SimpleNamespace(
        _PARAMS=params,
        _RATE_FNS=functions,
        CHEM_INPUT_BOUNDS={
            "C_A": (0.01, 100.0),
            "C_I": (0.0, 50.0),
            "C_B": (0.01, 100.0),
            "C_P": (0.0, 20.0),
            "Enz": (0.01, 10.0),
            "T": (278.0, 368.0),
            "pH": (4.0, 10.0),
        },
        CHEM_LOG_VARS={"C_A", "C_B", "Enz"},
        _secondary_effects=lambda temperature, ph: 1.0,
    )


def test_mixed_version_source_uses_only_outside_truths() -> None:
    counter = {"calls": 0}
    source = _source(counter)
    result = build_mixed_version_responses(
        source,
        DOMAINS,
        INITIAL,
        difficulty="easy",
        initial_version="v2",
        truth_version="v3",
        query_seed=7,
        assays=(_Assay((1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0)),),
        num_queries=3,
    )
    assert result.observation_means.shape == (57, 1)
    assert result.target_log_rates.shape == (57, 3)
    assert result.truth_indices == tuple(range(9, 57))
    assert result.version_by_model == ("v2",) * 9 + ("v3",) * 48
    assert np.isfinite(result.target_log_rates).all()
    assert counter["calls"] == 57 * 4


def test_mixed_version_preflight_fails_before_any_rate_call() -> None:
    counter = {"calls": 0}
    source = _source(counter)
    del source._PARAMS[OUTSIDE[0]]["easy"]["v3"]
    with pytest.raises(ValueError, match="missing frozen parameter state"):
        build_mixed_version_responses(
            source,
            DOMAINS,
            INITIAL,
            difficulty="easy",
            initial_version="v2",
            truth_version="v3",
            query_seed=7,
            assays=(_Assay((1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0)),),
            num_queries=3,
        )
    assert counter["calls"] == 0


def _empirical_source(counter: dict[str, int]) -> SimpleNamespace:
    source = _source(counter)
    for index, name in enumerate(DOMAINS):
        versions = ("v0", "v1", "v2") if name in INITIAL else ("v0", "v1", "v2", "v3", "v4")
        source._PARAMS[name] = {
            "easy": {
                version: {"scale": float(index + 1) * (1.0 + 0.05 * version_index)}
                for version_index, version in enumerate(versions)
            }
        }
    return source


def test_empirical_parameter_source_separates_candidate_and_truth_versions() -> None:
    counter = {"calls": 0}
    source = _empirical_source(counter)
    result = build_empirical_parameter_responses(
        source,
        DOMAINS,
        INITIAL,
        difficulty="easy",
        initial_candidate_versions=("v0", "v1", "v2"),
        outside_candidate_versions=("v0", "v1", "v2", "v3"),
        truth_version="v4",
        query_seed=8,
        assays=(_Assay((1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0)),),
        num_queries=2,
    )
    assert result.truth_indices == tuple(range(9, 57))
    assert result.candidate_versions[:9] == (("v0", "v1", "v2"),) * 9
    assert result.candidate_versions[9:] == (("v0", "v1", "v2", "v3"),) * 48
    assert result.truth_observation_means.shape == (48, 1)
    assert result.truth_target_log_rates.shape == (48, 2)
    assert counter["calls"] == 9 * 3 * 3 + 48 * 5 * 3


def test_empirical_parameter_preflight_fails_before_truth_response() -> None:
    counter = {"calls": 0}
    source = _empirical_source(counter)
    del source._PARAMS[OUTSIDE[-1]]["easy"]["v4"]
    with pytest.raises(ValueError, match="missing truth parameter state"):
        build_empirical_parameter_responses(
            source,
            DOMAINS,
            INITIAL,
            difficulty="easy",
            initial_candidate_versions=("v0", "v1", "v2"),
            outside_candidate_versions=("v0", "v1", "v2", "v3"),
            truth_version="v4",
            query_seed=8,
            assays=(_Assay((1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0)),),
            num_queries=2,
        )
    assert counter["calls"] == 0


def test_broadened_parameter_particles_are_deterministic_and_valid() -> None:
    states = (
        {"kcat": 1.0, "n": 1.0, "pKa1": 5.0, "pKa2": 8.0},
        {"kcat": 2.0, "n": 2.0, "pKa1": 6.0, "pKa2": 9.0},
    )
    first = sample_broadened_parameter_particles(states, num_particles=16, seed=17)
    second = sample_broadened_parameter_particles(states, num_particles=16, seed=17)
    assert first == second
    assert len(first) == 16
    assert all(item["kcat"] > 0 and item["n"] > 0 for item in first)
    assert all(item["pKa1"] < item["pKa2"] for item in first)
    assert min(item["kcat"] for item in first) < 1.0
    assert max(item["kcat"] for item in first) > 2.0


def test_broadened_empirical_prior_does_not_depend_on_truth_values() -> None:
    counter = {"calls": 0}
    source = _empirical_source(counter)
    kwargs = dict(
        source=source,
        domains=DOMAINS,
        initial_names=INITIAL,
        difficulty="easy",
        initial_candidate_versions=("v0", "v1", "v2"),
        outside_candidate_versions=("v0", "v1", "v2", "v3"),
        truth_version="v4",
        query_seed=8,
        assays=(_Assay((1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0)),),
        num_queries=2,
        broadened_num_particles=8,
        broadened_seed=19,
        broadened_expansion_factor=1.5,
    )
    first = build_empirical_parameter_responses(**kwargs)
    for name in OUTSIDE:
        source._PARAMS[name]["easy"]["v4"]["scale"] *= 10.0
    second = build_empirical_parameter_responses(**kwargs)
    assert first.version_plan_sha256 == second.version_plan_sha256
    for first_means, second_means in zip(
        first.particle_observation_means,
        second.particle_observation_means,
        strict=True,
    ):
        np.testing.assert_array_equal(first_means, second_means)
    assert not np.array_equal(first.truth_observation_means, second.truth_observation_means)
