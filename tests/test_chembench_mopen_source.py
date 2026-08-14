from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from environments.chembench_mopen.source import build_mixed_version_responses


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
