import json

import numpy as np
import pytest

from environments.location_finding.continuous_strategy import (
    ContinuousStrategyParseError,
    copex_signal,
    execute_continuous_step,
    parse_continuous_strategy_cell,
    score_continuous_strategy,
    update_copex_belief,
)


def _cell() -> str:
    return json.dumps(
        {
            "strategies": [
                {
                    "name": "diagonal scout",
                    "description": "Move north-east, then approach the leading mode.",
                    "steps": [
                        {"kind": "vector", "dx": 0.1, "dy": 0.1},
                        {"kind": "toward_rank", "rank": 0},
                    ],
                },
                {
                    "name": "bisect",
                    "description": "Probe between the two leading modes, then follow the mean.",
                    "steps": [
                        {"kind": "midpoint_ranks", "ranks": [0, 1]},
                        {"kind": "toward_mean"},
                    ],
                },
            ]
        }
    )


def test_continuous_strategy_parser_is_strict_and_complete() -> None:
    strategies = parse_continuous_strategy_cell(_cell(), expected_count=2, horizon=2)
    assert len(strategies) == 2
    with pytest.raises(ContinuousStrategyParseError, match="exactly 2 steps"):
        payload = json.loads(_cell())
        payload["strategies"][0]["steps"].pop()
        parse_continuous_strategy_cell(json.dumps(payload), expected_count=2, horizon=2)
    with pytest.raises(ContinuousStrategyParseError, match=r"max\(\|dx\|"):
        payload = json.loads(_cell())
        payload["strategies"][0]["steps"][0]["dx"] = 0.2
        parse_continuous_strategy_cell(json.dumps(payload), expected_count=2, horizon=2)


def test_executor_satisfies_box_and_transition_constraints() -> None:
    particles = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    probabilities = np.asarray([0.2, 0.8])
    strategy = parse_continuous_strategy_cell(_cell(), expected_count=2, horizon=2)[0]
    action = execute_continuous_step(
        strategy.steps[0],
        position=np.asarray([0.95, 0.95]),
        particles=particles,
        probabilities=probabilities,
    )
    assert np.allclose(action, [1.0, 1.0])
    assert np.max(np.abs(action - np.asarray([0.95, 0.95]))) <= 0.1


def test_analytic_likelihood_update_favors_matching_particle() -> None:
    particles = np.asarray([[0.2, 0.2], [0.8, 0.8]])
    probabilities = np.asarray([0.5, 0.5])
    query = np.asarray([0.2, 0.2])
    posterior = update_copex_belief(
        particles, probabilities, query, copex_signal(particles[0], query)
    )
    assert posterior[0] > posterior[1]
    assert np.isclose(np.sum(posterior), 1.0)


def test_strategy_score_is_deterministic_finite_and_constraint_safe() -> None:
    particles = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]])
    probabilities = np.full(4, 0.25)
    strategy = parse_continuous_strategy_cell(_cell(), expected_count=2, horizon=2)[0]
    uniforms = np.asarray([0.1, 0.4, 0.7, 0.9])
    noise = np.zeros((4, 2))
    first = score_continuous_strategy(
        strategy,
        position=np.asarray([0.5, 0.5]),
        particles=particles,
        probabilities=probabilities,
        truth_uniforms=uniforms,
        noise_zs=noise,
    )
    second = score_continuous_strategy(
        strategy,
        position=np.asarray([0.5, 0.5]),
        particles=particles,
        probabilities=probabilities,
        truth_uniforms=uniforms,
        noise_zs=noise,
    )
    assert first == second
    assert np.isfinite(first.total_eig)
    assert first.scorer_units == 8
    assert np.max(np.abs(np.asarray(first.root_action) - np.asarray([0.5, 0.5]))) <= 0.1
