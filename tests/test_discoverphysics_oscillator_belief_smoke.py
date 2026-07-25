from __future__ import annotations

import json
import math

import pytest

from scripts.discoverphysics_oscillator_belief_smoke import (
    compile_action,
    entropy,
    parse_refresh,
    parse_scorer,
    parse_tree,
)


def _action(index: int) -> dict[str, str]:
    return {
        "separation": ("near", "medium", "far")[index % 3],
        "initial_motion": (
            "rest",
            "tangential_slow",
            "radial_outward",
        )[index % 3],
        "start_phase": (
            "phase_0",
            "phase_1",
            "phase_2",
            "phase_3",
        )[index % 4],
        "source_strength": ("low", "high")[index % 2],
        "probe_inertia": ("light", "heavy")[(index // 2) % 2],
    }


def _tree() -> dict:
    return {
        "prior": {
            "hypotheses": [f"law {index}" for index in range(5)],
            "probabilities": [0.2] * 5,
        },
        "candidates": [
            {
                "id": f"R{root}",
                "action": _action(root),
                "branches": [
                    {
                        "probability": 0.5,
                        "observation": f"root {root} branch {branch}",
                        "posterior_probabilities": (
                            [0.96, 0.01, 0.01, 0.01, 0.01]
                            if root == 0
                            else [0.4, 0.3, 0.1, 0.1, 0.1]
                        ),
                    }
                    for branch in range(2)
                ],
            }
            for root in range(4)
        ],
    }


def test_tree_parser_scores_immediate_eig_and_compiles_schedule() -> None:
    tree = parse_tree(json.dumps(_tree()))

    assert math.isclose(entropy([0.5, 0.5]), math.log(2))
    assert tree["candidates"][0]["immediate_eig"] > (
        tree["candidates"][1]["immediate_eig"]
    )
    for candidate in tree["candidates"]:
        experiment = candidate["compiled_experiment"]
        assert experiment["measurement_times"][-1] == 5.0
        assert len(experiment["measurement_times"]) == 10


def test_tree_parser_rejects_duplicate_roots() -> None:
    value = _tree()
    value["candidates"][1]["action"] = value["candidates"][0]["action"]
    with pytest.raises(ValueError, match="duplicated"):
        parse_tree(json.dumps(value))


def test_refresh_parser_requires_new_legal_continuation() -> None:
    root = _action(0)
    value = {
        "hypotheses": [f"fresh law {index}" for index in range(5)],
        "probabilities": [0.2] * 5,
        "coverage_probability": 0.7,
        "continuation_action": _action(1),
        "expected_learning": "Tests whether the force changes with phase.",
    }
    parsed = parse_refresh(
        json.dumps(value), root_action=root, label="refresh"
    )
    assert parsed["coverage_probability"] == 0.7
    assert parsed["compiled_continuation"]["measurement_times"][-1] == 5.0

    value["continuation_action"] = root
    with pytest.raises(ValueError, match="repeats"):
        parse_refresh(
            json.dumps(value), root_action=root, label="refresh"
        )


def test_scorer_parser_is_exact_and_bounded() -> None:
    assert parse_scorer('{"root_scores":[20,40,80,60]}') == {
        "root_scores": [20, 40, 80, 60]
    }
    with pytest.raises(ValueError, match="integers"):
        parse_scorer('{"root_scores":[20,40,101,60]}')
    with pytest.raises(ValueError, match="trailing"):
        parse_scorer('{"root_scores":[20,40,80,60]}}')


def test_compiler_phase_and_physical_factors_are_deterministic() -> None:
    experiment = compile_action(
        {
            "separation": "far",
            "initial_motion": "tangential_slow",
            "start_phase": "phase_3",
            "source_strength": "high",
            "probe_inertia": "light",
        }
    )
    assert experiment["p1"] == 2.0
    assert experiment["p2"] == 0.5
    assert experiment["pos2"] == [5.0, 0.0]
    assert experiment["velocity2"] == [0.0, 0.25]
    assert experiment["start_time"] == 3.0
