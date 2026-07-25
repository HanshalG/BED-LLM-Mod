from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.discoverphysics_llm_bed import (
    concept_score_extra_dimensions,
    entropy,
    parse_final,
    parse_tree,
    select_candidate,
    validate_experiment,
)


def _belief(prefix: str, probabilities: list[float]) -> dict:
    return {
        "hypotheses": [f"{prefix} hypothesis {index}" for index in range(5)],
        "probabilities": probabilities,
    }


def _experiment(radius: float, speed: float = 0.0) -> dict:
    return {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [radius, 0.0],
        "velocity2": [0.0, speed],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    }


def _tree() -> dict:
    candidates = []
    for candidate_index in range(4):
        branches = []
        for branch_index in range(2):
            if candidate_index == 0:
                posterior = [0.96, 0.01, 0.01, 0.01, 0.01]
                terminal = [0.96, 0.01, 0.01, 0.01, 0.01]
            elif candidate_index == 1:
                posterior = [0.4, 0.3, 0.1, 0.1, 0.1]
                terminal = [0.99, 0.0025, 0.0025, 0.0025, 0.0025]
            else:
                posterior = [0.4, 0.2, 0.2, 0.1, 0.1]
                terminal = [0.5, 0.2, 0.1, 0.1, 0.1]
            branches.append(
                {
                    "probability": 0.5,
                    "observation": f"outcome {branch_index}",
                    "posterior_probabilities": posterior,
                    "continuation_experiment": _experiment(
                        5.0 + candidate_index, speed=0.1 + branch_index
                    ),
                    "terminal_belief": _belief(
                        f"c{candidate_index}b{branch_index}", terminal
                    ),
                }
            )
        candidates.append(
            {
                "id": f"E{candidate_index}",
                "experiment": _experiment(1.0 + candidate_index),
                "branches": branches,
            }
        )
    return {
        "prior": _belief("prior", [0.2] * 5),
        "candidates": candidates,
    }


def test_entropy_and_matched_tree_select_different_objectives() -> None:
    parsed = parse_tree(json.dumps(_tree()))
    assert math.isclose(entropy([0.5, 0.5]), math.log(2))
    assert select_candidate(parsed, "myopic") == 0
    assert select_candidate(parsed, "nonmyopic") == 1
    assert (
        parsed["candidates"][1]["immediate_eig"]
        < parsed["candidates"][0]["immediate_eig"]
    )
    assert (
        parsed["candidates"][1]["terminal_eig"]
        > parsed["candidates"][0]["terminal_eig"]
    )


def test_tree_parser_rejects_duplicate_root_experiments() -> None:
    tree = _tree()
    tree["candidates"][1]["experiment"] = tree["candidates"][0]["experiment"]
    with pytest.raises(ValueError, match="duplicated"):
        parse_tree(json.dumps(tree))


def test_experiment_validation_rejects_singular_and_short_measurement_plan() -> None:
    singular = _experiment(0.1)
    with pytest.raises(ValueError, match="singular"):
        validate_experiment(singular, label="candidate")
    short = _experiment(1.0)
    short["measurement_times"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="10 numbers"):
        validate_experiment(short, label="candidate")


def test_final_parser_is_strict_and_concept_score_recognizes_crossover() -> None:
    response = (
        "<explanation>An extra spatial dimension is compactified. The force "
        "crosses from inverse-square 1/r^2 at short range to 1/r at long "
        "range.</explanation>"
        "<final_law>def discovered_law(pos1, pos2, p1, p2, velocity2, "
        "duration, **params):\n    return pos2, velocity2</final_law>"
    )
    parsed = parse_final(response)
    assert "def discovered_law" in parsed["law_source"]
    score = concept_score_extra_dimensions(parsed["explanation"])
    assert score["score"] == 5
    with pytest.raises(ValueError, match="outside"):
        parse_final("preface " + response)


def test_frozen_smoke_artifact_records_pre_endpoint_failure() -> None:
    result_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "nonmyopic"
        / "discoverphysics_llm_bed_smoke"
        / "RESULT.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_closed"
    assert result["usage"]["requests"] == 1
    assert result["usage"]["reasoning_tokens"] == 0
    assert result["simulator_calls"] == 0
    assert not result["policy_endpoint_exists"]
    assert result["valid_root_experiments"] == 2
    assert result["valid_continuation_experiments"] == 3
