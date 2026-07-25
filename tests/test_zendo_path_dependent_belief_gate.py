from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from helpers import load_config
from scripts.zendo_path_dependent_belief_gate import (
    OBSERVATION_ACCURACY,
    audit_bank,
    behavior_agreements,
    eig,
    evaluate_rule,
    fixed_support_depth_two_scores,
    parse_hypotheses,
    parse_scenes,
    posterior_weights,
    raw_official_scene,
    spearman,
    truth_function,
    update_weights,
    validate_rule,
    validate_scene,
)


def _block(
    color: str,
    *,
    size: str = "small",
    orientation: str = "left",
    grounded: bool = True,
    touching: list[int] | None = None,
    stacking_on: int | None = None,
) -> dict[str, object]:
    return {
        "color": color,
        "size": size,
        "orientation": orientation,
        "grounded": grounded,
        "touching": touching or [],
        "stacking_on": stacking_on,
    }


def _hypothesis(index: int, rule: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"H{index:02d}",
        "rule_text": f"rule {index}",
        "rule": rule,
    }


def test_rule_dsl_evaluates_official_rule_shapes() -> None:
    scene = validate_scene(
        {
            "blocks": [
                _block("blue", touching=[2]),
                _block("red", size="large", touching=[1]),
            ]
        }
    )
    touching = validate_rule(
        {
            "op": "touching",
            "left": {
                "op": "attribute",
                "attribute": "color",
                "value": "blue",
            },
            "right": {
                "op": "attribute",
                "attribute": "color",
                "value": "red",
            },
        }
    )
    largest_red = validate_rule(
        {
            "op": "largest_all",
            "predicate": {
                "op": "attribute",
                "attribute": "color",
                "value": "red",
            },
        }
    )
    assert evaluate_rule(touching, scene)
    assert evaluate_rule(largest_red, scene)


def test_scene_validation_rejects_asymmetric_touching() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        validate_scene(
            {
                "blocks": [
                    _block("blue", touching=[2]),
                    _block("red"),
                ]
            }
        )


def test_strict_parsers_require_exact_counts_and_unique_rules() -> None:
    rules = [
        {
            "op": "count",
            "predicate": {"op": "any"},
            "comparison": "eq",
            "value": index,
        }
        for index in range(7)
    ]
    rules += [
        {
            "op": "count",
            "predicate": {
                "op": "attribute",
                "attribute": "color",
                "value": ["blue", "red", "green"][index % 3],
            },
            "comparison": "ge",
            "value": 1 + index // 3,
        }
        for index in range(5)
    ]
    hypotheses = parse_hypotheses(
        json.dumps(
            {
                "hypotheses": [
                    _hypothesis(index, rule)
                    for index, rule in enumerate(rules, start=1)
                ]
            }
        )
    )
    assert len(hypotheses) == 12
    with pytest.raises(json.JSONDecodeError):
        parse_hypotheses("```json\n" + json.dumps({"hypotheses": []}) + "\n```")

    scenes = [
        {
            "id": f"X{index:02d}",
            "scene": {"blocks": [_block(["blue", "red", "green"][index % 3], size=["small", "medium", "large"][index // 3])]},
        }
        for index in range(1, 9)
    ]
    assert len(parse_scenes(json.dumps({"scenes": scenes}))) == 8


def test_soft_posterior_and_eig_match_manual_binary_case() -> None:
    red_rule = {
        "op": "exists",
        "predicate": {
            "op": "attribute",
            "attribute": "color",
            "value": "red",
        },
    }
    blue_rule = {
        "op": "exists",
        "predicate": {
            "op": "attribute",
            "attribute": "color",
            "value": "blue",
        },
    }
    hypotheses = [
        _hypothesis(1, validate_rule(red_rule)),
        _hypothesis(2, validate_rule(blue_rule)),
    ]
    red_scene = validate_scene({"blocks": [_block("red")]})
    weights = posterior_weights(hypotheses, [])
    yes_weights, probability_yes = update_weights(
        hypotheses, weights, red_scene, True
    )
    assert probability_yes == pytest.approx(0.5)
    assert yes_weights == pytest.approx(
        [OBSERVATION_ACCURACY, 1.0 - OBSERVATION_ACCURACY]
    )
    expected = math.log(2.0) - (
        -OBSERVATION_ACCURACY * math.log(OBSERVATION_ACCURACY)
        - (1.0 - OBSERVATION_ACCURACY)
        * math.log(1.0 - OBSERVATION_ACCURACY)
    )
    assert eig(hypotheses, weights, red_scene) == pytest.approx(expected)


def test_fixed_depth_two_is_finite() -> None:
    hypotheses = [
        _hypothesis(
            index,
            validate_rule(
                {
                    "op": "count",
                    "predicate": {
                        "op": "attribute",
                        "attribute": "color",
                        "value": color,
                    },
                    "comparison": comparison,
                    "value": value,
                }
            ),
        )
        for index, (color, comparison, value) in enumerate(
            [
                ("red", "ge", 1),
                ("blue", "ge", 1),
                ("green", "ge", 1),
                ("red", "eq", 1),
            ],
            start=1,
        )
    ]
    scenes = [
        validate_scene({"blocks": [_block(color)]})
        for color in ("red", "blue", "green", "red", "blue", "green", "red", "blue")
    ]
    weights = posterior_weights(hypotheses, [])
    scores = fixed_support_depth_two_scores(hypotheses, weights, scenes)
    assert len(scores) == 4
    assert all(math.isfinite(score) for score in scores)


def test_truth_functions_and_behavior_agreement() -> None:
    scene = validate_scene(
        {
            "blocks": [
                _block("blue", touching=[2]),
                _block("red", touching=[1]),
            ]
        }
    )
    assert truth_function("zeta")(scene)
    assert truth_function("xi")(scene)
    exact_xi = [
        _hypothesis(
            1,
            validate_rule(
                {
                    "op": "touching",
                    "left": {
                        "op": "attribute",
                        "attribute": "color",
                        "value": "blue",
                    },
                    "right": {
                        "op": "attribute",
                        "attribute": "color",
                        "value": "red",
                    },
                }
            ),
        )
    ]
    assert behavior_agreements(exact_xi, [scene], truth_function("xi")) == [1.0]


def test_official_scene_conversion_and_audit_are_deterministic() -> None:
    cases = json.loads(
        Path(
            "external/doing-experiments-and-revising-rules/data/zendo_cases.json"
        ).read_text()
    )
    initial = raw_official_scene(cases[0]["t"][0])
    assert truth_function("zeta")(initial)
    bank_a = audit_bank(0, cases[0])
    bank_b = audit_bank(0, cases[0])
    assert bank_a == bank_b
    assert len(bank_a) >= 512


def test_spearman_handles_ties_and_direction() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_zendo_openrouter_config_has_frozen_caps() -> None:
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_concurrency == 8
    assert config.openrouter_projected_cost_usd == pytest.approx(0.30)
    assert config.openrouter_run_budget_usd == pytest.approx(0.75)
    assert config.openrouter_max_output_tokens == 8192
