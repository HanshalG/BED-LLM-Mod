from __future__ import annotations

import json

import pytest

from scripts.number_game_generator_aware_bed import DOMAIN, RuleHypothesis
from scripts.number_game_pooled_support import encode_pooled_responses
from scripts.number_game_proposal_frequency_calibration_audit import (
    pooled_particle_multiset,
    posterior_predictive_brier,
    retain_particle_multiset,
    unique_extensions,
)


def _response(expressions: list[str]) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"name": f"h{index}", "expression": expression}
                for index, expression in enumerate(expressions)
            ]
        }
    )


def test_pooled_particle_multiset_preserves_cross_draw_multiplicity() -> None:
    first_draw = [f"n < {threshold}" for threshold in range(1, 25)]
    second_draw = ["n < 1"] + [
        f"n > {threshold}" for threshold in range(24, 47)
    ]
    response = encode_pooled_responses(
        [
            _response(first_draw),
            _response(second_draw),
        ]
    )

    particles = pooled_particle_multiset(response)

    assert len(particles) == 48
    assert len(unique_extensions(particles)) == 47
    below_one = tuple(number < 1 for number in DOMAIN)
    assert sum(item.extension == below_one for item in particles) == 2


def test_retain_particle_multiset_adds_every_consistent_parent_particle() -> None:
    false = RuleHypothesis("false", "manual", (False,) * len(DOMAIN))
    true = RuleHypothesis("true", "manual", (True,) * len(DOMAIN))

    retained = retain_particle_multiset(
        generated=[true],
        parent=[false, true, true],
        query=4,
        label=True,
    )

    assert retained == [true, true, true]


def test_frequency_weighted_brier_uses_particle_mass() -> None:
    false = RuleHypothesis("false", "manual", (False,) * len(DOMAIN))
    true = RuleHypothesis("true", "manual", (True,) * len(DOMAIN))

    uniform = posterior_predictive_brier([false, true], true)
    weighted = posterior_predictive_brier([false, true, true], true)

    assert uniform == pytest.approx(0.25)
    assert weighted == pytest.approx(1.0 / 9.0)


def test_frequency_weighted_brier_respects_excluded_queries() -> None:
    alternating = RuleHypothesis(
        "alternating",
        "manual",
        tuple(number % 2 == 0 for number in DOMAIN),
    )
    target = RuleHypothesis("target", "manual", (True,) * len(DOMAIN))

    score = posterior_predictive_brier(
        [alternating],
        target,
        excluded=(0,),
    )

    assert score == pytest.approx(0.5)
