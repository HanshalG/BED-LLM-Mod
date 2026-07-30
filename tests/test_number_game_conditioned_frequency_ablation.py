from __future__ import annotations

from scripts.number_game_conditioned_frequency_ablation import (
    conditioned_support_variants,
)
from scripts.number_game_generator_aware_bed import DOMAIN, RuleHypothesis


def _hypothesis(name: str, positive: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="manual",
        extension=tuple(number in positive for number in DOMAIN),
    )


def test_conditioned_variants_keep_initial_uniform_and_isolate_propagation() -> None:
    first = _hypothesis("first", {1, 2, 3})
    repeated = _hypothesis("repeated", {1, 3})
    second = _hypothesis("second", {1})
    first_key = (1, True)
    second_key = (1, True, 3, True)
    supports = {
        "uniform_initial": [first],
        "uniform_first": {first_key: [first, repeated]},
        "generated_first": {first_key: [repeated, repeated]},
        "generated_second": {second_key: [second, second]},
    }

    variants = conditioned_support_variants(supports)

    propagated = variants["propagated_conditioned_particles"]
    local = variants["local_refresh_particles"]
    assert propagated["weighted_initial"] == [first]
    assert local["weighted_initial"] == [first]
    assert propagated["weighted_first"][first_key] == [
        repeated,
        repeated,
        first,
    ]
    assert local["weighted_first"][first_key] == propagated[
        "weighted_first"
    ][first_key]
    assert propagated["weighted_second"][second_key] == [
        second,
        second,
        repeated,
        repeated,
        first,
    ]
    assert local["weighted_second"][second_key] == [
        second,
        second,
        first,
        repeated,
    ]
