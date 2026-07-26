from __future__ import annotations

import pytest

from scripts import infoquest_information_need_gate as needs
from scripts import infoquest_information_need_scoring_audit as audit


def _belief() -> needs.InformationNeedBelief:
    return needs.InformationNeedBelief(
        needs=tuple(f"Need {index}" for index in range(5)),
        weights=(100, 80, 60, 40, 20),
        resolution_probabilities=(
            (90, 20, 10, 10, 10),
            (60, 60, 60, 60, 60),
            (10, 10, 95, 10, 10),
            (30, 20, 10, 5, 1),
        ),
        scores=(0.0, 0.0, 0.0, 0.0),
        selected_action_index=0,
        selected_question="Question.",
    )


def test_variant_scores_include_full_frozen_family():
    scores = audit.variant_scores(_belief())
    assert list(scores) == [
        "power_1",
        "power_2",
        "power_3",
        "power_4",
        "power_6",
        "power_8",
        "power_12",
        "power_16",
        "max_resolution",
        "weighted_max_resolution",
        "resolution_margin",
        "max_minus_mean_remainder",
        "peak_ratio",
    ]
    assert scores["max_resolution"] == (90.0, 60.0, 95.0, 30.0)
    assert scores["resolution_margin"] == (70.0, 0.0, 85.0, 10.0)


def test_power_one_matches_preregistered_linear_score():
    belief = _belief()
    scores = audit.variant_scores(belief)["power_1"]
    assert scores == pytest.approx(
        tuple(
            needs.expected_resolved_mass(
                belief.weights,
                profile,
            )
            for profile in belief.resolution_probabilities
        )
    )
