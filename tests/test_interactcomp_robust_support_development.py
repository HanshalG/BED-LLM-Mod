import math

import pytest

from scripts.interactcomp_robust_support_development import (
    balanced_model_information,
    robust_root_scores,
)


def test_balanced_model_information_is_zero_for_matching_predictions():
    assert balanced_model_information(
        ["Y", "Y", "N", "N"],
        ["Y", "Y", "N", "N"],
    ) == pytest.approx(0.0)


def test_balanced_model_information_is_log_two_for_opposite_consensus():
    assert balanced_model_information(
        ["Y"] * 8,
        ["N"] * 8,
    ) == pytest.approx(math.log(2.0))


def test_robust_score_adds_model_identity_information_to_current_eig():
    current = ["YYYY", "YYYY", "NYYY", "NYYY"] * 2
    auxiliary = ["NNYY", "NNYY", "NNYY", "NNYY"] * 2
    current_eig, model_information, robust = robust_root_scores(
        current,
        auxiliary,
    )
    assert current_eig[0] == pytest.approx(math.log(2.0))
    assert current_eig[1] == pytest.approx(0.0)
    assert model_information[1] == pytest.approx(math.log(2.0))
    assert robust[1] == pytest.approx(math.log(2.0))
    assert robust[0] == pytest.approx(
        current_eig[0] + model_information[0]
    )
