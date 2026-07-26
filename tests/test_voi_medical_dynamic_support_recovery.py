import math

import pytest

from scripts.voi_medical_dynamic_support_recovery import (
    EXPECTED_REQUESTS,
    immediate_eig,
    outcome_masses,
    parse_answer_labels,
)


def test_answer_map_and_uniform_eig() -> None:
    text = "\n".join(
        ["H1|Yes", "H2|Yes", "H3|Yes", "H4|No", "H5|No", "H6|Maybe"]
    )
    labels = parse_answer_labels(text)
    assert outcome_masses(labels) == {
        "Yes": 0.5,
        "No": 1 / 3,
        "Maybe": 1 / 6,
    }
    expected = math.log(6) - (
        0.5 * math.log(3) + (1 / 3) * math.log(2)
    )
    assert immediate_eig(labels) == pytest.approx(expected)
    assert EXPECTED_REQUESTS == 32
