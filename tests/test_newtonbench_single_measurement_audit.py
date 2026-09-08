import math

import pytest

from scripts.newtonbench_single_measurement_audit import (
    coefficient,
    classification_bound,
    literal_assignments,
)


def test_pairwise_gaussian_overlap_and_bayes_bound():
    assert coefficient(0, 1, 0, 1) == 1
    assert coefficient(0, 1, 2, 1) == pytest.approx(math.exp(-0.5))
    actual_binary_error = 0.5 * math.erfc(1 / math.sqrt(2))
    assert actual_binary_error < classification_bound([0, 2], [1, 1])
    assert classification_bound([0, 0, 0], [1, 1, 1]) == pytest.approx(2 / 3)
    assert coefficient(1, 0.1, 4, 0.2) == coefficient(4, 0.2, 1, 0.1)
    assert coefficient(-1, 0.1, -4, 0.2) == coefficient(1, 0.1, 4, 0.2)


def test_small_unit_scales_do_not_change_the_bound():
    assert coefficient(1e-12, 1e-13, 2e-12, 2e-13) == pytest.approx(
        coefficient(1, 0.1, 2, 0.2)
    )


@pytest.mark.parametrize(
    "args", [(0, 0, 1, 1), (math.nan, 1, 0, 1), (0, 1, math.inf, 1)]
)
def test_bad_likelihood_is_not_silently_omitted(args):
    with pytest.raises(ValueError):
        coefficient(*args)


def test_literal_extraction_does_not_execute_source():
    assert literal_assignments(
        "A=2\nB: dict={'x':[1,2]}\nC=explode()\nraise RuntimeError()"
    ) == {"A": 2, "B": {"x": [1, 2]}}
