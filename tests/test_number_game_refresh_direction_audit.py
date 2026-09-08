from fractions import Fraction as F

import pytest

from scripts.number_game_refresh_direction_audit import terms


def test_wrong_direction_vs_overshoot_are_distinct():
    assert terms([F(0)], [F(1)], [0]) == (0, 1)
    a, b = terms([F(2, 5)], [F(1)], [F(1, 2)])
    assert a < 0 and a + b > 0


def test_quadratic_identity_at_multiple_steps_independently():
    before, after, truth = [F(1, 3), F(2, 3)], [F(4, 5), F(1, 5)], [1, 0]
    a, b = terms(before, after, truth)
    for t in (F(0), F(1, 4), F(1), F(2)):
        p = [x + t * (z - x) for x, z in zip(before, after)]
        actual = (
            sum((x - y) ** 2 - (z - y) ** 2 for x, z, y in zip(p, before, truth)) / 2
        )
        assert actual == t * a + t * t * b
    assert b >= 0


def test_mismatched_vectors_rejected():
    with pytest.raises(ValueError):
        terms([F(1)], [], [1])
