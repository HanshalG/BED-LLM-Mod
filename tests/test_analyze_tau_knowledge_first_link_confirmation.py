import pytest

from scripts.analyze_tau_knowledge_first_link_confirmation import (
    _quantile,
    exact_sign_flip_pvalue,
)


def test_exact_sign_flip_ignores_zeroes():
    assert exact_sign_flip_pvalue([2, 2, 1, -1, 1, 0]) == 0.125


def test_exact_sign_flip_all_zero_is_one():
    assert exact_sign_flip_pvalue([0, 0]) == 1.0


def test_linear_quantile():
    assert _quantile([0.0, 1.0, 2.0], 0.25) == pytest.approx(0.5)
