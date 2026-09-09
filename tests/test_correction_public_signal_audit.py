import pytest

from scripts.correction_public_signal_audit import linear_signal


def test_exact_linear_signal_and_offset_invariance():
    x = [0., 1., 2., 3., 4., 5.]
    first = linear_signal(x, [2*v+1 for v in x])
    second = linear_signal(x, [2*v+8 for v in x])
    assert first == second
    assert first['slope'] == 2
    assert first['linear_sse'] == 0
    assert first['permutation_extreme'] == 2
    assert first['permutation_total'] == 720


def test_constant_input_rejected():
    with pytest.raises(ValueError, match='unvaried'):
        linear_signal([1.]*6, [0.]*6)
