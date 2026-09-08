from fractions import Fraction

import pytest

from core.research_headroom import assess_headroom


def check(base, lower, gains=('1/20', '1/20')):
    return assess_headroom(base, lower, gains, scope='exact finite test population; B4; Brier')


def test_adjacent_gains_multiply():
    result = check(1, '19/20')
    assert result['exact_required_total_gain'] == '39/400'
    assert result['status'] == 'ruled_out'
    assert not result['sufficient_for_success']


def test_equality_and_conservative_bound_do_not_establish_success():
    assert check(1, '361/400')['status'] == 'not_ruled_out'
    result = check(1, 0)
    assert result['status'] == 'not_ruled_out'
    assert not result['paid_calls_authorized']


def test_nonuniform_gains():
    assert check(1, 0, [Fraction(1, 10), Fraction(1, 5)])['exact_required_total_gain'] == '7/25'


@pytest.mark.parametrize('base,lower,gains', [
    (0, 0, ['1/20']), (1, -1, ['1/20']), (1, 2, ['1/20']),
    (1., 0, ['1/20']), (True, 0, ['1/20']), (1, 0, []),
    (1, 0, ['nan']), (1, 0, [1]), (1, 0, ['-1/2']),
])
def test_invalid_or_approximate_inputs_fail_closed(base, lower, gains):
    with pytest.raises(ValueError):
        check(base, lower, gains)


def test_scope_required():
    with pytest.raises(ValueError):
        assess_headroom(1, 0, ['1/20'], scope='')
