import pytest

from environments.scilaws.weighted_intervals import weighted_min_interval


def test_minimum_and_weighted_bounds():
    r = weighted_min_interval([.25, .75], [[(1, 2), (3, 4)], [(2, 3), (1, 4)]], correction=-.1)
    assert r['lower'] == pytest.approx(.9)
    assert r['upper'] == pytest.approx(2.65)
    assert r['ideal_exact_branch_refinements'] == 2
    assert r['refinement_order'] == [1, 0]
    assert not r['outer_error_bounded']


def test_low_mass_wide_branch_can_remain():
    r = weighted_min_interval([1-1e-6, 1e-6], [[(1, 1)], [(0, 1)]])
    assert r['within_terminal_budget']
    assert r['ideal_exact_branch_refinements'] == 0


@pytest.mark.parametrize('p,v', [([.5], [[(0, 1)]]), ([1.], [[]]), ([1.], [[(2, 1)]])])
def test_malformed_rejected(p, v):
    with pytest.raises(ValueError):
        weighted_min_interval(p, v)
