import pytest

from environments.scilaws.interval_refinement import refine


def test_noncompetitive_action_not_integrated():
    calls = []
    def evaluate(i, a):
        calls.append((i, a))
        return .5, 0.
    r = refine([1.], [[(0, 1), (2, 3)]], evaluate)
    assert calls == [(0, 0)]
    assert r['midpoint'] == .5


def test_overlap_tie_and_weighted_truth():
    p = [.1, .9]
    values = [[.5, .5], [.8, .2]]
    r = refine(p, [[(0, 1), (0, 1)], [(0, 1), (0, 1)]],
               lambda i, a: (values[i][a], 1e-9))
    truth = sum(w*min(v) for w, v in zip(p, values))
    assert r['lower'] <= truth <= r['upper']
    assert abs(r['midpoint']-truth) <= 5e-5
    assert r['trace'][0]['branch'] == 1
    assert not r['numerical_enclosures_rigorous']


@pytest.mark.parametrize('value,error', [(2, 0), (.5, -1), (.5, 1e-3), (float('nan'), 0)])
def test_conflict_or_bad_reference_fails(value, error):
    with pytest.raises(ValueError):
        refine([1.], [[(0, 1)]], lambda i, a: (value, error))


def test_no_call_when_analytic_interval_suffices():
    def bomb(*args):
        raise AssertionError('unnecessary integration')
    assert refine([1.], [[(1, 1.00001)]], bomb)['integrations'] == 0
