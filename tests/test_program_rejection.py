from time import monotonic

import pytest

from environments.program_induction.rejection import conditioned_draws


def run(values, history, **kwargs):
    iterator = iter(values)
    return conditioned_draws(lambda: next(iterator), lambda p, x: p[x], history,
                             deadline=monotonic()+10, **kwargs)


def test_full_history_and_duplicate_mass():
    a, b, c = ('x', 'z'), ('x', 'y'), ('n', 'y')
    result = run([a, c, b, b], [(0, 'x'), (1, 'y')], num_particles=2, max_draws=4)
    assert result.complete
    assert result.particles == (b, b)
    assert result.draws == 4 and result.evaluations == 7


def test_partial_and_zero_support_are_not_complete():
    result = run([('x',), ('y',)], [(0, 'x')], num_particles=2, max_draws=2)
    assert not result.complete and result.particles == (('x',),)
    result = run([('y',)], [(0, 'x')], num_particles=2, max_draws=1)
    assert not result.complete and not result.particles


def test_empty_history_is_prior_without_evaluation():
    result = run([('a',), ('b',)], [], num_particles=2, max_draws=3)
    assert result.complete and result.evaluations == 0 and result.draws == 2


def test_deadline_and_errors_are_not_silent_rejections():
    with pytest.raises(TimeoutError):
        conditioned_draws(lambda: 1, lambda p, x: p, [], num_particles=1,
                          max_draws=1, deadline=monotonic()-1)
    with pytest.raises(IndexError):
        run([()], [(0, 'a')], num_particles=1, max_draws=1)


@pytest.mark.parametrize('n,b', [(0, 1), (1, 0), (True, 1), (1, 1.5)])
def test_invalid_budgets(n, b):
    with pytest.raises(ValueError):
        run([], [], num_particles=n, max_draws=b)
