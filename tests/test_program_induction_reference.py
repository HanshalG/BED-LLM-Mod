from fractions import Fraction
from functools import lru_cache
import random

import pytest

from environments.program_induction.reference import ProgramReference


def test_multiclass_risk_and_duplicate_prior_mass():
    ref = ProgramReference([['a', 'x'], ['b', 'y'], ['b', 'y']], 1)
    assert ref.risk(ref.initial_state) == pytest.approx(2/9)
    branches = ref.branches(ref.initial_state, 0)
    assert [b.probability for b in branches] == [1/3, 2/3]
    assert ref.planner.plan(ref.initial_state, 1).root.expected_risk == 0
    ref.clear()


@pytest.mark.parametrize('seed', range(3))
def test_against_independent_exact_enumeration(seed):
    rng = random.Random(seed)
    rows = [[str(rng.randrange(3)) for _ in range(6)] for _ in range(7)]
    ref = ProgramReference(rows, 4)

    @lru_cache(None)
    def value(state, menu, depth):
        if not depth:
            return sum((Fraction(sum(rows[i][t] != rows[j][t] for i in state for j in state),
                                 2*len(state)**2) for t in (4, 5)), Fraction())/2
        scores = []
        for q in menu:
            children = [tuple(i for i in state if rows[i][q] == label)
                        for label in sorted({rows[i][q] for i in state})]
            scores.append(sum((Fraction(len(c), len(state))*value(
                c, tuple(a for a in menu if a != q), depth-1) for c in children), Fraction()))
        return min(scores)

    for depth in (1, 2, 3, 4):
        actual = ref.planner.plan(ref.initial_state, depth).root.expected_risk
        assert actual == pytest.approx(float(value(ref.initial_state, (0, 1, 2, 3), depth)), abs=1e-12)
    assert ref.deployed(ref.initial_state, (0, 1, 2, 3), 4, 4) == pytest.approx(float(
        value(ref.initial_state, (0, 1, 2, 3), 4)), abs=1e-12)
    ref.clear()


def test_targets_do_not_drop_when_queries_selected():
    ref = ProgramReference([['a', 'b', 'x'], ['a', 'c', 'y']], 2)
    assert ref.risk((0, 1)) == .25
    assert ref.deployed((0, 1), (0,), 1, 1) == .25
    assert ref.deployed((0, 1), (1,), 1, 1) == 0
    ref.clear()


@pytest.mark.parametrize('rows,count', [([], 1), ([['a']], 1), ([['a', 'b'], ['a']], 1),
                                         ([[1, 2]], 1), ([['a', 'b']], True)])
def test_invalid_input(rows, count):
    with pytest.raises(ValueError):
        ProgramReference(rows, count)


def test_whole_panel_limit():
    ref = ProgramReference([['a', 'x']], 1, seconds=-1)
    with pytest.raises(TimeoutError):
        ref.risk((0,))
    ref.clear()
