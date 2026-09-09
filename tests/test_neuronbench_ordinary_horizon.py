from fractions import Fraction
from itertools import product

import pytest

from scripts.neuronbench_ordinary_horizon import Solver


def test_horizons_on_two_independent_bits():
    rows = tuple(product((0, 1), repeat=2))
    solver = Solver(rows, rows)
    support = tuple(range(4))
    assert solver.risk(support) == Fraction(1, 4)
    assert solver.value(support, (0, 1), 1) == Fraction(1, 8)
    assert solver.value(support, (0, 1), 2) == 0
    assert solver.choose(support, (0, 1), 1) == 0
    for truth in support:
        assert solver.replay(support, truth, 1, budget=2)[0] == 0


def test_multicategory_shared_predictor_and_random():
    solver = Solver(((0, 0), (1, 0), (2, 1)), ((0,), (2,), (8,)))
    support = (0, 1, 2)
    assert solver.value(support, (0, 1), 1) == 0
    assert solver.choose(support, (0, 1), 1) == 0
    # Random one-query choice: query0 resolves; query1 leaves first two at mean1.
    assert solver.random_value(support, (0, 1), 1, 0) == Fraction(1, 2)
    assert solver.random_value(support, (0, 1), 1, 2) == 0


def test_resource_cap_is_terminal():
    solver = Solver(((0,), (1,)), ((0,), (1,)), max_states=0)
    with pytest.raises(RuntimeError, match='resource cap'):
        solver.value((0, 1), (0,), 1)
