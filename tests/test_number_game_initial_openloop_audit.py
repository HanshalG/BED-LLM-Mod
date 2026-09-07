from fractions import Fraction
from itertools import combinations, product

import pytest

from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon
from scripts.number_game_initial_openloop_audit import OpenLoopMembership


def independent_risk(rows):
    return sum(
        (
            Fraction(sum(row[q] for row in rows), len(rows))
            * (1 - Fraction(sum(row[q] for row in rows), len(rows)))
            for q in range(len(rows[0]))
        ),
        Fraction(0),
    ) / len(rows[0])


def independent_plan(rows, horizon):
    def value(queries):
        groups = {}
        for row in rows:
            groups.setdefault(tuple(row[q] for q in queries), []).append(row)
        return sum(
            (
                Fraction(len(g), len(rows)) * independent_risk(g)
                for g in groups.values()
            ),
            Fraction(0),
        )

    # Include short sets and all original queries, unlike optimized partition DP.
    candidates = [
        (value(q), q)
        for k in range(min(horizon, len(rows[0])) + 1)
        for q in combinations(range(len(rows[0])), k)
    ]
    return min(v for v, _ in candidates)


@pytest.mark.parametrize(
    "selection", [[0, 1, 2, 3, 6, 9, 12, 15], [0, 3, 5, 10, 12], [0, 15]]
)
def test_openloop_matches_independent_explicit_outcome_groups(selection):
    universe = list(product((False, True), repeat=4))
    rows = [universe[i] for i in selection]
    solver = ExactMembershipHorizon(rows)
    control = OpenLoopMembership(solver)
    for h in range(4):
        assert control.plan(solver.full, h)[0] == independent_plan(rows, h)
    committed = control.plan(solver.full, 3)[0]
    receding = control.receding(solver.full, 3)
    assert solver.plan(solver.full, 3)[0] <= receding <= committed


def test_exact_positive_adaptivity_example_and_receding_control():
    # First query routes to one of two distinct informative second queries.
    rows = [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, False, True),
    ]
    solver = ExactMembershipHorizon(rows)
    control = OpenLoopMembership(solver)
    assert solver.plan(solver.full, 2)[0] == 0
    assert control.plan(solver.full, 2)[0] == Fraction(1, 24)
    assert control.receding(solver.full, 2) == 0


def test_cap_and_resolved_support():
    solver = ExactMembershipHorizon([(False,), (True,)])
    with pytest.raises(RuntimeError, match="query-set cap"):
        OpenLoopMembership(solver, max_sets=0).plan(solver.full, 3)
    assert OpenLoopMembership(solver).receding(1, 3) == 0
