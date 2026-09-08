from fractions import Fraction
from itertools import product

from scripts.semantic_scene_opportunity import MenuHorizon, evaluate, menu_for, scenes


def test_public_scene_coverage_and_label_free_menus():
    universe = scenes()
    assert len(universe) == 18 + 18 * 19 // 2 == 189
    for i in range(4):
        menu = menu_for(universe, i)
        assert len(menu) == len(set(menu)) == 12
        assert menu == menu_for(universe, i)
        assert all(0 <= q < 189 for q in menu)


def test_actions_restricted_but_unqueried_targets_still_scored():
    rows = [(False, False), (False, True), (True, False), (True, True)]
    solver = MenuHorizon(rows, (0,))
    assert list(q for q, _, _ in solver.partitions(solver.full)) == [0]
    assert solver.deployed(solver.full, 3, 3) == Fraction(1, 8)
    assert solver.risk(solver.full) == Fraction(1, 4)
    solver.deployed.cache_clear()


def test_exact_policy_values_against_independent_uniform_bit_worlds():
    rows = list(product((False, True), repeat=3))
    solver = MenuHorizon(rows, (0, 1, 2))
    for budget in range(4):
        # Each queried independent bit removes 1/12 of average target variance.
        expected = Fraction(3 - budget, 12)
        for horizon in (1, 2, 3):
            assert solver.deployed(solver.full, budget, horizon) == expected
        assert solver.random_policy(solver.full, (0, 1, 2), budget) == expected
    solver.deployed.cache_clear()
    solver.random_policy.cache_clear()


def test_complete_panel_row_runs_all_controls_and_handles_plateau():
    row = evaluate(
        list(product((False, True), repeat=3)),
        (0, 1, 2),
        {
            "max_menu_seconds": 5,
            "max_states_per_menu": 1000,
            "max_openloop_sets_per_menu": 1000,
        },
    )
    assert len(row["exact_values"]) == 6
    assert all(Fraction(value) == 0 for value in row["exact_values"].values())
