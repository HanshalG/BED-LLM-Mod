"""Characterize the saved planner, without changing or rerunning its endpoints."""
from environments.neuronbench_compose.mechanics import CompositionalPlanner


class CountingBank:
    def branches(self, state, action):
        return ((0, 1.0),)

    def transition(self, state, action, observation, *, proposal_mode):
        return state + (action,)

    def leaf_risk(self, state):
        return 10.0 - len(state)


def test_level_two_evaluates_full_budget_not_two_observations():
    planner = CompositionalPlanner(CountingBank())
    available = (0, 1, 2, 3)
    assert planner.action_value((), available, 4, 1, 0) == 9.0
    assert planner.action_value((), available, 4, 2, 0) == 6.0
    # A true two-observation horizon on this same bank would stop at risk 8.
    bank = CountingBank()
    state = bank.transition((), 0, 0, proposal_mode='oracle')
    state = bank.transition(state, 1, 0, proposal_mode='oracle')
    assert bank.leaf_risk(state) == 8.0


def test_ladder_evaluation_changes_with_execution_budget_at_fixed_level():
    planner = CompositionalPlanner(CountingBank())
    assert planner.action_value((), (0, 1, 2, 3), 3, 2, 0) == 7.0
    assert planner.action_value((), (0, 1, 2, 3), 4, 2, 0) == 6.0
