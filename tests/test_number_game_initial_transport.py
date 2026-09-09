from fractions import Fraction

from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon
from scripts.number_game_initial_transport import route


def test_own_prior_replays_expected_full_budget_policy():
    support = [(False,False,False,False), (True,False,True,False),
               (False,True,False,True), (True,True,False,False)]
    solver = ExactMembershipHorizon(support)
    reference = solver.compare()['exact_full_budget_values']
    for depth in (1,2,3):
        outcomes = [route(solver, truth, depth) for truth in support]
        assert all(failed is None for _,failed in outcomes)
        assert sum(loss for loss,_ in outcomes)/len(support) == Fraction(reference[depth-1])


def test_impossible_answer_is_failure_not_prior_reset():
    solver = ExactMembershipHorizon([(False,False,False,False)])
    loss, failed_round = route(solver, (False,True,False,False), 1)
    assert loss is None
    assert failed_round == 2
