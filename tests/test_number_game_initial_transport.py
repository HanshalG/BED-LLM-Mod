from fractions import Fraction
import json
from pathlib import Path

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


def test_complete_bank_preserves_parent_estimand_and_failure_bounds():
    result = json.loads(Path('results/nonmyopic/number_game_initial_transport_20260909/RESULT.json').read_text())
    parent = json.loads(Path('results/nonmyopic/number_game_initial_horizon_audit/20260908-v1/RESULT.json').read_text())
    assert result['status'] == 'complete'
    assert len(result['rows']) == 32
    assert not result['future_supports_opened'] and not result['historical_targets_opened']
    assert result['model_calls'] == result['cost_usd'] == 0
    for depth in range(3):
        own = sum(r['own_prior_brier'][depth] for r in result['rows'])/32
        assert abs(own-parent['aggregate_full_budget_values'][depth]) < 1e-12
        for row in result['rows']:
            lower, upper, failed = (row[k][depth] for k in (
                'unconditional_brier_lower_bound', 'unconditional_brier_upper_bound', 'failure_mass'))
            assert 0 <= lower <= upper <= 1
            assert abs(upper-lower-failed) < 1e-12
