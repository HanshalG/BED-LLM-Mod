import pytest

from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimitExceeded, SearchLimits
from environments.scilaws.split_resolution import h2_work_bound, leaf_work_bound, score_h2
from scripts.scilaws_mixed_refinement_audit import fixture, HISTORIES


@pytest.mark.parametrize('history', HISTORIES)
def test_equal_resolution_full_root_equivalence(history):
    model, state = fixture(4, history)
    diagnostic = score_h2(model, state, inner_order=4)
    plan = HorizonPlanner(model).plan(state, 2, allow_repeats=True)
    assert dict(diagnostic['root_action_values']) == pytest.approx(dict(plan.root_action_values), abs=1e-12)
    assert diagnostic['action'] == plan.root.action
    assert diagnostic['evaluated_nodes'] <= diagnostic['unmerged_work_bound']


def test_unequal_resolution_manual_and_no_mutation():
    outer, state = fixture(4, ((0, .7),))
    inner, _ = fixture(8, ())
    result = score_h2(outer, state, inner_order=8)
    for action, value in result['root_action_values']:
        expected = sum(b.probability * min(inner.expected_terminal_risk(b.state, a)[0]
                       for a in range(2)) for b in outer.branches(state, action))
        expected += outer.horizon_chance_risk_correction(state, action, 2)
        assert value == pytest.approx(expected, abs=1e-12)
    assert outer.quadrature_order == 4


def test_preflight_rejects_before_branch_work(monkeypatch):
    model, state = fixture(4, ())
    def bomb(*args):
        raise AssertionError('branches opened')
    monkeypatch.setattr(model, 'branches', bomb)
    with pytest.raises(SearchLimitExceeded, match='before evaluation'):
        score_h2(model, state, inner_order=4, limits=SearchLimits(max_nodes=10))


def test_cost_and_invalid_order():
    assert leaf_work_bound(8, 4, (16, 16)) == 262144
    assert leaf_work_bound(8, 4, (2, 2, 2)) == 262144
    assert h2_work_bound(8, 4, 16, 4) == 66049
    for bad in (True, 0, 1.5):
        with pytest.raises(ValueError):
            leaf_work_bound(8, 4, (bad,))
