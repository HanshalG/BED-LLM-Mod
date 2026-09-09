import math

from scripts.discoverphysics_objective_scope_audit import audit_source


def test_actual_selection_uses_different_objectives():
    result = audit_source()
    assert result['selectors']['myopic_root']['expression'] == (
        'min(immediate_values, key=lambda root_id: (-immediate_values[root_id], root_id))')
    assert result['selectors']['lookahead_root']['expression'] == (
        'min(internal_risks, key=lambda root_id: (internal_risks[root_id], root_id))')


def test_objective_change_can_win_without_any_lookahead():
    prior = [.45, .45, .1]
    targets = [0., 0., 10.]
    actions = {'A': [0, 1, 0], 'B': [0, 0, 1]}
    scores = {}
    for name, outcomes in actions.items():
        information, risk = 0., 0.
        for outcome in set(outcomes):
            ids = [i for i, y in enumerate(outcomes) if y == outcome]
            mass = sum(prior[i] for i in ids)
            information -= mass*math.log(mass)
            mean = sum(prior[i]*targets[i] for i in ids)/mass
            risk += sum(prior[i]*(targets[i]-mean)**2 for i in ids)
        scores[name] = information, risk
    assert scores['A'][0] > scores['B'][0]
    assert scores['A'][1] > 8
    assert scores['B'][1] == 0
