import pytest

from scripts.scilaws_mixed_refinement_audit import ORDERS, assess, fixture


def rows():
    return [dict(order=o, status='completed', action=0,
                 root_action_values=[(0, 0.2), (1, 0.3)]) for o in ORDERS]


def test_refinement_gate_requires_reference_and_all_roots():
    r = rows()
    assert all(c['passed'] for c in assess(r)['candidates'])
    r[-1]['root_action_values'][0] = (0, 0.201)
    assert not assess(r)['reference_valid']
    assert not any(c['passed'] for c in assess(r)['candidates'])
    r[-1]['status'] = 'resource_limit'
    assert assess(r)['reason'] == 'reference_incomplete'
    with pytest.raises(ValueError):
        assess(r[:-1])
    r = rows()
    r[0]['root_action_values'] = [(0, 0.2)]
    with pytest.raises(ValueError):
        assess(r)


def test_regret_and_max_error_use_reference_not_candidate_minimum():
    r = rows()
    r[0]['action'] = 1
    r[0]['root_action_values'] = [(0, 0.4), (1, 0.1)]
    c = assess(r)['candidates'][0]
    assert c['regret'] == pytest.approx(0.1)
    assert c['max_root_error'] == pytest.approx(0.2)
    assert not c['passed']


def test_fixture_history_changes_belief_not_geometry():
    a, s = fixture(4, ())
    b, t = fixture(4, ((0, 0.7),))
    assert s != t
    assert a.num_actions == b.num_actions == 2
    assert len(s.components) == len(t.components) == 2
