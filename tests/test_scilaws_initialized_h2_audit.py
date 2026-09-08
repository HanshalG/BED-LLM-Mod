import pytest

from scripts.scilaws_initialized_h2_audit import ORDERS, assess


def rows():
    return [dict(order=o, status='completed', action=0,
                 root_action_values=list(enumerate(range(8)))) for o in ORDERS]


def test_full_pass_and_reference_failure():
    data = rows()
    assert all(c['passed'] for c in assess(data)['candidates'])
    data[-1]['root_action_values'][0] = (0, .001)
    assert not assess(data)['reference_valid']
    assert not any(c['passed'] for c in assess(data)['candidates'])
    data[-1]['status'] = 'resource_limit'
    assert assess(data)['reason'] == 'reference_incomplete'


def test_missing_actions_and_orders_rejected():
    data = rows()
    with pytest.raises(ValueError, match='order coverage'):
        assess(data[:-1])
    data[0]['root_action_values'] = [(0, 0)]
    with pytest.raises(ValueError, match='action coverage'):
        assess(data)


def test_candidate_regret_cannot_pass():
    data = rows()
    data[0]['action'] = 1
    assert not assess(data)['candidates'][0]['passed']
