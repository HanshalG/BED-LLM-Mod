import pytest
from scripts.deepcoder_local_horizon_audit import measure


def test_identical_behavior_has_no_internal_headroom():
    r = measure([['a']*9, ['a']*9])
    assert r['initial_risk'] == 0
    assert all(v == 0 for v in r['terminal_risk'].values())


def test_perfectly_distinguishing_queries_resolve_target():
    r = measure([['a']*8+['x'], ['b']*8+['y']])
    assert r['initial_risk'] == pytest.approx(.25)
    assert all(v == 0 for v in r['terminal_risk'].values())
