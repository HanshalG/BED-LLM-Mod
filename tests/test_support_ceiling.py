import pytest
from environments.program_induction.support_ceiling import oracle_mixture


def test_missing_all_outputs_has_positive_unavoidable_floor():
    r=oracle_mixture([['a'],['b']],['c'])
    assert r['lower']==pytest.approx(.75)
    assert r['upper']==pytest.approx(.75)
    assert r['unsupported_targets']==1


def test_shared_weights_not_independent_per_target_choices():
    r=oracle_mixture([['a','b'],['b','a']],['a','a'])
    assert r['pointwise_floor']==0
    assert r['lower']==pytest.approx(.25)
    assert r['upper']==pytest.approx(.25)


def test_correct_world_and_duplicate_invariance():
    rows=[['a','a'],['b','a'],['b','b']]
    a=oracle_mixture(rows,['a','a'])
    b=oracle_mixture(list(reversed(rows))+rows,['a','a'])
    assert a['upper']==pytest.approx(0,abs=1e-10)
    assert b['upper']==pytest.approx(a['upper'],abs=1e-10)


def test_empty_support_uses_fixed_abstention_penalty():
    r=oracle_mixture([],['a'])
    assert r['lower']==r['upper']==1 and r['abstention']


def test_shape_failure():
    with pytest.raises(ValueError):
        oracle_mixture([['a','b']],['a'])
