import pytest
from environments.program_induction.property_observation import observe, PROPERTIES


def test_total_boolean_semantics():
    assert [observe(None,p) for p in PROPERTIES] == [True]+[False]*7
    assert [observe([],p) for p in PROPERTIES] == [False,True,True,False,False,False,False,False]
    assert observe(3,'positive_scalar') and not observe([3],'positive_scalar')
    assert observe([-2,0,-2],'all_even_nonempty')
    assert observe([-2,0,-2],'palindrome_nonempty')
    assert observe([-2,0,-2],'has_negative') and observe([-2,0,-2],'length_ge3')
    assert not observe([1,2],'palindrome_nonempty')
    for y in (None,[],[1],[-2,2],0,-3,4):
        assert all(type(observe(y,p)) is bool for p in PROPERTIES)
    with pytest.raises(ValueError):observe(True,'list')
    with pytest.raises(ValueError):observe([1],'unknown')
