import pytest
from scripts.deepcoder_prior_reserve_audit import reserve


def test_fixed_draws_preserve_multiplicity_and_filter_all_history():
    calls = []
    def draw(i):
        calls.append(i)
        return i%2
    hist = [{'inputs':0,'output':0},{'inputs':1,'output':1}]
    pool, work = reserve(draw, lambda p,x:p+x, hist,6)
    assert calls == list(range(6)) and pool == [0,0,0] and work == 9
    assert reserve(lambda _:1,lambda p,x:p,hist,2)[0] == []
    with pytest.raises(ValueError):
        reserve(draw,lambda p,x:p,hist,True)
