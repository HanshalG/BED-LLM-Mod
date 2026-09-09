import pytest
from scripts.physgym_proposal_error_audit import evaluate, mse
from scripts.physgym_scale_error_audit import scale_fit


def test_invalid_not_clipped_or_omitted():
    values=evaluate('sqrt(x0-4)',['N'],[{'N':3},{'N':5}])
    assert values==[None,0.]
    assert mse(values,[0.,0.]) is None


def test_mean_square_and_lengths():
    assert mse([1.,3.],[0.,1.])==2.5
    with pytest.raises(ValueError):
        mse([1.],[])


def test_scale_fit_uses_training_only_and_keeps_domain_failure():
    assert scale_fit([1.,2.],[3.,4.])==2.
    with pytest.raises(ValueError):
        scale_fit([None],[0.])
