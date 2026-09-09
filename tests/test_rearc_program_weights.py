import pytest
from scripts.rearc_program_weights import condition_programs


def test_duplicates_do_not_inflate_prior_and_invalids_have_zero_likelihood():
    result=condition_programs([{'p':1},{'p':1},{'p':2},{'p':3}],
                             [[[[1]]],[[[1]]],[[[1]]],[None]], [[[1]]])
    assert result['weights']==[.5,0,.5,0]
    assert result['unique_programs']==3
    assert result['consistent_programs']==2


def test_empty_posterior_is_explicit_failure_not_uniform_reset():
    result=condition_programs([{'p':1}],[[[[2]]]],[[[1]]])
    assert result['failed'] and result['weights']==[0]
    assert result['failure_forecast']=='unit_mass_on_execution_failure'


def test_duplicate_replay_disagreement_and_missing_predictions_rejected():
    with pytest.raises(ValueError,match='inconsistent'):
        condition_programs([{'p':1}]*2,[[[[1]]],[[[2]]]],[[[1]]])
    with pytest.raises(ValueError,match='missing'):
        condition_programs([{'p':1}],[[]],[[[1]]])
