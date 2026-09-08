import json
import pytest

from environments.program_induction import constrained
from environments.program_induction.execution_feedback import checks, revision_request
from scripts.deepcoder_opportunity import load_dsl
from scripts.deepcoder_luna_medium_probe import bumped


def fixture():
    d=load_dsl()
    ps=constrained.decode(d,'{"programs":[{"statement":"x2 = Reverse x0",'
        '"next":{"statement":"x3 = Head x2","next":null}}]}')
    return d,ps,[dict(inputs=[[1,2],[3]],output=1)]


def test_exact_failure_and_success():
    d,ps,h=fixture()
    r=checks(d,ps,h)[0]
    assert r==dict(fits_observed_history=False,checks=[dict(example_index=0,
        actual_output=2,expected_output=1,passed=False)])
    h[0]['output']=2
    assert checks(d,ps,h)[0]['fits_observed_history']


def test_paired_payload_only_feedback_differs():
    d,ps,h=fixture()
    a=revision_request(d,ps,h,123,with_feedback=True)
    b=revision_request(d,ps,h,123,with_feedback=False)
    aa=json.loads(a['messages'][1]['content'])
    bb=json.loads(b['messages'][1]['content'])
    assert aa.pop('execution_feedback')==checks(d,ps,h)
    assert aa==bb and aa['history']==h
    a['messages'][1]=b['messages'][1]
    assert a==b
    assert bumped(b)['reasoning']['effort']=='medium'


def test_no_extra_or_hidden_history_fields():
    d,ps,h=fixture()
    h[0]['hidden_program']='not allowed'
    with pytest.raises(ValueError):
        checks(d,ps,h)


def test_error_and_scalar_are_distinct():
    d,ps,h=fixture()
    h[0]['output']=None
    r=checks(d,ps,h)[0]['checks'][0]
    assert r['expected_output'] is None and r['actual_output']==2 and not r['passed']


def test_empty_pool_and_implicit_mode_rejected():
    d,ps,h=fixture()
    with pytest.raises(ValueError):
        checks(d,[],h)
    with pytest.raises(ValueError):
        revision_request(d,ps,h,123,with_feedback=1)
