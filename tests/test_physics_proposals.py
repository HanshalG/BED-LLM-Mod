import json
import pytest
from environments.program_induction.physics_proposals import messages, decode, predict, symbolic


def test_semantic_context_only_difference():
    args=(['R'],'thermal resistance',{'R':'resistance'},[({'R':1.},0.)])
    a=json.loads(messages(*args,True)[1]['content'])
    b=json.loads(messages(*args,False)[1]['content'])
    assert a.pop('context')=='thermal resistance'
    a.pop('variable_meanings')
    assert a==b


def test_strict_schema_and_dedup():
    assert decode('{"expressions":["x0", "(x0)"]}',1)==['x0']
    with pytest.raises(ValueError):
        decode('{"expressions":["__import__(1)"]}',1)
    with pytest.raises(ValueError):
        decode('{"expressions":["x0"],"weight":1}',1)


def test_belief_and_empty_support():
    out=predict(['x0','2*x0'],['a'],[({'a':1.},0.)],[{'a':2.}])
    assert out['weights'][0]>.999
    assert predict(['-1'],['a'],[({'a':1.},0.)],[{'a':2.}])['status']=='empty_support'


def test_symbolic_constant_and_no_target_labels():
    values=symbolic(['a'],[({'a':1.},2.),({'a':2.},2.)],[{'a':3.}])
    assert values==pytest.approx([2.])
