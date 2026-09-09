import pytest
from scripts.rearc_python_forecast import forecast_python_slots
from scripts.rearc_python_contract import validate


def test_shared_scorer_retains_future_failure_mass_and_dedupes_formatting():
    case={'inputs':[[[0]]],'outputs':[[[0]]],'target_inputs':[[[1]]]}
    a='def transform(g): return g'
    same='def transform(g):\n    # duplicate\n    return g\n'
    b='def transform(g):\n if g[0][0]==0: return g\n raise ValueError()'
    calls=[]
    def evaluate(code,xs):
        calls.append((code,xs))
        return [None if code==b and x==[[1]] else x for x in xs]
    result=forecast_python_slots([a,same,b,None],case,evaluate)
    assert result['weights']==[.5,0,.5,0]
    assert result['outputs']==[[[[1]],None,None,None]]
    assert result['conditioning']['unique_programs']==2
    assert result['replacement_candidates']==0
    assert all(code!=same for code,x in calls)


def test_all_invalid_slots_unit_failure():
    result=forecast_python_slots([None]*8,{'inputs':[[[0]]],'outputs':[[[0]]],'target_inputs':[[[1]]]},
        lambda *a:pytest.fail('invalid program executed'))
    assert result['weights']==[1.] and result['outputs']==[[None]]


def test_malformed_source_not_silently_salvaged():
    with pytest.raises(SyntaxError):
        forecast_python_slots(['bad('],{},None)


def test_annotation_and_version_contract():
    validate('def transform(g: list[list[int]]) -> list[list[int]]:\n return g')
    with pytest.raises(SyntaxError):
        validate('def transform(g):\n match g:\n  case _: return g')
