import json
import pytest
from scripts.rearc_qualification_panel import request_body,collect

DSL='def identity(x: Grid) -> Grid: pass'
G={'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
CASE={'inputs':[[[0]]]*3,'outputs':[[[0]]]*3,'target_inputs':[[[0]]]*8}


def test_blind_prompt_invariant_to_hidden_answers():
    other={**CASE,'outputs':[[[0]],[[1]],[[2]]]}
    assert request_body(CASE,'blind',0,DSL)==request_body(other,'blind',0,DSL)
    assert request_body(CASE,'aware',0,DSL)!=request_body(other,'aware',0,DSL)


def test_endpoints_only_after_forecasts_and_exact_call_count():
    events=[]
    def request(tag,body):
        events.append(tag)
        return json.dumps({'hypotheses':[G]*4})
    def endpoints():
        assert events[-1]=='sealed'
        return [[[[0]]]*8 for _ in range(4)]
    result=collect([CASE]*4,DSL,request,lambda graph,inputs:inputs,lambda x,y:[G],
                   lambda forecasts:events.append('sealed'),endpoints)
    assert len(events)==13
    assert not result['qualification_passed']


def test_failed_initial_does_not_open_endpoints():
    def bomb(): raise AssertionError('endpoint opened')
    with pytest.raises(ValueError,match='initial semantic'):
        collect([CASE]*4,DSL,lambda t,b:json.dumps({'hypotheses':[G]*4}),
                lambda g,x:[None]*len(x),lambda x,y:[G],lambda f:None,bomb)


def test_positive_fixture_requires_predictive_improvement_not_just_calls():
    def graph(op): return {'steps':[{'id':'x0','op':op,'args':['I']}],'output':'x0'}
    def request(tag,body):
        return json.dumps({'hypotheses':[graph('good' if tag.endswith('aware') else 'bad')]*4})
    def evaluate(g,inputs):
        if g['steps'][0]['op']=='good' or len(inputs)==1:
            return [[[0]]]*len(inputs)
        return [[[0]],[[1]]]+[[[0]]]*(len(inputs)-2)
    result=collect([CASE]*4,'def good(x: Grid) -> Grid: pass\ndef bad(x: Grid) -> Grid: pass',
        request,evaluate,lambda x,y:[graph('good')],lambda f:None,lambda:[[[[0]]]*8 for _ in range(4)])
    assert result['qualification_passed']
    assert result['task_wins']==4
    assert result['depth_authorized'] is False
