import json
import pytest
from scripts.rearc_named_plan_panel import collect,score
from scripts.rearc_named_plan_contract import body,validate_body,validate_route,response_text
from scripts.rearc_named_plan_interface import schema


def forecasts(diverse=True,correct=True):
    row = {'contrasting':{'weights':[.5,.5], 'outputs':[
        [[[1]],[[0]]] if diverse else [[[1]],[[1]]] for _ in range(10)]},
        'ordinary':{'weights':[1.], 'outputs':[[[[0]]] for _ in range(10)]}}
    if not correct:
        row['contrasting']['outputs']=[[[[2]],[[0]]] for _ in range(10)]
    return [row for _ in range(6)]


def test_full_gate_and_separate_query_target_reports():
    result = score(forecasts(),[[[[1]]]*10 for _ in range(6)])
    assert result['qualification_passed'] and not result['depth_authorized']
    assert result['covered_query_answers']==12 and result['diverse_tasks']==6
    assert set(result['means']['contrasting'])=={'all','query','target'}


def test_accurate_unambiguous_and_diverse_wrong_forecasts_fail():
    labels = [[[[1]]]*10 for _ in range(6)]
    assert not score(forecasts(diverse=False),labels)['gates']['predictive_disagreement']
    assert not score(forecasts(correct=False),labels)['gates']['answer_support']


@pytest.mark.parametrize('failed',[False,True])
def test_exact_calls_blind_labels_and_sealing(failed):
    requests,seals,updates = {},[],[]
    cases = [{'inputs':[[[0]]],'outputs':[[[0]]],'query_inputs':[[[1]],[[2]]],
        'target_inputs':[[[3]]]*8,'target_hashes':['sealed']*10} for _ in range(6)]
    dsl = 'def identity(x: Any) -> Any:\n return x\n'
    def request(tag,value):
        requests[tag]=value
        if tag.endswith('_plan'):
            return json.dumps({f'p{i}':'Possible rule' for i in range(4)})
        return json.dumps({'hypotheses':['identity(I)']*8})
    def seal(value):
        assert len(requests)==36
        seals.append(value)
    def targets():
        assert len(seals)==1 and not failed
        return [[[[1]],[[2]]]+[[[3]]]*8 for _ in range(6)]
    result = collect(cases,dsl,request,lambda g,x:[None]*len(x) if failed else x,
        lambda g,x:{'status':'ok','output':x},lambda *args:updates.append(args),seal,targets)
    assert len(requests)==36 and len(updates)==12
    assert len(seals)==(0 if failed else 1)
    assert result['status']==('initial_coverage_null' if failed else 'complete')
    for i in range(6):
        for stage in ('plan','compile','repair'):
            a,b = [requests[f'{i}_{arm}_{stage}'] for arm in ('contrasting','ordinary')]
            assert a['seed']==b['seed']
            assert len(json.loads(a['messages'][1]['content'])['observations'])==1
    assert not result['qualification_passed']


def test_contract_rejects_features_size_reasoning_and_applicable_prices():
    value = body([],40400,schema('plan'))
    validate_body(value)
    for changed in ({**value,'tools':[]},{**value,'messages':[{'content':'x'*65536}]},
                    {**value,'reasoning':{'effort':'high'}},{**value,'seed':True}):
        with pytest.raises(ValueError):
            validate_body(changed)
    prices = {'prompt':'.0000002','completion':'.0000012','input_cache_write':'.00000025','input_cache_read':'.00000002'}
    validate_route({'pricing':prices})
    with pytest.raises(ValueError):
        validate_route({'pricing':{**prices,'overrides':[{'min_prompt_tokens':131072}]}})
    with pytest.raises(ValueError):
        validate_route({'pricing':{**prices,'prompt':'.0000003'}})


def test_receipt_context_and_cost_caps():
    raw = {'model':'openai/gpt-5.6-luna','provider':'OpenAI',
        'usage':{'prompt_tokens':131072,'completion_tokens':16384,'cost':.08,
                 'completion_tokens_details':{'reasoning_tokens':100}},
        'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}
    assert response_text(raw)=='ok'
    for key,value in (('prompt_tokens',131073),('cost',.08001)):
        changed = {**raw,'usage':{**raw['usage'],key:value}}
        with pytest.raises(ValueError):
            response_text(changed)
