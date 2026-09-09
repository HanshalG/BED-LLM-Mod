import json
import pytest
from scripts.rearc_mechanism_interface import messages,parse_plan,parse_programs,schema
from scripts.rearc_prediction_diversity import summarize

DSL = 'def identity(x: Any) -> Any:\n return x\n'


def plan():
    return {'plans':[{'id':f'p{i}','description':'A possible explanation.'} for i in range(4)]}


def test_modes_share_public_data_and_never_accept_extra_observation_fields():
    args = dict(inputs=[[[0]],[[1]]],observations=[{'index':0,'output':[[0]]}],dsl_source=DSL)
    for stage in ('plan','compile'):
        a,b = [messages(stage,mode,**args,plan=plan() if stage=='compile' else None)
               for mode in ('contrasting','ordinary')]
        assert a[1]==b[1] and a[0]!=b[0]
        payload = json.loads(a[1]['content'])
        assert len(payload['observations'])==1
        assert ('dsl' in payload)==(stage=='compile')
    args['observations'][0]['hidden_output']=[[1]]
    with pytest.raises(ValueError):
        messages('plan','contrasting',**args)


def test_plan_strict_order_duplicate_keys_and_byte_bounds():
    assert parse_plan(json.dumps(plan()))==plan()
    for value in ({'plans':plan()['plans'][::-1]}, {'plans':plan()['plans'][:3]}):
        with pytest.raises(ValueError):
            parse_plan(json.dumps(value))
    with pytest.raises(ValueError,match='duplicate'):
        parse_plan('{"plans":[],"plans":[]}')
    value = plan()
    value['plans'][0]['description']='a'*513
    with pytest.raises(ValueError):
        parse_plan(json.dumps(value))


def test_eight_programs_no_partial_batch_salvage():
    value = {'hypotheses':['identity(I)']*8}
    assert len(parse_programs(json.dumps(value),DSL)['graphs'])==8
    value['hypotheses'][-1]='eval(I)'
    with pytest.raises(ValueError):
        parse_programs(json.dumps(value),DSL)
    assert schema('compile')['json_schema']['schema']['properties']['hypotheses']['maxItems']==8


def test_names_or_program_count_do_not_imply_predictive_diversity():
    result = summarize(['a','b','c'],[[[[0]]]]*3,1)
    assert result['unique_programs']==3 and result['prediction_classes']==1
    assert result['queries'][0]['entropy_nats']==0
    assert summarize(['a','a'],[[[[0]]]]*2,1)['unique_programs']==1


def test_inconsistent_duplicate_missing_predictions_and_failure_mass():
    with pytest.raises(ValueError):
        summarize(['a','a'],[[[[0]]],[[[1]]]],1)
    with pytest.raises(ValueError):
        summarize(['a'],[[]],1)
    result = summarize(['a','b'],[[[[0]]],[None]],1)
    assert result['prediction_classes']==2
    assert result['queries'][0]['failure_probability']==.5
    assert summarize([],[],1)['status']=='empty_support'


@pytest.mark.parametrize('mode',['contrasting','ordinary'])
def test_three_call_update_exact_slots_and_public_feedback(mode):
    from scripts.rearc_mechanism_update import update
    calls = []
    def request(stage,messages,format):
        calls.append((stage,messages,format))
        return json.dumps(plan() if stage=='plan' else {'hypotheses':['identity(I)']*8})
    result = update(mode=mode,inputs=[[[0]],[[1]]],observations=[{'index':0,'output':[[0]]}],
        dsl_source=DSL,request=request,diagnose=lambda g,x:{'status':'ok','output':x})
    assert [c[0] for c in calls]==['plan','compile','repair']
    assert len(result['slots'])==len(result['graphs'])==16
    feedback = json.loads(calls[-1][1][-1]['content'])['public_execution_feedback']['programs']
    assert all(len(row)==1 and row[0]['example_index']==0 for row in feedback)


def test_invalid_compilation_gets_only_fixed_repair_and_no_execution():
    from scripts.rearc_mechanism_update import update
    calls = []
    def request(stage,*args):
        calls.append(stage)
        return json.dumps(plan()) if stage=='plan' else 'invalid'
    result = update(mode='contrasting',inputs=[[[0]]],observations=[{'index':0,'output':[[0]]}],
        dsl_source=DSL,request=request,diagnose=lambda *a:pytest.fail('invalid program executed'))
    assert calls==['plan','compile','repair'] and result['slots']==[None]*16


def test_invalid_plan_stops_before_compilation():
    from scripts.rearc_mechanism_update import update
    calls = []
    def request(stage,*args):
        calls.append(stage)
        return 'invalid'
    with pytest.raises(ValueError):
        update(mode='contrasting',inputs=[[[0]]],observations=[],dsl_source=DSL,
               request=request,diagnose=None)
    assert calls==['plan']
