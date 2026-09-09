import json
import pytest
from scripts.rearc_native_revision import paired_revision,forecast_revision

DSL='def identity(x: Any) -> Any:\n return x\n'
CODE='def transform(g): return g'


@pytest.mark.parametrize('order',[('aware','blind'),('blind','aware')])
def test_exact_calls_reveal_order_and_blind_label_privacy(order):
    prompts={}; events=[]
    def request(stage,prompt,fmt):
        prompts[stage]=prompt
        if stage.endswith('plan'):
            return json.dumps({f'p{i}':'rule' for i in range(4)})
        return json.dumps({'hypotheses':[CODE]*8})
    def reveal():
        assert events==['initial']
        assert list(prompts)==['initial_plan','initial_compile','initial_repair']
        events.append('reveal')
        return {'index':1,'output':[[8]]}
    result=paired_revision(inputs=[[[1]],[[2]],[[3]]],observation={'index':0,'output':[[1]]},
        dsl_source=DSL,request=request,diagnose=lambda p,x:{'status':'ok','output':x},
        reveal=reveal,bank_update=lambda tag,value:events.append(tag),order=order)
    assert result['calls']==len(prompts)==9
    assert events==['initial','reveal']+list(order)
    for stage,prompt in prompts.items():
        payload=json.loads(prompt[1]['content'])
        assert len(payload['observations'])==(2 if stage.startswith('aware') else 1)
        if not stage.startswith('aware'):
            assert '[[8]]' not in json.dumps(prompt)
    a=json.loads(prompts['aware_plan'][1]['content'])
    b=json.loads(prompts['blind_plan'][1]['content'])
    assert a['previous_proposals']==b['previous_proposals']
    assert all(len(result['arms'][a]['slots'])==32 for a in ('aware','blind'))


def test_both_arms_condition_on_second_answer():
    result={'observations':[{'index':0,'output':[[1]]},{'index':1,'output':[[8]]}],
            'arms':{a:{'slots':[CODE]*32} for a in ('aware','blind')}}
    seen=[]
    def evaluate(code,inputs):
        seen.extend(inputs)
        return inputs
    forecasts=forecast_revision(result,[[[1]],[[2]]],[[[3]]],evaluate)
    assert all(f['conditioning']['failed'] for f in forecasts.values())
    assert [[2]] in seen and [[3]] not in seen


def test_invalid_reveal_stops_before_branch_calls():
    calls=[]
    def request(stage,prompt,fmt):
        calls.append(stage)
        return json.dumps(
            {f'p{i}':'rule' for i in range(4)} if stage.endswith('plan') else {'hypotheses':[CODE]*8})
    with pytest.raises(ValueError):
        paired_revision(inputs=[[[1]]],observation={'index':0,'output':[[1]]},dsl_source=DSL,
            request=request,diagnose=lambda p,x:{'status':'ok','output':x},
            reveal=lambda:{'index':0,'output':[[8]]},bank_update=lambda *a:None)
    assert calls==['initial_plan','initial_compile','initial_repair']
