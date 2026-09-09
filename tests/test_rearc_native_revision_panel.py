import hashlib
import json
import pytest
from scripts.rearc_native_revision_panel import collect,score

DSL='def identity(x: Any) -> Any:\n return x\n'
CODE='def transform(g): return g'


@pytest.mark.parametrize('fits',[True,False])
def test_real_revision_panel_exact_calls_and_endpoint_order(fits):
    second=[[2]] if fits else [[8]]
    case={'inputs':[[[1]]],'outputs':[[[1]]],'reveal_input':[[2]],
        'reveal_hash':hashlib.sha256(json.dumps(second,separators=(',',':')).encode()).hexdigest(),
        'query_inputs':[[[3]]]*2,'target_inputs':[[[3]]]*7,'target_hashes':['sealed']*9}
    calls=[];updates=[];events=[]
    def request(tag,payload):
        calls.append(tag)
        return json.dumps({f'p{i}':'identity' for i in range(4)} if tag.endswith('_plan') else {'hypotheses':[CODE]*8})
    def reveal(i):
        assert f'{i}_initial' in updates
        events.append(f'reveal{i}')
        return second
    def seal(f):
        assert len(calls)==36 and len(updates)==12
        events.append('seal')
    def targets():
        assert events[-1]=='seal'
        events.append('targets')
        return [[[[3]]]*9 for _ in range(4)]
    result=collect([case]*4,DSL,request,lambda code,xs:xs,
        lambda code,x:{'status':'ok','output':x},reveal,
        lambda tag,value:updates.append(tag),seal,targets)
    assert len(calls)==36
    assert result['endpoints_opened']==fits and not result['depth_authorized']
    assert not result['qualification_passed'] # Perfect ties cannot establish recovery.
    assert ('targets' in events)==fits


def test_support_recovery_is_distinct_from_merely_fitting():
    good={'weights':[.5,.5],'outputs':[[[[1]],[[2]]]]*9}
    bad={'weights':[1.],'outputs':[[None]]*9}
    f=[{'aware':good,'blind':bad,'initial':bad} for _ in range(4)]
    labels=[[[[1]]]*9 for _ in range(4)]
    result=score(f,labels)
    assert result['qualification_passed'] and result['recovery_tasks']==4
    f=[{'aware':good,'blind':bad,'initial':good} for _ in range(4)]
    result=score(f,labels)
    assert not result['gates']['new_support_recovery'] and not result['qualification_passed']
