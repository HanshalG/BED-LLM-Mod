import json
import pytest
from scripts.rearc_paired_repair_panel import collect, score


@pytest.mark.parametrize('failed',[False,True])
def test_24_calls_paired_seeds_and_sealed_endpoint(failed):
    calls, seals = {}, []
    cases = [dict(inputs=[[[0]]],outputs=[[[0]]],query_inputs=[[[1]]]*2,
        target_inputs=[[[2]]]*8,target_hashes=['sealed']*10) for _ in range(6)]
    def request(tag,body):
        calls[tag] = body
        return json.dumps({f'p{i}':'Rule' for i in range(4)} if tag.endswith('_plan')
            else {'hypotheses':['identity(I)']*8})
    def seal(values):
        assert len(calls)==24
        seals.append(values)
    def targets():
        assert seals and not failed
        return [[[[1]]]*2+[[[2]]]*8 for _ in range(6)]
    result = collect(cases,'def identity(x: Any) -> Any:\n return x\n',request,
        lambda g,x:[None]*len(x) if failed else x,lambda g,x:{'status':'ok','output':x},
        lambda *a:None,seal,targets)
    assert len(calls)==24 and len(seals)==(0 if failed else 1)
    assert not result['qualification_passed']
    for i in range(6):
        a,b = [calls[f'{i}_{arm}_repair'] for arm in ('actionable','generic')]
        assert a==b
        assert a['seed']==42402+3*i


def test_support_positive_without_intervention_still_fails():
    f=[{'actionable':{'weights':[.5,.5],'outputs':[[[[1]],[[0]]]]*10},
        'generic':{'weights':[1.],'outputs':[[[[0]]]]*10}} for _ in range(6)]
    labels = [[[[1]]]*10 for _ in range(6)]
    assert score(f,labels,[True,True,False,False,False,False])['qualification_passed']
    result=score(f,labels,[False]*6)
    assert not result['qualification_passed']
    assert set(result['means'])=={'actionable','generic'}
