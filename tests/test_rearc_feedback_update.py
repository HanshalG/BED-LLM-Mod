import json
import pytest
from scripts.rearc_feedback_update import propose_and_repair, proposal_feedback, messages
from scripts.rearc_feedback_cohort import select

DSL='def identity(x: Grid) -> Grid: pass'
G={'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
TEXT=json.dumps({'hypotheses':[G]*4})


def test_fixed_two_calls_and_only_observed_examples_executed():
    requests=[]; seen=[]
    def request(phase,prompt): requests.append((phase,prompt)); return TEXT
    def evaluate(g,x): seen.append(x); return {'status':'ok','output':x}
    r=propose_and_repair(inputs=[[[0]],[[8]]],observations=[{'index':0,'output':[[1]]}],
        dsl_source=DSL,request=request,evaluate=evaluate)
    assert [p for p,_ in requests]==['proposal','repair']
    assert len(seen)==8 and all(x==[[0]] for x in seen)
    assert len(r['graphs'])==8
    fb=r['proposal_feedback']['programs'][0][0]
    assert fb['first_mismatch']=={'row':0,'column':0,'predicted':0,'observed':1}


def test_malformed_batch_repair_not_partial_salvage():
    replies=iter(['{}',TEXT])
    r=propose_and_repair(inputs=[[[0]]],observations=[{'index':0,'output':[[0]]}],
        dsl_source=DSL,request=lambda *args:next(replies),evaluate=lambda g,x:{'status':'ok','output':x})
    assert r['proposal_feedback']['status']=='invalid_batch'
    assert len(r['graphs'])==4
    assert r['model_calls']==2


def test_feedback_privacy_rejects_extra_labels():
    with pytest.raises(ValueError):
        proposal_feedback(TEXT,[[[0]]],[{'index':0,'output':[[0]],'hidden_target':[[9]]}],DSL,
                          lambda *args:pytest.fail('execution should not open'))
    prompt=messages([[[0]]],[{'index':0,'output':[[0]]}],DSL)
    assert 'I is not public_inputs' in prompt[0]['content']


def test_selection_disjoint_permutation_invariant():
    inventory=[f'{i:08x}' for i in range(400)]
    excluded=inventory[:4]
    chosen=select(inventory,excluded)
    assert len(chosen)==8 and not set(chosen)&set(excluded)
    assert chosen==select(list(reversed(inventory)),excluded)
