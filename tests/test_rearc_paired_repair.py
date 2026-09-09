import json
import pytest
from scripts.rearc_paired_repair import paired_update

DSL = 'def identity(x: Any) -> Any:\n return x\n'


@pytest.mark.parametrize('invalid', [False, True])
@pytest.mark.parametrize('order', [('actionable','generic'), ('generic','actionable')])
def test_shared_proposal_exact_calls_and_no_cross_arm_history(invalid, order):
    calls = {}
    def request(stage, prompt, schema):
        calls[stage] = prompt
        if stage=='plan':
            return json.dumps({f'p{i}':'Identity' for i in range(4)})
        expr = '__bed_call1(I)' if invalid and stage=='compile' else 'identity(I)'
        return json.dumps({'hypotheses':[expr]*8})
    seen = []
    def diagnose(g, x):
        seen.append(x)
        return {'status':'ok','output':x}
    result = paired_update(inputs=[[[1]],[[9]]], observations=[{'index':0,'output':[[1]]}],
        dsl_source=DSL, request=request, diagnose=diagnose, order=order)
    assert list(calls)==['plan','compile']+[a+'_repair' for a in order]
    assert result['calls']==4 and all(x==[[1]] for x in seen)
    for arm in order:
        assert len(result['arms'][arm]['slots'])==16
        assert result['arms'][arm]['slots'][:8]==([None]*8 if invalid else ['identity(I)']*8)
    a,b = [calls[arm+'_repair'] for arm in ('actionable','generic')]
    assert a[:-1]==b[:-1]
    pa,pb = json.loads(a[-1]['content']), json.loads(b[-1]['content'])
    if invalid:
        assert pa.pop('compiler_feedback')['accepted_slots']==0
    assert pa==pb
    assert result['compiler_intervention_active']==invalid


def test_invalid_arm_coverage_opens_no_call():
    with pytest.raises(ValueError, match='arm coverage'):
        paired_update(inputs=[],observations=[],dsl_source=DSL,order=('generic','generic'),
            request=lambda *a:pytest.fail('unexpected paid call'), diagnose=None)
