import json
import pytest
from scripts.rearc_representation_update import representation_update, interpret_python

DSL = 'def identity(x: Any) -> Any:\n return x\n'
CODE = 'def transform(grid):\n return grid\n'


@pytest.mark.parametrize('order', [('python','dsl'), ('dsl','python')])
@pytest.mark.parametrize('invalid', [False, True])
def test_equal_calls_slots_public_feedback_and_arm_isolation(order, invalid):
    calls = {}
    def request(stage, prompt, response_schema):
        calls[stage] = prompt
        if stage == 'plan':
            assert 'dsl' not in json.loads(prompt[1]['content'])
            return json.dumps({f'p{i}':'Identity' for i in range(4)})
        python = stage.startswith('python')
        body = json.loads(prompt[1]['content'])
        assert ('dsl' in body) != python
        assert body['plan'] == {f'p{i}':'Identity' for i in range(4)}
        if stage.endswith('repair'):
            original = json.loads(prompt[2]['content'])['hypotheses'][0]
            expected = ('import os' if invalid else CODE) if python else 'identity(I)'
            assert original == expected
        value = ('import os' if invalid and stage=='python_compile' else CODE) if python else 'identity(I)'
        return json.dumps({'hypotheses':[value]*8})
    seen = []
    def diagnose(program, x):
        seen.append(x)
        return {'status':'ok','output':x}
    result = representation_update(inputs=[[[1]],[[9]]], observations=[{'index':0,'output':[[1]]}],
        dsl_source=DSL, request=request, diagnose_python=diagnose, diagnose_dsl=diagnose, order=order)
    assert list(calls) == ['plan']+[a+s for a in order for s in ('_compile','_repair')]
    assert result['calls'] == 5
    assert all(len(a['slots']) == 16 for a in result['arms'].values())
    assert all(x == [[1]] for x in seen)
    assert result['arms']['python']['slots'][:8] == ([None]*8 if invalid else [CODE]*8)


def test_invalid_slot_rejects_whole_batch_without_execution():
    codes = [CODE]*7+['import os']
    slots, feedback = interpret_python(json.dumps({'hypotheses':codes}), [[[1]]],
        [{'index':0,'output':[[1]]}], lambda *a:pytest.fail('invalid batch executed'))
    assert slots == [None]*8 and feedback['accepted_slots'] == 0
    assert feedback['slot_errors'] == [{'slot':7,'error_type':'ValueError'}]


def test_python_runtime_failure_does_not_leak_exception_text():
    slots, feedback = interpret_python(json.dumps({'hypotheses':[CODE]*8}), [[[1]]],
        [{'index':0,'output':[[1]]}],
        lambda *a:{'status':'failed','error_type':'ValueError','error':'PRIVATE_SENTINEL'})
    assert slots == [CODE]*8
    assert 'PRIVATE_SENTINEL' not in json.dumps(feedback)
    assert feedback['programs'][0][0]['status'] == 'execution_error'


@pytest.mark.parametrize('observations,order', [([],('python','dsl')),
    ([{'index':0,'output':[[1]]}],('python','python')),
    ([{'index':True,'output':[[1]]}],('python','dsl'))])
def test_invalid_public_contract_opens_no_calls(observations, order):
    with pytest.raises(ValueError):
        representation_update(inputs=[[[1]]], observations=observations, order=order,
            dsl_source=DSL, request=lambda *a:pytest.fail('unexpected call'),
            diagnose_python=None, diagnose_dsl=None)
