import json
import math
import jsonschema
import pytest
from environments.program_induction import physics_edits as edits
from environments.program_induction.scalar_expression import ScalarExpression


def payload(operation, correction):
    return {'edits': [dict(operation=operation, correction=correction)]}


@pytest.mark.parametrize('operation', ['add', 'multiply'])
def test_compiled_edit_preserves_base(operation):
    correction = dict(op='sin', arg=dict(op='variable', name='x1'))
    data = payload(operation, correction)
    jsonschema.validate(data, edits.schema(2))
    base = 'x0/(1+x0)'
    compiled = edits.decode(json.dumps(data), 2, base)[0]
    f = ScalarExpression(compiled['expression'], ['x0', 'x1'])
    for x, y in ((.5, .7), (1., 1.), (2., 1.4)):
        expected = x/(1+x)+math.sin(y) if operation == 'add' else x/(1+x)*math.sin(y)
        assert f({'x0': x, 'x1': y}) == pytest.approx(expected)


def test_negative_correction_and_invalid_domain_not_repaired():
    text = json.dumps(payload('add', dict(op='constant', value=-1)))
    expr = edits.decode(text, 1, 'x0')[0]['expression']
    assert ScalarExpression(expr, ['x0'])({'x0': .5}) == -.5
    # Compilation is not a positivity gate: the numerical updater owns that check.


def test_strict_batch_and_untrusted_base():
    good = payload('add', dict(op='constant', value=1))
    doubled = {'edits': good['edits']*2}
    assert len(edits.decode(json.dumps(doubled), 1, 'x0')) == 1
    for text in ('{"edits":[]}', '{"edits":[],"edits":[]}', ' '*32769,
                 json.dumps(payload('replace', dict(op='pi')))):
        with pytest.raises(ValueError):
            edits.decode(text, 1, 'x0')
    with pytest.raises(ValueError):
        edits.decode(json.dumps(good), 1, '__import__("os")')
    good['edits'].append(dict(operation='add', correction=dict(op='unknown')))
    with pytest.raises(ValueError):
        edits.decode(json.dumps(good), 1, 'x0')


def test_public_history_is_only_diagnostic_source():
    args = (['a'], 'semantic context', {'a': 'input'})
    a = edits.messages(*args, [({'a': 1.}, 0.)], 'x0')
    b = edits.messages(*args, [({'a': 1.}, 0.), ({'a': 2.}, 12.345)], 'x0')
    assert a[0] == b[0]
    assert '12.345' not in json.dumps(a) and '12.345' in json.dumps(b)
    p = json.loads(a[1]['content'])
    assert p['base_expression'] == 'x0'
    assert not {'targets', 'truth', 'equation', 'task_id'} & set(p)
