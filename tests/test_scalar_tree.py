import json
import math
import pytest
import jsonschema
from environments.program_induction import scalar_tree as tree
from environments.program_induction.scalar_expression import ScalarExpression

C = {'op': 'constant', 'value': 2}
X = {'op': 'variable', 'name': 'x0'}


@pytest.mark.parametrize('op', list(tree.BINARY)+list(tree.UNARY))
def test_every_operation_matches_reference(op):
    if op in tree.BINARY:
        node = dict(op=op, left=X, right=C)
        source = f'(x0{tree.BINARY[op]}2)'
    else:
        node = dict(op=op, arg=X)
        source = '-x0' if op == 'neg' else f'{op}(x0)'
    payload = {'trees': [node]}
    jsonschema.validate(payload, tree.schema(1))
    compiled = tree.decode(json.dumps(payload), 1)[0]
    for x in (.5, 1., 1.3):
        assert ScalarExpression(compiled, ['x0'])({'x0': x}) == pytest.approx(ScalarExpression(source, ['x0'])({'x0': x}))


@pytest.mark.parametrize('node', [dict(op='sin', left=X, right=C), dict(op='add', left=X),
    dict(op='constant', value=True), dict(op='constant', value=float('inf')),
    dict(op='variable', name='x8'), dict(op='ref', index=1), dict(op='pi', args=[]),
    dict(op='__import__', arg=C)])
def test_bad_trees_fail(node):
    with pytest.raises(ValueError):
        tree.compile_tree(node, 1)


def test_caps_cycles_and_collection():
    cyclic = {'op': 'sin'}
    cyclic['arg'] = cyclic
    with pytest.raises(ValueError, match='cap'):
        tree.compile_tree(cyclic, 1)
    wide = C
    for _ in range(8):
        wide = dict(op='add', left=wide, right=wide)
    with pytest.raises(ValueError, match='cap'):
        tree.compile_tree(wide, 1)
    for raw in ('{"trees":[],"trees":[]}', 'null', ' '*32769):
        with pytest.raises(ValueError):
            tree.decode(raw, 1)
    assert tree.decode(json.dumps({'trees': [C, C]}), 1) == ['2']
    with pytest.raises(ValueError):
        tree.decode(json.dumps({'trees': [C, {'op': 'bad'}]}), 1)


def test_nested_pi_and_domain():
    node = dict(op='sin', arg=dict(op='div', left={'op': 'pi'}, right=X))
    expression = tree.compile_tree(node, 1)
    assert ScalarExpression(expression, ['x0'])({'x0': 4}) == pytest.approx(math.sin(math.pi/4))
    bad_domain = tree.compile_tree(dict(op='sqrt', arg=dict(op='neg', arg=X)), 1)
    with pytest.raises(ValueError):
        ScalarExpression(bad_domain, ['x0'])({'x0': 1})
