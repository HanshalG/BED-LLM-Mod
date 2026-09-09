import json
import math
import random

import jsonschema
import pytest

from environments.program_induction import scalar_graph as graph
from environments.program_induction.scalar_expression import ScalarExpression


def node(op, args=None, value=None, variable=None):
    return dict(op=op, args=args or [], value=value, variable=variable)


def test_shared_trigonometric_graph_matches_manual():
    nodes = [node('variable', variable=0), node('pi'), node('div', [1, 0]),
             node('sin', [2]), node('constant', value=2), node('pow', [3, 4]),
             node('add', [5, 5]), node('sqrt', [6])]
    payload = {'graphs': [{'nodes': nodes}]}
    jsonschema.validate(payload, graph.schema())
    expression = graph.decode(json.dumps(payload), 1)[0]
    f = ScalarExpression(expression, ['x0'])
    for n in range(3, 13):
        assert f({'x0': n}) == pytest.approx(math.sqrt(2*math.sin(math.pi/n)**2))


@pytest.mark.parametrize('bad', [
    node('add', [0]), node('add', [0, 1]), node('add', [False, 0]),
    node('variable', variable=True), node('variable', variable=1),
    node('constant', value=True), node('constant', value=float('inf')),
    node('pi', value=1), node('constant', value=2, variable=0),
    node('__import__'), node('add', [-1, 0]),
])
def test_invalid_nodes_fail_closed(bad):
    with pytest.raises(ValueError):
        graph.compile_graph({'nodes': [node('constant', value=1), bad]}, 1)


def test_cycles_extra_keys_and_caps():
    with pytest.raises(ValueError):
        graph.compile_graph({'nodes': [node('neg', [0])]}, 1)
    with pytest.raises(ValueError):
        graph.compile_graph({'nodes': [node('pi')], 'code': 'evil'}, 1)
    with pytest.raises(ValueError):
        graph.compile_graph({'nodes': [node('pi')]*65}, 1)
    with pytest.raises(ValueError):
        graph.compile_graph({'nodes': [node('constant', value=10**1000)]}, 1)
    nodes = [node('variable', variable=0)]
    nodes += [node('add', [i-1, i-1]) for i in range(1, 30)]
    with pytest.raises(ValueError, match='cap'):
        graph.compile_graph({'nodes': nodes}, 1)


def test_strict_collection_and_deduplication():
    g = {'nodes': [node('pi')]}
    assert graph.decode(json.dumps({'graphs': [g, g]}), 1) == ['pi']
    for text in ('{"graphs":[],"graphs":[]}', '{"graphs":[]}', 'null', ' '*32769):
        with pytest.raises(ValueError):
            graph.decode(text, 1)
    with pytest.raises(ValueError):
        graph.decode(json.dumps({'graphs': [g, {'nodes': [node('bad')]}]}), 1)


def test_domain_errors_are_not_repaired():
    f = graph.compile_graph({'nodes': [node('variable', variable=0), node('sqrt', [0])]}, 1)
    with pytest.raises(ValueError):
        ScalarExpression(f, ['x0'])({'x0': -1.})


def test_random_arithmetic_graphs_match_direct_values():
    rng = random.Random(72001)
    for _ in range(100):
        nodes = [node('variable', variable=0)]
        values = [rng.uniform(.5, 2)]
        x = values[0]
        for index in range(1, 9):
            a, b = rng.randrange(index), rng.randrange(index)
            op = rng.choice(['add', 'sub', 'mul'])
            nodes.append(node(op, [a, b]))
            values.append({'add': lambda: values[a]+values[b],
                           'sub': lambda: values[a]-values[b],
                           'mul': lambda: values[a]*values[b]}[op]())
        f = ScalarExpression(graph.compile_graph({'nodes': nodes}, 1), ['x0'])
        assert f({'x0': x}) == pytest.approx(values[-1])


def test_graph_prompt_preserves_public_only_history_contract():
    from environments.program_induction.physics_graph_proposals import messages
    a = messages(['a'], 'context', {'a': 'scale'}, [({'a': 1.}, 0.)], ['x0'])
    b = messages(['a'], 'context', {'a': 'scale'}, [({'a': 1.}, 0.), ({'a': 2.}, 12.345)], ['x0'])
    assert a[0] == b[0]
    assert '12.345' not in json.dumps(a)
    assert '12.345' in json.dumps(b)
    assert 'graphs array' in a[0]['content']
    assert 'expressions array' not in a[0]['content']
    assert set(json.loads(a[1]['content'])) == {
        'variables', 'history', 'context', 'variable_meanings',
        'initial_proposal_diagnostics', 'public_domain', 'log_scale_prior'}
