import pytest
from scripts.herb_candidate_expression import to_graph

FUNCTIONS = {'identity', 'compose', 'vmirror', 'hconcat'}


def test_nested_and_shared_calls():
    graph = to_graph('hconcat(vmirror(I), vmirror(I))', FUNCTIONS, set())
    assert len(graph['steps']) == 2
    assert graph['steps'][1]['args'] == ['x0', 'x0']


def test_computed_callable():
    graph = to_graph('__bed_call1(compose(identity, identity), I)', FUNCTIONS, set())
    assert graph['steps'][1] == {'id': 'x1', 'op': 'x0', 'args': ['I']}


@pytest.mark.parametrize('expression', ['I.__class__', '__import__(I)', 'identity(x=I)',
                                       '[I]', 'lambda x: x', '__bed_call1(I)'])
def test_no_candidate_python(expression):
    with pytest.raises(ValueError):
        to_graph(expression, FUNCTIONS, set())
