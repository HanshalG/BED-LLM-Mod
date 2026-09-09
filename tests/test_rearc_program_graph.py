import pytest
from scripts.rearc_program_graph import exports, source_graph, validate_graph


def test_literals_not_type_aliases_and_higher_order_calls():
    funcs, constants = exports('from typing import Tuple\nGrid=Tuple[int]\nONE=1\ndef compose(a,b): pass\ndef mirror(g): pass')
    assert constants == {'ONE'}
    graph = source_graph('def f(I):\n x0=compose(mirror,mirror)\n x1=x0(I)\n return x1', funcs, constants)
    assert graph['steps'][1]['op'] == 'x0'


@pytest.mark.parametrize('body', ['import os', 'while True: pass', 'x0=I.__class__()', 'x0=eval(I)', 'x0=mirror(x1)'])
def test_unsupported_source_is_not_executed(body):
    with pytest.raises(ValueError):
        source_graph('def f(I):\n '+body+'\n return x0', {'mirror'}, set())
