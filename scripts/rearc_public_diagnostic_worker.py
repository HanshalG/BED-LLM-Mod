"""Container-only public-example diagnostics; no source generator or labels."""
import json
from pathlib import Path
import resource
import sys
from rearc_graph_worker import grid
from rearc_program_graph import exports, validate_graph


def describe(value):
    if callable(value):
        return {'kind':'callable'}
    try:
        g = grid(value)
        return {'kind':'grid','height':len(g),'width':len(g[0])}
    except ValueError:
        return {'kind':type(value).__name__,
                'length':len(value) if isinstance(value,(list,tuple,set,frozenset)) else None}


def diagnose(graph, input_grid, namespace, functions, constants):
    validate_graph(graph, functions, constants)
    values = {name:namespace[name] for name in functions | constants}
    values['I'] = grid(input_grid)
    trace = []
    for step in graph['steps']:
        row = {'id':step['id'],'op':step['op'],
               'arguments':[describe(values[arg]) for arg in step['args']]}
        try:
            values[step['id']] = values[step['op']](*(values[arg] for arg in step['args']))
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            trace.append(row)
            return {'status':'execution_error','trace':trace}
        row['result'] = describe(values[step['id']])
        trace.append(row)
    try:
        output = grid(values[graph['output']])
    except ValueError:
        return {'status':'invalid_output','trace':trace,'output_type':describe(values[graph['output']])}
    return {'status':'ok','trace':trace,'output':output}


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    raw = sys.stdin.buffer.read(65537)
    if len(raw)>65536:
        raise ValueError('request size')
    request = json.loads(raw)
    if set(request) != {'graph','input'}:
        raise ValueError('public input only')
    import dsl
    functions, constants = exports(Path('/app/dsl.py').read_text())
    print(json.dumps(diagnose(request['graph'],request['input'],vars(dsl),functions,constants)))
