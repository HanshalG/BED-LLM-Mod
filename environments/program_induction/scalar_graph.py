"""Compile bounded, topologically ordered arithmetic graphs into safe expressions."""
import ast
import json
import math

from .scalar_expression import ScalarExpression

UNARY = ('neg', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan')
BINARY = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
OPS = ('constant', 'variable', 'pi') + UNARY + tuple(BINARY)
MAX_NODES = 64
MAX_BYTES = 32768


def schema():
    """Outer schema enforces types; compilation additionally enforces topology/arity."""
    node = dict(type='object', additionalProperties=False,
                required=['op', 'args', 'value', 'variable'], properties={
                    'op': {'type': 'string', 'enum': list(OPS)},
                    'args': {'type': 'array', 'maxItems': 2, 'items': {'type': 'integer', 'minimum': 0}},
                    'value': {'type': ['number', 'null']},
                    'variable': {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 7}})
    return dict(type='object', additionalProperties=False, required=['graphs'], properties={
        'graphs': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['nodes'],
            'properties': {'nodes': {'type': 'array', 'minItems': 1, 'maxItems': MAX_NODES,
                                     'items': node}}}}})


def compile_graph(graph, dimensions):
    if type(dimensions) is not int or not 1 <= dimensions <= 8:
        raise ValueError('invalid dimensions')
    if type(graph) is not dict or set(graph) != {'nodes'}:
        raise ValueError('invalid graph object')
    nodes = graph['nodes']
    if type(nodes) is not list or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError('graph node cap')
    expressions = []
    for index, node in enumerate(nodes):
        if type(node) is not dict or set(node) != {'op', 'args', 'value', 'variable'}:
            raise ValueError('invalid graph node')
        op, args, value, variable = (node[k] for k in ('op', 'args', 'value', 'variable'))
        if type(op) is not str or op not in OPS or type(args) is not list:
            raise ValueError('invalid graph operation')
        arity = 2 if op in BINARY else 1 if op in UNARY else 0
        if len(args) != arity or any(type(i) is not int or not 0 <= i < index for i in args):
            raise ValueError('invalid arity or backward reference')
        if op != 'constant' and value is not None or op != 'variable' and variable is not None:
            raise ValueError('unused field must be null')
        if op == 'constant':
            if type(value) not in (int, float):
                raise ValueError('invalid constant')
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('nonfinite constant')
            expression = repr(value)
        elif op == 'variable':
            if type(variable) is not int or not 0 <= variable < dimensions:
                raise ValueError('invalid variable index')
            expression = f'x{variable}'
        elif op == 'pi':
            expression = 'pi'
        elif op in BINARY:
            expression = f'({expressions[args[0]]}{BINARY[op]}{expressions[args[1]]})'
        elif op == 'neg':
            expression = f'(-{expressions[args[0]]})'
        else:
            expression = f'{op}({expressions[args[0]]})'
        # Expansion is bounded at every node, including unused nodes; DAG sharing
        # must never bypass the existing interpreter's expression resource caps.
        if len(expression.encode()) > 8192:
            raise ValueError('expanded expression byte cap')
        ScalarExpression(expression, [f'x{i}' for i in range(dimensions)])
        expressions.append(expression)
    return expressions[-1]


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode(text, dimensions):
    if type(text) is not str or len(text.encode()) > MAX_BYTES:
        raise ValueError('graph response byte cap')
    try:
        data = json.loads(text, object_pairs_hook=_object)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid graph JSON') from error
    if type(data) is not dict or set(data) != {'graphs'} or type(data['graphs']) is not list or not 1 <= len(data['graphs']) <= 8:
        raise ValueError('invalid graph collection')
    expressions, seen = [], set()
    for graph in data['graphs']:
        expression = compile_graph(graph, dimensions)
        key = ast.dump(ScalarExpression(expression, [f'x{i}' for i in range(dimensions)]).tree)
        if key not in seen:
            expressions.append(expression)
            seen.add(key)
    return expressions
