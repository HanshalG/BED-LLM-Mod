"""Convert nested data-only expressions, never executing candidate Python."""
import ast
from scripts.rearc_program_graph import validate_graph


def to_graph(expression, functions, constants):
    if not isinstance(expression, str) or len(expression.encode()) > 8192:
        raise ValueError('expression size')
    try:
        tree = ast.parse(expression, mode='eval').body
    except (SyntaxError, RecursionError) as error:
        raise ValueError('expression syntax') from error
    if sum(1 for _ in ast.walk(tree)) > 1024:
        raise ValueError('expression nodes')
    steps, memo = [], {}

    def lower(node):
        if isinstance(node, ast.Name):
            if node.id not in functions | constants | {'I'}:
                raise ValueError('unknown terminal')
            return node.id
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError('named positional calls only')
        key = ast.dump(node)
        if key in memo:
            return memo[key]
        name, args = node.func.id, node.args
        if name in {f'__bed_call{i}' for i in range(1, 5)}:
            if len(args) != int(name[-1]) + 1:
                raise ValueError('computed call arity')
            op, args = lower(args[0]), args[1:]
        elif name in functions:
            op = name
        else:
            raise ValueError('unknown function')
        refs = [lower(arg) for arg in args]
        identifier = f'x{len(steps)}'
        steps.append({'id': identifier, 'op': op, 'args': refs})
        memo[key] = identifier
        return identifier

    output = lower(tree)
    if not steps:
        steps.append({'id': 'x0', 'op': 'identity', 'args': [output]})
        output = 'x0'
    graph = {'steps': steps, 'output': output}
    validate_graph(graph, functions, constants)
    return graph
