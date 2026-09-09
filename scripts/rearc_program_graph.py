"""Validate straight-line DSL graphs without executing submitted code."""
import ast


def exports(source):
    functions, constants = set(), set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            constants.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return functions, constants


def source_graph(source, functions, constants):
    """Source-inspection adapter, not a candidate Python execution path."""
    module = ast.parse(source)
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError('one function required')
    function = module.body[0]
    if len(function.args.args) != 1 or function.args.args[0].arg != 'I' or function.decorator_list:
        raise ValueError('single grid argument required')
    steps = []
    for node in function.body[:-1]:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError('straight-line assignments required')
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.keywords or any(not isinstance(a, ast.Name) for a in value.args):
            raise ValueError('named positional DSL calls only')
        steps.append({'id': node.targets[0].id, 'op': value.func.id, 'args': [a.id for a in value.args]})
    last = function.body[-1]
    if not isinstance(last, ast.Return) or not isinstance(last.value, ast.Name):
        raise ValueError('named return required')
    graph = {'steps': steps, 'output': last.value.id}
    validate_graph(graph, functions, constants)
    return graph


def validate_graph(graph, functions, constants):
    if not isinstance(graph, dict) or set(graph) != {'steps', 'output'}:
        raise ValueError('graph fields')
    steps = graph['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        raise ValueError('1..128 steps required')
    values = {'I'} | set(functions) | set(constants)
    callable_values = set(functions)
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {'id', 'op', 'args'}:
            raise ValueError('step fields')
        if step['id'] != f'x{index}' or step['id'] in values:
            raise ValueError('canonical fresh step identifiers required')
        if not isinstance(step['op'], str) or step['op'] not in callable_values:
            raise ValueError('unknown callable or forward reference')
        if not isinstance(step['args'], list) or len(step['args']) > 4 or any(not isinstance(a, str) or a not in values for a in step['args']):
            raise ValueError('unknown argument or forward reference')
        values.add(step['id'])
        callable_values.add(step['id'])
    if not isinstance(graph['output'], str) or graph['output'] not in {s['id'] for s in steps}:
        raise ValueError('output must reference a step')
