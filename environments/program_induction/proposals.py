"""No-network, strict executable proposal interface for active DeepCoder."""
import json


SEMANTICS = {
    'Head': 'first list element; ERROR on empty list',
    'Last': 'last list element; ERROR on empty list',
    'Take': 'Python xs[:n], including Python negative slice semantics',
    'Drop': 'Python xs[n:], including Python negative slice semantics',
    'Access': 'xs[n] only when 0 <= n < len(xs); otherwise ERROR',
    'Minimum': 'minimum list element; ERROR on empty list',
    'Maximum': 'maximum list element; ERROR on empty list',
    'Reverse': 'reverse the list',
    'Sort': 'sort ascending',
    'Sum': 'sum list elements; zero for empty list',
    'Map': 'apply the chosen unary integer lambda to each element',
    'Filter': 'keep elements satisfying the chosen predicate',
    'Count': 'count elements satisfying the chosen predicate',
    'ZipWith': 'apply binary lambda to paired elements; stop at shorter list',
    'Scanl1': 'left cumulative binary lambda; retain first element; empty gives empty',
}


def _history(history):
    if not isinstance(history, (list, tuple)) or len(history) > 4:
        raise ValueError('at most four real observations required')
    result = []
    for row in history:
        if not isinstance(row, dict) or set(row) != {'inputs', 'output'}:
            raise ValueError('history row schema mismatch')
        xs, y = row['inputs'], row['output']
        if (not isinstance(xs, list) or len(xs) != 2
                or any(type(x) is not list or not 1 <= len(x) <= 5
                       or any(type(v) is not int or not -10 <= v <= 10 for v in x) for x in xs)):
            raise ValueError('input outside frozen public input law')
        if y is not None and not (
            type(y) is int and -50 <= y <= 50
            or type(y) is list and len(y) <= 5
            and all(type(v) is int and -50 <= v <= 50 for v in y)
        ):
            raise ValueError('invalid bounded output; null denotes ERROR')
        result.append({'inputs': [list(x) for x in xs],
                       'output': list(y) if isinstance(y, list) else y})
    return result


def messages(dsl, history, *, history_blind=False):
    validated = _history(history)
    if type(history_blind) is not bool:
        raise ValueError('history_blind must be boolean')
    if {op.token for op in dsl.OPERATIONS} != set(SEMANTICS):
        raise ValueError('unexpected interpreter vocabulary')
    operations = []
    for op in dsl.OPERATIONS:
        higher = isinstance(op, dsl.HigherOrderOperation)
        arg_types = op.inputs_type[1:] if higher else op.inputs_type
        spec = dict(op=op.token, args=[t.__name__ for t in arg_types],
                    output_type=op.output_type.__name__, semantics=SEMANTICS[op.token])
        if higher:
            spec['lambda_choices'] = [lam.token for lam in dsl.LAMBDAS
                                      if (lam.inputs_type, lam.output_type) == op.inputs_type[0]]
        operations.append(spec)
    instructions = (
        'Infer alternative executable programs explaining the observed examples. '
        'Return only JSON: {"programs":[{"steps":[...]}]}. Return 1 to 8 programs, '
        'each with 2 to 4 steps. Inputs x0,x1 are integer lists. Step i assigns x(i+2). '
        'Each step has exactly op and args; higher-order steps also have lambda. '
        'args are names of earlier variables, not literals. After the first step, '
        'each step must use the preceding step result. The last result is the output. '
        'Any intermediate out-of-range integer or undefined operation gives ERROR '
        'for the entire program. All intermediate integers must be in [-50,50]. '
        'JSON null in an observed output denotes ERROR. Lambda divisions use floor '
        'division, including negative integers. No prose, Python, weights or fitted constants.'
        ' The source prior chooses length uniformly from 2,3,4 and each subsequent '
        'step uniformly from the type-valid operation/lambda/argument tuples satisfying '
        'the previous-result rule. Input list lengths are uniform 1..5 and elements '
        'uniform -10..10, independently of the hidden program.'
    )
    return [{'role': 'system', 'content': instructions},
            {'role': 'user', 'content': json.dumps(
                {'operations': operations, 'history': [] if history_blind else validated},
                sort_keys=True, separators=(',', ':'))}]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def parse_proposals(dsl, text):
    """Validate the WHOLE batch before returning canonical syntax-deduped programs.

    Programs need not fit the history to be syntactically valid. A future runner
    must separately record consistency and predictive quality, without silently
    regenerating responses. This parser supplies no posterior weights.
    """
    if type(text) is not str or len(text.encode('utf-8')) > 32768:
        raise ValueError('response must be bounded text')
    try:
        data = json.loads(text, object_pairs_hook=_unique_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (json.JSONDecodeError, RecursionError):
        raise ValueError('invalid JSON response') from None
    if not isinstance(data, dict) or set(data) != {'programs'}:
        raise ValueError('response schema mismatch')
    if type(data['programs']) is not list or not 1 <= len(data['programs']) <= 8:
        raise ValueError('one to eight programs required')
    programs = {}
    for candidate in data['programs']:
        if not isinstance(candidate, dict) or set(candidate) != {'steps'}:
            raise ValueError('program schema mismatch')
        steps = candidate['steps']
        if type(steps) is not list or not 2 <= len(steps) <= 4:
            raise ValueError('two to four steps required')
        types = {'x0': list, 'x1': list}
        statements = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or type(step.get('op')) is not str:
                raise ValueError('invalid operation')
            op = dsl.TOKEN_TO_OPERATION.get(step['op'])
            if op is None:
                raise ValueError('unknown operation')
            higher = isinstance(op, dsl.HigherOrderOperation)
            if set(step) != ({'op', 'args', 'lambda'} if higher else {'op', 'args'}):
                raise ValueError('step schema mismatch')
            args = step['args']
            expected = op.inputs_type[1:] if higher else op.inputs_type
            if (type(args) is not list or len(args) != len(expected)
                    or any(type(arg) is not str or types.get(arg) != typ
                           for arg, typ in zip(args, expected))):
                raise ValueError('argument scope or type mismatch')
            if i and f'x{i+1}' not in args:
                raise ValueError('each subsequent step must consume previous result')
            compiled_args = list(args)
            if higher:
                if type(step['lambda']) is not str:
                    raise ValueError('lambda must be a vocabulary token')
                lam = dsl.TOKEN_TO_LAMBDA.get(step['lambda'])
                if lam is None or (lam.inputs_type, lam.output_type) != op.inputs_type[0]:
                    raise ValueError('lambda type mismatch')
                compiled_args.insert(0, lam)
            name = f'x{i+2}'
            statements.append(dsl.Statement(name, op, compiled_args))
            types[name] = op.output_type
        program = dsl.Program(['x0', 'x1'], statements)
        programs.setdefault(str(program), program)
    return tuple(programs.values())
