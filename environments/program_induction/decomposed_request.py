"""Strict, transport-free batched step construction on public observations."""
import json
import random

from . import constrained_request as cr
from .execution_steps import completed_program, next_menu, public_state
from .local_support import evaluate
from .prediction import canonical
from .proposals import _history, _unique_keys


def _paths(paths):
    if (type(paths) is not list or len(paths) != 8
            or any(type(p) is not list for p in paths)
            or len({len(p) for p in paths}) != 1 or not 0 <= len(paths[0]) < 4):
        raise ValueError('eight synchronous prefixes of length zero through three required')


def demonstrations(dsl):
    """Four fixed source-only worked examples, never selected using test outcomes."""
    examples = []
    inputs = [[[1, -2, 3], [4, 0]], [[-3, 2], [1, -1, 2]], [[0], [3, 1]]]
    for i in range(4):
        rng = random.Random(38100000+i)
        path = []
        for _ in range(rng.choice([2, 3, 4])):
            path.append(rng.randrange(len(next_menu(dsl, path))))
        program = completed_program(dsl, path)
        history = [dict(inputs=x, output=evaluate(program, x)) for x in inputs]
        examples.append(dict(program=str(program), history=history,
                             actual_execution=public_state(dsl, path, history)['rows']))
    return examples


def whole_request(dsl, history, seed):
    body = cr.request(dsl, history, seed)
    public = json.loads(body['messages'][1]['content'])
    public['worked_examples'] = demonstrations(dsl)
    body['messages'][1]['content'] = canonical(public)
    if len(canonical(body).encode()) > 32768:
        raise ValueError('complete whole-program request exceeds byte reservation')
    return body


def request(dsl, history, paths, seed, *, subgoals):
    history = _history(history)
    _paths(paths)
    if type(subgoals) is not bool or not history:
        raise ValueError('boolean mode and nonempty observed history required')
    body = whole_request(dsl, history, seed)
    menus, branches, properties = {}, [], {}
    output = {'anyOf': [{'type': 'null'}, {'type': 'integer', 'minimum': -50, 'maximum': 50},
                       {'type': 'array', 'items': {'type': 'integer', 'minimum': -50, 'maximum': 50},
                        'maxItems': 5}]}
    for i, path in enumerate(paths):
        menu = next_menu(dsl, path)
        key = canonical(menu)
        if key not in menus:
            menus[key] = 'menu'+str(len(menus))
        branches.append(dict(branch=str(i), menu=menus[key], state=public_state(dsl, path, history)))
        props = {}
        if subgoals:
            props['subgoals'] = dict(type='array', items=output, minItems=len(history), maxItems=len(history))
        props['choice'] = dict(type='integer', enum=[m['choice'] for m in menu])
        properties[str(i)] = dict(type='object', properties=props, required=list(props), additionalProperties=False)
    body['response_format']['json_schema'] = dict(name='typed_execution_steps', strict=True,
        schema=dict(type='object', properties=properties, required=list(properties), additionalProperties=False))
    # Replace the full-program shape instructions, retaining the same public DSL description.
    public = json.loads(body['messages'][1]['content'])
    public.update(branches=branches, menus={v: json.loads(k) for k, v in menus.items()})
    body['messages'] = [dict(role='system', content=(
        'Construct eight alternative source programs one step at a time. '
        'Return the required JSON object keyed by branch ID. For each branch choose '
        'one integer index from its assigned menu. Do not return Python or prose. '
        'Each independent branch will run four steps; consistent prefixes of length '
        'two, three and four are retained as hypotheses. Preserve plausible behavioral '
        'alternatives, not just different syntax for the same behavior. '
        'Observed final outputs constrain the completed program, not every intermediate step. '
        'Execution values are actual interpreter results. Null means absorbing ERROR; '
        'integers outside [-50,50] or undefined operations cause ERROR. '
        'Division uses floor division; Take/Drop use Python slice semantics. '
        'The source length prior is uniform on 2,3,4, then uniform over each type-valid '
        'statement using the preceding result. Inputs are two lists of length 1..5 '
        'with integer elements -10..10 independently of the program. '
        + ('First predict the next intermediate output for each observed example in subgoals, '
           'then choose the step. Actual execution replaces these predictions next round.' if subgoals else
           'Choose the next step using the actual execution states.'))),
        dict(role='user', content=canonical(public))]
    if len(canonical(body).encode()) > 32768:
        raise ValueError('complete step request exceeds byte reservation')
    return body


def decode(dsl, text, history, paths, *, subgoals):
    """Validate the entire batch before any branch advances; return disagreement audit."""
    import jsonschema
    body = request(dsl, history, paths, 0, subgoals=subgoals)
    if type(text) is not str or len(text.encode()) > 32768:
        raise ValueError('response byte cap exceeded')
    data = json.loads(text, object_pairs_hook=_unique_keys,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    jsonschema.validate(data, body['response_format']['json_schema']['schema'])
    # JSON Schema permits integral floats; the source path API intentionally does not.
    if any(type(data[str(i)]['choice']) is not int for i in range(8)):
        raise ValueError('integer syntax index required')
    advanced = [p+[data[str(i)]['choice']] for i, p in enumerate(paths)]
    states = [public_state(dsl, p, history) for p in advanced]
    discrepancies = []
    if subgoals:
        for i, state in enumerate(states):
            actual = [r['values'][f'x{len(advanced[i])+1}'] for r in state['rows']]
            discrepancies.append(sum(canonical(a) != canonical(b)
                                     for a, b in zip(actual, data[str(i)]['subgoals'])))
    return advanced, dict(states=states, subgoal_disagreements=discrepancies)
