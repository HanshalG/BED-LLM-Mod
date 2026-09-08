"""A source-complete typed continuation language, without repair or transport."""
import json

from .prior import program_probability, statement_options
from .proposals import _unique_keys


def _options(dsl, types):
    variables = [(f'x{i}', typ) for i, typ in enumerate(types)]
    name = f'x{len(types)}'
    return {str(dsl.Statement(name, op, args)): (op, args)
            for op, args in statement_options(dsl, variables, len(types) > 2)}


def schema(dsl):
    """All accepted paths are 2--4 source-valid steps, via typed finite states.

    This describes the language; provider support/enforcement is NOT assumed.
    Local decoding repeats the transition checks regardless of provider behavior.
    """
    definitions = {}

    def state(types):
        key = ''.join('l' if typ is list else 'i' for typ in types)
        if key in definitions:
            return {'$ref': '#/$defs/'+key}
        depth = len(types)-2
        alternatives = [{'type': 'null'}] if depth >= 2 else []
        if depth < 4:
            options = _options(dsl, types)
            for typ in (int, list):
                labels = [text for text, (op, _) in options.items() if op.output_type is typ]
                if not labels:
                    continue
                alternatives.append(dict(type='object', additionalProperties=False,
                    required=['statement', 'next'], properties={
                        'statement': {'type': 'string', 'enum': labels},
                        'next': state(types+(typ,)),
                    }))
        definitions[key] = alternatives[0] if len(alternatives) == 1 else {'anyOf': alternatives}
        return {'$ref': '#/$defs/'+key}

    root = state((list, list))
    return dict(type='object', additionalProperties=False, required=['programs'],
                properties={'programs': dict(type='array', minItems=1, maxItems=8, items=root)},
                **{'$defs': definitions})


def encode(dsl, programs):
    """Canonical source syntax maps bijectively to a continuation path."""
    if not 1 <= len(programs) <= 8:
        raise ValueError('one to eight source programs required')
    paths = []
    for program in programs:
        program_probability(dsl, program)
        path = None
        for statement in reversed(program.statements):
            path = {'statement': str(statement), 'next': path}
        paths.append(path)
    return {'programs': paths}


def decode(dsl, text):
    if type(text) is not str or len(text.encode()) > 32768:
        raise ValueError('bounded text required')
    try:
        data = json.loads(text, object_pairs_hook=_unique_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (json.JSONDecodeError, RecursionError):
        raise ValueError('invalid JSON') from None
    if (type(data) is not dict or set(data) != {'programs'}
            or type(data['programs']) is not list or not 1 <= len(data['programs']) <= 8):
        raise ValueError('invalid program batch')
    programs = {}
    for node in data['programs']:
        types, statements = (list, list), []
        while node is not None:
            if (len(statements) >= 4 or type(node) is not dict
                    or set(node) != {'statement', 'next'} or type(node['statement']) is not str):
                raise ValueError('invalid continuation node')
            options = _options(dsl, types)
            if node['statement'] not in options:
                raise ValueError('statement outside typed continuation language')
            op, args = options[node['statement']]
            statements.append(dsl.Statement(f'x{len(types)}', op, args))
            types += (op.output_type,)
            node = node['next']
        if len(statements) < 2:
            raise ValueError('premature termination')
        program = dsl.Program(['x0', 'x1'], statements)
        programs.setdefault(str(program), program)
    return tuple(programs.values())
