"""Public-only nested expressions for a new, prospectively frozen interface."""
import json
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_proposal_interface import build_messages as graph_messages
from scripts.rearc_program_graph import exports


def response_format():
    return {'type': 'json_schema', 'json_schema': {'name': 'grid_expressions', 'strict': True,
        'schema': {'type': 'object', 'additionalProperties': False, 'required': ['hypotheses'],
                   'properties': {'hypotheses': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                       'items': {'type': 'string', 'minLength': 1, 'maxLength': 8192}}}}}}


def build_messages(*, inputs, observations, dsl_source):
    messages = graph_messages(inputs=inputs, observations=observations, dsl_source=dsl_source)
    messages[0]['content'] = (
        'Infer the unknown grid transformation from the observed examples. Return exactly four '
        'plausible executable hypotheses as nested-expression strings using the supplied generic DSL. '
        'I is ONE input grid, not the list of examples. Each expression is applied independently '
        'to each input. Example syntax: hconcat(vmirror(I), I). Use only DSL function and constant '
        'names and I. No assignments, temporary variable names, literals, attributes, Python code, '
        'task identifiers or reference solutions. For an intermediate callable use '
        '__bed_call1(compose(identity, identity), I); __bed_call2 through __bed_call4 are also '
        'available. Row, object, higher-order and resizing operations are allowed. '
        'Do not treat unobserved outputs as known facts.')
    if len(json.dumps(messages).encode()) > 32768:
        raise ValueError('message size')
    return messages


def parse_response(text, dsl_source):
    if not isinstance(text, str) or len(text.encode()) > 65536:
        raise ValueError('response size')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    data = json.loads(text, object_pairs_hook=unique)
    if not isinstance(data, dict) or set(data) != {'hypotheses'}:
        raise ValueError('response fields')
    expressions = data['hypotheses']
    if not isinstance(expressions, list) or len(expressions) != 4:
        raise ValueError('four expressions required')
    functions, constants = exports(dsl_source)
    graphs = [to_graph(expr, functions, constants) for expr in expressions]
    return {'expressions': expressions, 'graphs': graphs}
