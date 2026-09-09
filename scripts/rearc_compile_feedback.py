"""Prospective bounded compiler feedback, not a change to closed run acceptance."""
from scripts.rearc_saved_compile_diagnostic import diagnose

HINTS = {
    'expression syntax': 'Use one balanced nested expression, with no assignments.',
    'computed call arity': '__bed_callN takes a callable followed by exactly N arguments; for example __bed_call1(identity,I).',
    'unknown function': 'Use only function names in the supplied DSL.',
    'unknown terminal': 'Use only I, DSL functions and supplied named constants.',
    'named positional calls only': 'Use named positional calls, no literals, attributes, keywords or lambdas.',
    'expression size': 'Keep each expression within the compiler byte limit.',
    'expression nodes': 'Reduce expression complexity to the fixed AST-node limit.',
}


def compiler_feedback(text, dsl):
    result = diagnose(text, dsl)
    if result['batch_valid']:
        return {'status': 'structurally_valid', 'semantic_validity': 'not_tested'}
    if not result['slots']:
        return {'status': 'invalid_batch', 'error': 'response_schema',
                'instruction': 'Return exactly eight expression strings in the hypotheses field.'}
    return {'status': 'invalid_batch', 'accepted_slots': 0,
            'instruction': 'Return one complete replacement batch; structural validity does not imply correctness.',
            'slot_errors': [{'slot': row['slot'],
                'error': row['error'] if row['error'] in HINTS else 'graph_validation',
                'hint': HINTS.get(row['error'], 'Respect the supplied graph and DSL limits.')}
                for row in result['slots'] if not row['valid']]}
