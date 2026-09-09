"""Exactly two public-feedback calls with fixed proposal-slot accounting."""
import json
from scripts.rearc_expression_interface import build_messages, parse_response
from scripts.rearc_feedback_update import example_feedback


def update(*, inputs, observations, dsl_source, request, diagnose):
    base = build_messages(inputs=inputs, observations=observations, dsl_source=dsl_source)
    def interpret(text):
        try:
            parsed = parse_response(text, dsl_source)
        except (ValueError, TypeError, KeyError) as error:
            return [None]*4, [], {'status': 'invalid_batch', 'error_type': type(error).__name__}
        feedback = [[{'example_index': observation['index'], **example_feedback(
            graph, inputs[observation['index']], observation['output'], diagnose)}
            for observation in observations] for graph in parsed['graphs']]
        return parsed['expressions'], parsed['graphs'], {'status': 'evaluated', 'programs': feedback}
    original = request('proposal', base)
    first, graphs, feedback = interpret(original)
    repair = base + [{'role': 'assistant', 'content': original}, {'role': 'user', 'content': json.dumps({
        'public_execution_feedback': feedback,
        'instruction': 'Return four revised or alternative nested-expression hypotheses. Repair observed errors and preserve useful diversity. Only the listed observations are facts.'}, sort_keys=True)}]
    if len(json.dumps(repair).encode()) > 32768:
        raise ValueError('repair message cap')
    second, repaired, repair_feedback = interpret(request('repair', repair))
    return {'slots': first+second, 'graphs': graphs+repaired,
            'proposal_feedback': feedback, 'repair_feedback': repair_feedback, 'calls': 2}
