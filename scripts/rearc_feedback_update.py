"""Two-call, public-history-only executable proposal update for a new protocol."""
import json
from scripts.rearc_proposal_interface import build_messages, parse_response
from scripts.rearc_graph_worker import grid

CONTRACT = (
    'Each graph is a function of ONE grid I, evaluated independently on every example. '
    'I is not public_inputs and is not a collection of examples. first(I) is a ROW '
    'of integer colors, not a grid. A Grid is a tuple of rows; a Row is a tuple of '
    'colors; an Object is a set of colored coordinates, not a Grid. Return a Grid. '
    'For illustration only, a single step x0=identity(I), output x0, is a valid '
    'grid-to-grid graph; it is not a suggested solution. Use all generic DSL '
    'operations, including valid row intermediates and higher-order callables.'
)


def messages(inputs, observations, dsl_source):
    result = build_messages(inputs=inputs,observations=observations,dsl_source=dsl_source)
    result[0]['content'] += ' '+CONTRACT
    return result


def example_feedback(graph, x, y, evaluate):
    result = evaluate(graph,x)
    status = result['status']
    if status == 'ok':
        predicted, expected = grid(result['output']), grid(y)
        height, width = max(len(predicted),len(expected)), max(len(predicted[0]),len(expected[0]))
        mismatches = []
        for r in range(height):
            for c in range(width):
                p = predicted[r][c] if r<len(predicted) and c<len(predicted[0]) else None
                e = expected[r][c] if r<len(expected) and c<len(expected[0]) else None
                if p!=e: mismatches.append({'row':r,'column':c,'predicted':p,'observed':e})
        return {'status':'matches' if predicted==expected else 'mismatch',
                'predicted_shape':[len(predicted),len(predicted[0])],
                'observed_shape':[len(expected),len(expected[0])],
                'mismatching_cells':len(mismatches),'first_mismatch':mismatches[0] if mismatches else None}
    if status in {'execution_error','invalid_output'}:
        # Only shape/type metadata from the terminal step; no arbitrary worker fields.
        terminal = result.get('trace',[])[-1:]
        return {'status':status,'terminal_step':terminal}
    if status=='runtime_failed':
        return {'status':status,'returncode':result.get('returncode')}
    raise ValueError('unknown diagnostic outcome')


def proposal_feedback(text, inputs, observations, dsl_source, evaluate):
    # Reuse public-history validation even when the candidate batch is malformed.
    messages(inputs,observations,dsl_source)
    try:
        graphs = parse_response(text,dsl_source)
    except (ValueError,TypeError,KeyError) as exc:
        return [], {'status':'invalid_batch','error_type':type(exc).__name__}
    records = []
    for graph in graphs:
        records.append([{'example_index':o['index'],
                         **example_feedback(graph,inputs[o['index']],o['output'],evaluate)}
                        for o in observations])
    return graphs, {'status':'evaluated','programs':records}


def propose_and_repair(*, inputs, observations, dsl_source, request, evaluate):
    base = messages(inputs,observations,dsl_source)
    original = request('proposal',base)
    graphs, feedback = proposal_feedback(original,inputs,observations,dsl_source,evaluate)
    repair_messages = base + [
        {'role':'assistant','content':original},
        {'role':'user','content':json.dumps({'public_execution_feedback':feedback,
            'instruction':'Return four revised or alternative hypotheses. Repair execution and observed-example errors; preserve useful diversity. Only the listed observed labels are facts.'},sort_keys=True)}]
    # This is an intermediate cap; the transport must also cap the COMPLETE body.
    if len(json.dumps(repair_messages).encode())>32768:
        raise ValueError('repair message byte cap')
    revised = request('repair',repair_messages)
    repaired, repaired_feedback = proposal_feedback(revised,inputs,observations,dsl_source,evaluate)
    return {'graphs':graphs+repaired,'proposal_feedback':feedback,
            'repair_feedback':repaired_feedback,'model_calls':2,
            'scope':'public-history finite proposal pool; not a calibrated posterior'}
