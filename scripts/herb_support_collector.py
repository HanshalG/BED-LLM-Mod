"""Bounded multi-program collection with public-history conditioning only."""
from itertools import islice
import json
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_program_weights import condition_programs


def collect(expressions, history, functions, constants, execute, *, attempts):
    if type(attempts) is not int or attempts <= 0 or not history:
        raise ValueError('positive attempt cap and public history required')
    graphs, predictions, records, seen = [], [], [], set()
    for expression in islice(expressions, attempts):
        try:
            graph = to_graph(expression, functions, constants)
        except ValueError as error:
            records.append({'status': 'invalid_expression', 'error': str(error)})
            continue
        key = json.dumps(graph, sort_keys=True)
        if key in seen:
            records.append({'status': 'duplicate'})
            continue
        seen.add(key)
        predicted = []
        for observation in history:
            result = execute(graph, observation['input'])
            predicted.append(result['output'] if result['status'] == 'ok' else None)
        graphs.append(graph)
        predictions.append(predicted)
        records.append({'status': 'evaluated', 'program_index': len(graphs)-1})
    posterior = condition_programs(graphs, predictions, [row['output'] for row in history])
    return {'graphs': graphs, 'predictions': predictions, 'posterior': posterior,
            'attempts': len(records), 'records': records,
            'evaluation_requests': len(graphs)*len(history)}
