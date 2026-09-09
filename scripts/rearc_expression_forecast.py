"""Exact deterministic conditioning with no execution of zero-mass target rows."""
import json
from scripts.rearc_program_weights import condition_programs
from scripts.rearc_predictive_score import outcome
from scripts.rearc_graph_worker import grid


def forecast(graphs, case, evaluate):
    observed = [grid(y) for y in case['outputs']]
    cache, predictions, checked = {}, [], []
    for graph in graphs:
        key = json.dumps(graph,sort_keys=True)
        if key not in cache:
            row = []
            for x,y in zip(case['inputs'],observed):
                values = evaluate(graph,[x])
                if len(values) != 1:
                    raise ValueError('demonstration prediction coverage')
                row.append(values[0])
                if outcome(values[0]) != y:
                    break
            cache[key] = (row+[None]*(len(observed)-len(row)),len(row))
        row,count = cache[key]
        predictions.append(row)
        checked.append(count)
    conditioned = condition_programs(graphs,predictions,case['outputs'])
    if conditioned['failed']:
        return {'conditioning':conditioned,'outputs':[[None] for _ in case['target_inputs']],
                'weights':[1.], 'demonstrations_checked':checked}
    outputs = [[None]*len(graphs) for _ in case['target_inputs']]
    for index,(graph,weight) in enumerate(zip(graphs,conditioned['weights'])):
        if weight <= 0:
            continue
        future = evaluate(graph,case['target_inputs'])
        if len(future) != len(outputs):
            raise ValueError('target prediction coverage')
        for row,value in zip(outputs,future):
            row[index] = value
    return {'conditioning':conditioned, 'outputs':outputs, 'weights':conditioned['weights'],
            'demonstrations_checked':checked}
