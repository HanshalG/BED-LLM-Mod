"""Declared finite-pool deterministic conditioning, not selection correction."""
import json
from scripts.rearc_predictive_score import outcome
from scripts.rearc_graph_worker import grid


def condition_programs(graphs, demonstration_predictions, observed_outputs):
    observed=tuple(grid(value) for value in observed_outputs)
    if not observed or len(graphs)!=len(demonstration_predictions):
        raise ValueError('aligned programs and nonempty demonstrations required')
    records={}
    for index,(graph,predicted) in enumerate(zip(graphs,demonstration_predictions)):
        if len(predicted)!=len(observed):
            raise ValueError('missing demonstration predictions')
        key=json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False)
        normalized=tuple(outcome(value) for value in predicted)
        if key in records and records[key][1]!=normalized:
            raise ValueError('same program produced inconsistent replay')
        records.setdefault(key,(index,normalized))
    surviving=[index for index,predictions in records.values() if predictions==observed]
    weights=[0.0]*len(graphs)
    for index in surviving:
        weights[index]=1/len(surviving)
    return {'weights':weights,'unique_programs':len(records),'consistent_programs':len(surviving),
            'failed':not surviving,'failure_forecast':'unit_mass_on_execution_failure' if not surviving else None}
