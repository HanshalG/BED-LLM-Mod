"""Output diversity of unique programs, independent of verbal family labels."""
from collections import Counter
import json
import math
from scripts.rearc_predictive_score import outcome


def summarize(graphs,predictions,num_inputs):
    if type(num_inputs) is not int or num_inputs<=0 or len(graphs)!=len(predictions):
        raise ValueError('prediction coverage')
    unique = {}
    for graph,row in zip(graphs,predictions):
        if graph is None or len(row)!=num_inputs:
            raise ValueError('convertible programs and complete predictions required')
        key = json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False)
        signature = tuple(outcome(value) for value in row)
        if key in unique and unique[key]!=signature:
            raise ValueError('inconsistent deterministic program')
        unique[key] = signature
    if not unique:
        return {'status':'empty_support','unique_programs':0,'prediction_classes':0}
    signatures = list(unique.values())
    queries = []
    for i in range(num_inputs):
        counts = Counter(row[i] for row in signatures)
        probabilities = [count/len(signatures) for count in counts.values()]
        queries.append({'distinct_outputs':len(counts),
            'entropy_nats':-sum(p*math.log(p) for p in probabilities),
            'failure_probability':counts.get(None,0)/len(signatures)})
    return {'status':'complete','unique_programs':len(unique),
        'prediction_classes':len(set(signatures)),'queries':queries,
        'weighting':'uniform_unique_programs_not_family_labels'}
