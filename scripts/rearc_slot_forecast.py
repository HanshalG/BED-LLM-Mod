"""Map finite-program conditioning back to fixed attempted-candidate slots."""
from scripts.rearc_expression_forecast import forecast


def forecast_slots(slots, case, evaluate):
    if any(row['slot'] != i for i, row in enumerate(slots)):
        raise ValueError('ordered slot coverage')
    indices = [i for i, row in enumerate(slots) if row['graph'] is not None]
    result = forecast([slots[i]['graph'] for i in indices], case, evaluate)
    result['candidate_slots'] = len(slots)
    result['convertible_slots'] = indices
    result['replacement_candidates'] = 0
    checked = [0]*len(slots)
    for i, value in zip(indices, result['demonstrations_checked']):
        checked[i] = value
    result['demonstrations_checked'] = checked
    # Invalid syntax is not a program hypothesis. After nonempty observations it
    # has no posterior mass; valid programs that fail on future inputs retain mass.
    weights = [0.]*len(slots)
    for i, value in zip(indices, result['conditioning']['weights']):
        weights[i] = value
    result['conditioning']['weights'] = weights
    if not result['conditioning']['failed']:
        outputs = [[None]*len(slots) for _ in result['outputs']]
        for target, values in zip(outputs, result['outputs']):
            for i, value in zip(indices, values):
                target[i] = value
        result['outputs'] = outputs
        result['weights'] = weights
    return result
