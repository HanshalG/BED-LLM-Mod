"""Proper whole-grid and fixed-domain marginal scores; no survivor filtering."""
from collections import defaultdict
import math
from scripts.rearc_graph_worker import grid

PADDING = 10
FAILED = 11
SIDE = 30


def outcome(value):
    if value is None:
        return None
    try:
        return grid(value)
    except ValueError:
        return None


def padded(value):
    if value is None:
        return (FAILED,)*(SIDE*SIDE)
    return tuple(value[i][j] if i < len(value) and j < len(value[0]) else PADDING
                 for i in range(SIDE) for j in range(SIDE))


def mixture_scores(outputs, truth, weights=None):
    """Brier scores divided by two, hence bounded in [0,1].

    None denotes runtime failure, a distinct forecast outcome, never discarded.
    These scores assess supplied predictions; they do not perform posterior fitting.
    """
    truth = grid(truth)
    outputs = tuple(outputs)
    if not outputs:
        raise ValueError('empty forecast must be handled as explicit experiment failure')
    if weights is None:
        weights = (1/len(outputs),)*len(outputs)
    weights = tuple(weights)
    if len(weights) != len(outputs) or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('finite nonnegative weights required')
    if not math.isclose(sum(weights), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError('weights must already sum to one')
    mass = defaultdict(float)
    for output, weight in zip(outputs, weights):
        mass[outcome(output)] += weight
    whole = (1-2*mass.get(truth,0)+sum(p*p for p in mass.values()))/2
    truth_cells = padded(truth)
    cell_mass = [defaultdict(float) for _ in truth_cells]
    for value, probability in mass.items():
        for probabilities, cell in zip(cell_mass, padded(value)):
            probabilities[cell] += probability
    marginal = sum((1-2*p.get(y,0)+sum(v*v for v in p.values()))/2
                   for p,y in zip(cell_mass,truth_cells))/len(truth_cells)
    correct = mass.get(truth,0)
    return {'whole_grid_brier':whole, 'fixed_canvas_brier':marginal,
            'exact_grid_probability':correct, 'failure_probability':mass.get(None,0),
            'whole_grid_log_loss':-math.log(correct) if correct else math.inf}
