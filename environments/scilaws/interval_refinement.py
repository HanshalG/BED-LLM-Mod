"""Conservative scheduling; numerical error estimates are not rigorous enclosures."""
import math

from .weighted_intervals import weighted_min_interval


def refine(probabilities, intervals, evaluate, *, correction=0., tolerance=5e-5):
    bounds = [[tuple(pair) for pair in row] for row in intervals]
    evaluated, trace = set(), []
    while True:
        summary = weighted_min_interval(probabilities, bounds, correction=correction,
                                        tolerance=tolerance)
        if summary['within_terminal_budget']:
            return dict(**summary, integrations=len(trace), trace=trace,
                        midpoint=(summary['lower']+summary['upper'])/2,
                        numerical_enclosures_rigorous=False)
        candidates = []
        for i, row in enumerate(bounds):
            incumbent = min(hi for _, hi in row)
            actions = [a for a, (lo, _) in enumerate(row)
                       if (i, a) not in evaluated and lo <= incumbent]
            if actions:
                a = min(actions, key=lambda a: (row[a][0], a))
                candidates.append((summary['weighted_widths'][i], i, a))
        if not candidates:
            raise ValueError('remaining uncertainty cannot meet tolerance')
        _, i, a = min(candidates, key=lambda x: (-x[0], x[1], x[2]))
        value, error = evaluate(i, a)
        if not math.isfinite(value) or not math.isfinite(error) or not 0 <= error <= 1e-7:
            raise ValueError('invalid numerical reference error')
        lo, hi = bounds[i][a]
        updated = (max(lo, value-error), min(hi, value+error))
        if updated[0] > updated[1]:
            raise ValueError('numerical reference conflicts with analytic interval')
        bounds[i][a] = updated
        evaluated.add((i, a))
        trace.append(dict(branch=i, action=a, value=value, error=error,
                          lower=updated[0], upper=updated[1]))
