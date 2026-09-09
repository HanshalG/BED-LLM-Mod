"""Opt-in fixed-target predictive-risk continuation; no model or endpoint I/O."""
from fractions import Fraction


class EmptyPredictiveSupport(ValueError):
    pass


def _support(extensions):
    values = tuple(tuple(row) for row in extensions)
    if not values:
        raise EmptyPredictiveSupport('no consistent predictive hypotheses')
    size = len(values[0])
    if not size or any(len(row) != size or any(type(v) is not bool for v in row) for row in values):
        raise ValueError('equally sized boolean extensions required')
    if len(set(values)) != len(values):
        raise ValueError('canonical unique extensions required; prior is uniform over extensions')
    return values


def _indices(indices, size):
    values = tuple(indices)
    if not values or len(set(values)) != len(values) or any(type(q) is not int or not 0 <= q < size for q in values):
        raise ValueError('nonempty unique in-range integer coordinates required')
    return values


def predictions(extensions, targets):
    support = _support(extensions)
    targets = _indices(targets, len(support[0]))
    return tuple(Fraction(sum(row[q] for row in support), len(support)) for q in targets)


def best_predictive_query(extensions, *, queries, targets):
    support = _support(extensions)
    queries = _indices(queries, len(support[0]))
    targets = _indices(targets, len(support[0]))
    values = {}
    for query in queries:
        expected = Fraction(0)
        for answer in (False, True):
            branch = tuple(row for row in support if row[query] == answer)
            if branch:
                p = predictions(branch, targets)
                expected += Fraction(len(branch), len(support))*sum(v*(1-v) for v in p)/len(p)
        values[query] = expected
    query = min(queries, key=lambda q: (values[q], q))
    return query, values[query]


def continuation(extensions, *, history, queries, targets):
    """History-only policy interface shared by prospective simulation/execution."""
    support = _support(extensions)
    queries = _indices(queries, len(support[0]))
    targets = _indices(targets, len(support[0]))
    history = tuple(history)
    for q, answer in history:
        if type(q) is not int or not 0 <= q < len(support[0]) or type(answer) is not bool:
            raise ValueError('invalid public observation')
    used = {q for q, _ in history}
    consistent = tuple(row for row in support if all(row[q] == y for q, y in history))
    return best_predictive_query(consistent, queries=tuple(q for q in queries if q not in used), targets=targets)


def terminal_brier(extensions, *, targets, truth):
    support = _support(extensions)
    truth = tuple(truth)
    if len(truth) != len(support[0]) or any(type(v) is not bool for v in truth):
        raise ValueError('invalid evaluator truth')
    targets = _indices(targets, len(truth))
    p = predictions(support, targets)
    return sum((v-int(truth[q]))**2 for q, v in zip(targets, p))/len(targets)
