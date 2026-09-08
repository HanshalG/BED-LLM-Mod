"""Preserve pre-observation roots so a surprising answer can seed local repair."""
from .local_support import evaluate, expand
from .proposals import _history


def prepare(dsl, previous_pool, previous_history):
    """Precompute one-hop alternatives before the next answer is known."""
    history = _history(previous_history)
    if len(history) >= 4:
        raise ValueError('no room for another observation')
    return expand(dsl, previous_pool, history)


def condition(candidates, previous_history, observation):
    history = _history([*_history(previous_history), observation])
    return [p for p in candidates if all(evaluate(p, h['inputs']) == h['output'] for h in history)]
