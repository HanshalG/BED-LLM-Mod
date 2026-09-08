"""Exact conditioning on deterministic public context in finite reference worlds."""

from fractions import Fraction
from collections.abc import Mapping


def condition_on_public_context(weights, context_by_world, observed_context):
    """Return prior weights conditional on an already visible context identity.

    Context identities must bind ALL public inputs (code, logs, task text, etc.).
    They are supplied by the caller: equality here cannot certify redaction or
    that the real policy received only those inputs. Stochastic context needs a
    likelihood model instead and is outside this deterministic helper's scope.
    """
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError('nonempty world-weight mapping required')
    if not isinstance(context_by_world, Mapping) or weights.keys() != context_by_world.keys():
        raise ValueError('context coverage must exactly match worlds')
    if any(not isinstance(world, str) or not world for world in weights):
        raise ValueError('nonempty string world identities required')
    contexts = [observed_context, *context_by_world.values()]
    if any(not isinstance(context, str) or not context for context in contexts):
        raise ValueError('nonempty public-context identities required')
    exact = {}
    for world, value in weights.items():
        if type(value) not in (int, str, Fraction):
            raise ValueError('exact rational weights required')
        try:
            exact[world] = Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise ValueError('invalid rational weight') from None
        if exact[world] < 0:
            raise ValueError('negative weight')
    total = sum(exact.values(), Fraction())
    retained = sum((weight for world, weight in exact.items()
                    if context_by_world[world] == observed_context), Fraction())
    if not total or not retained:
        raise ValueError('observed context has zero prior probability')
    posterior = {world: weight / retained
                 if context_by_world[world] == observed_context else Fraction()
                 for world, weight in exact.items()}
    return {
        'weights': posterior,
        'context_probability': retained / total,
        'positive_support_size': sum(weight > 0 for weight in posterior.values()),
        'paid_calls_authorized': False,
        'scientific_pass': False,
    }
