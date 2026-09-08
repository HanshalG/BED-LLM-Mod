"""Executable finite-world joint forecasts, without independence shortcuts."""
import math

from .local_support import evaluate
from .prediction import _targets, category
from .prior import restricted_weights
from .proposals import _history


def validate(law):
    if not isinstance(law, dict) or set(law) != {'worlds', 'interpretation'}:
        raise ValueError('explicit joint worlds required')
    if law['interpretation'] != 'restricted_syntax_prior_not_full_posterior':
        raise ValueError('unsupported weighting interpretation')
    worlds = law['worlds']
    if not isinstance(worlds, list) or not worlds:
        raise ValueError('nonempty world list required')
    shape = None
    keys = set()
    for world in worlds:
        if not isinstance(world, dict) or set(world) != {'key', 'weight', 'answers', 'targets'}:
            raise ValueError('invalid world fields')
        key, weight = world['key'], world['weight']
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError('unique world keys required')
        keys.add(key)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('positive finite world weight required')
        for name in ('answers', 'targets'):
            values = world[name]
            if not isinstance(values, list) or not values or any(
                    not isinstance(v, str) or not v for v in values):
                raise ValueError('nonempty categorical output lists required')
        current = (len(world['answers']), len(world['targets']))
        if shape is not None and current != shape:
            raise ValueError('world dimensions differ')
        shape = current
    if abs(math.fsum(w['weight'] for w in worlds)-1) > 1e-10:
        raise ValueError('world weights must sum to one')
    return shape


def forecast(dsl, programs, history, queries, targets):
    """Build before query/target outcomes are available; None means no support.

    Programs may come from several independent calls. Syntax duplicates do not
    gain weight through repetition. Search-selection bias remains uncorrected.
    """
    history = _history(history)
    _targets(queries)
    _targets(targets)
    supported = {}
    for p in programs:
        if all(evaluate(p, h['inputs']) == h['output'] for h in history):
            supported.setdefault(str(p), p)
    if not supported:
        return None
    weights = restricted_weights(dsl, supported.values())
    worlds = [dict(key=key, weight=float(weights[key]),
                   answers=[category(evaluate(p, x)) for x in queries],
                   targets=[category(evaluate(p, x)) for x in targets])
              for key, p in supported.items()]
    law = dict(worlds=worlds, interpretation='restricted_syntax_prior_not_full_posterior')
    validate(law)
    return law


def condition(law, observations):
    """Return joint evidence probability and branch-conditioned target marginals.

    Observations map query-column indices to labels. The empty mapping yields
    the pre-observation target law. Unsupported branches have no target law;
    there is no fabricated OTHER category or fallback to unconditional targets.
    """
    nq, nt = validate(law)
    if not isinstance(observations, dict) or any(
            type(i) is not int or not 0 <= i < nq or not isinstance(y, str) or not y
            for i, y in observations.items()):
        raise ValueError('invalid query observations')
    retained = [w for w in law['worlds']
                if all(w['answers'][i] == y for i, y in observations.items())]
    mass = math.fsum(w['weight'] for w in retained)
    if not retained:
        return dict(evidence_probability=0., distributions=None, support_size=0)
    marginals = []
    for j in range(nt):
        categories = {w['targets'][j] for w in retained}
        marginals.append({y: math.fsum(w['weight'] for w in retained
                                      if w['targets'][j] == y)/mass
                          for y in sorted(categories)})
    return dict(evidence_probability=mass, distributions=marginals,
                support_size=len(retained))
