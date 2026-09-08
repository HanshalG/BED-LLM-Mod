"""Diagnostic h2 root scores with separate integration resolutions.

No policy tree or deployment API: equal-order scores must match the existing
corrected operator before unequal-order accuracy is tested independently.
"""
import math
from time import monotonic

import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded, SearchLimits, _integer
from .horizon_control_variate import HorizonControlVariateMixture


def leaf_work_bound(actions, families, orders):
    """Unmerged full-tree terminal leaves, excluding internal/materialization work."""
    actions = _integer(actions, 'actions', minimum=1)
    families = _integer(families, 'families', minimum=1)
    if not orders:
        raise ValueError('at least one resolution required')
    return math.prod(actions * families * _integer(q, 'order', minimum=2) for q in orders)


def h2_work_bound(actions, families, outer_order, inner_order):
    leaves = leaf_work_bound(actions, families, (outer_order, inner_order))
    return 1 + actions * families * outer_order + leaves


def score_h2(model, state, *, inner_order, limits=None):
    if type(model) is not HorizonControlVariateMixture:
        raise ValueError('explicit corrected model required')
    inner_order = _integer(inner_order, 'inner_order', minimum=2)
    if inner_order > 128:
        raise ValueError('order exceeds cap')
    limits = limits or SearchLimits(max_nodes=100000, max_seconds=5)
    if limits.max_depth < 2:
        raise SearchLimitExceeded('depth-two exceeds depth cap')
    bound = h2_work_bound(model.num_actions, len(state.components),
                         model.quadrature_order, inner_order)
    if bound > limits.max_nodes:
        raise SearchLimitExceeded('unmerged diagnostic work exceeds cap before evaluation')
    start = monotonic()
    inner = HorizonControlVariateMixture(
        model.action_features, model.target_features, state.components,
        np.exp(model._state(state)), target_weights=model.target_weights,
        quadrature_order=inner_order,
        include_observation_noise=model.include_observation_noise)
    nodes = 1

    def charge(count=0):
        nonlocal nodes
        nodes += count
        if nodes > limits.max_nodes or monotonic()-start > limits.max_seconds:
            raise SearchLimitExceeded('split-resolution diagnostic exceeded limits')

    roots = []
    for action in range(model.num_actions):
        charge()
        branches = model.branches(state, action)
        charge(len(branches))
        values = []
        for branch in branches:
            terminal = []
            for next_action in range(model.num_actions):
                value, count = inner.expected_terminal_risk(branch.state, next_action)
                charge(count)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('invalid terminal risk')
                terminal.append(value)
            values.append(branch.probability * min(terminal))
        value = math.fsum(values) + model.horizon_chance_risk_correction(state, action, 2)
        charge()
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid root risk')
        roots.append((action, value))
    return dict(root_action_values=roots, action=min(roots, key=lambda av: (av[1], av[0]))[0],
                evaluated_nodes=nodes, unmerged_work_bound=bound,
                seconds=monotonic()-start, outer_order=model.quadrature_order,
                inner_order=inner_order, deployment_authorized=False)
