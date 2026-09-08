"""Diagnostic continuous-model risk bound from revealing the family label.

This does not reveal coefficients or noise, and is not a certified bound on
the finite quadrature objective. It is deliberately not wired into pruning.
"""

from itertools import combinations_with_replacement

import numpy as np

from environments.chembench_mopen.horizon import _integer


def family_oracle_bound(model, state, horizon, *, first_action=None):
    horizon = _integer(horizon, "horizon")
    if horizon > 3:
        raise ValueError("diagnostic horizon limited to three")
    weights = np.exp(model._state(state))
    if first_action is not None:
        first_action = model._action(first_action)
        if horizon == 0:
            raise ValueError("no first action at horizon zero")
    prefix = () if first_action is None else (first_action,)
    suffixes = tuple(
        combinations_with_replacement(range(model.num_actions), horizon - len(prefix))
    )
    risks, witnesses = [], []
    for b, x, target in zip(
        state.components, model.action_features, model.target_features, strict=True
    ):
        gram = target.T @ (model.target_weights[:, None] * target)
        candidates = []
        for suffix in suffixes:
            sequence = prefix + suffix
            precision = np.asarray(b.precision).copy()
            for action in sequence:
                precision += np.outer(x[action], x[action])
            # Given the family, expected posterior noise variance is a martingale.
            # The target leverage depends on the chosen action multiset, not y.
            risk = b.noise_variance * (
                np.trace(np.linalg.solve(precision, gram))
                + int(model.include_observation_noise)
            )
            if not np.isfinite(risk) or risk < 0:
                raise ValueError("invalid bound")
            candidates.append((float(risk), sequence))
        best = min(candidates)
        risks.append(best[0])
        witnesses.append(list(best[1]))
    return dict(
        value=float(weights @ risks),
        component_values=risks,
        component_sequences=witnesses,
        sequence_count_per_component=len(suffixes),
        interpretation="family_revelation_continuous_model_lower_bound",
        quadrature_pruning_authorized=False,
    )
