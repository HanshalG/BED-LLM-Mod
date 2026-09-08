"""Family-revelation / best-linear-prediction terminal-risk interval."""

import math

import numpy as np


def terminal_risk_interval(model, state, action):
    action = model._action(action)
    weights = np.exp(model._state(state))
    means, locations, variances, covariances = [], [], [], []
    for b, features, targets in zip(state.components, model.action_features,
                                    model.target_features, strict=True):
        phi = features[action]
        solve = np.linalg.solve(np.asarray(b.precision), phi)
        means.append(targets @ np.asarray(b.mean))
        locations.append(float(phi @ np.asarray(b.mean)))
        variances.append(b.noise_variance*(1+phi @ solve))
        covariances.append(b.noise_variance*(targets @ solve))
    means, locations = np.asarray(means), np.asarray(locations)
    mean = weights @ means
    location = float(weights @ locations)
    variance = float(weights @ (np.asarray(variances)+(locations-location)**2))
    covariance = weights @ (np.asarray(covariances)+(means-mean)*(locations-location)[:, None])
    prior = model.risk(state)
    linear = float(prior-model.target_weights @ covariance**2/variance)
    lower = float(model.action_risk_lower_bound(state, action, 1))
    if not all(math.isfinite(x) for x in (variance, linear, lower)) or variance <= 0:
        raise ValueError('invalid linear prediction moments')
    padding = 1e-12*max(1.0, abs(prior), abs(linear), abs(lower))
    if lower > linear+2*padding or linear < -padding:
        raise ValueError('inconsistent terminal risk interval')
    lo, hi = max(0.0, lower-padding), max(0.0, linear+padding)
    return dict(lower=lo, upper=hi, width=hi-lo, linear_risk=linear,
                family_risk=lower, float_padding=padding,
                target_mean=mean.tolist(), observation_mean=location,
                slope=(covariance/variance).tolist(),
                interpretation='continuous_working_model_not_source_or_approximate_update_bound')
