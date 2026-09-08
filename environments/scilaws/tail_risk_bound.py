"""Analytic predictive-tail second-moment bound; no tail renormalization."""

import math

import numpy as np
from scipy.stats import t


def student_tail_moments(df, left, right):
    """Unnormalized E[Z**j 1{Z<left or Z>right}], j=0,1,2, Student-t(df)."""
    if not all(math.isfinite(x) for x in (df, left, right)) or df <= 2 or left >= right:
        raise ValueError("invalid finite-variance Student-t interval")

    def first(c):
        return math.exp(t.logpdf(c, df) + np.logaddexp(math.log(df),
                        2*math.log(abs(c)) if c else -math.inf) - math.log(df-1))

    def second(c):
        return (df*(df-1)/(df-2)*t.sf(c*math.sqrt((df-2)/df), df-2)
                - df*t.sf(c, df))

    mass = float(t.cdf(left, df)+t.sf(right, df))
    mean = first(right)-first(left)
    square = float(second(right)+second(-left))
    if not all(math.isfinite(x) for x in (mass, mean, square)) or mass < 0 or square < 0:
        raise ValueError("invalid tail moments")
    return mass, mean, square


def tail_risk_bound(model, state, action, left, right):
    """Bound E[optimal future Bayes risk 1{next observation outside interval}].

    Zero prediction is a feasible terminal decision. Its conditional target
    second moment upper-bounds posterior-mean risk, and expected future learning
    cannot increase optimal risk. This applies to the continuous working model,
    not arbitrary approximate quadrature operators or the source-world truth.
    """
    action = model._action(action)
    weights = np.exp(model._state(state))
    bounds, masses = [], []
    for b, features, targets in zip(state.components, model.action_features,
                                    model.target_features, strict=True):
        phi = features[action]
        df, loc, scale2 = b.predictive(phi)
        scale = math.sqrt(scale2)
        mass, first, second = student_tail_moments(
            df, (left-loc)/scale, (right-loc)/scale)
        solve = np.linalg.solve(np.asarray(b.precision), phi)
        denominator = 1 + phi @ solve
        means = targets @ np.asarray(b.mean)
        slopes = targets @ (solve/denominator)
        updated = np.asarray(b.precision) + np.outer(phi, phi)
        leverage = np.sum(targets * np.linalg.solve(updated, targets.T).T, axis=1)
        factor = float(model.target_weights @ leverage) + int(model.include_observation_noise)
        a = float(model.target_weights @ slopes**2) + factor/(2*denominator*(b.shape-.5))
        linear = float(2*model.target_weights @ (means*slopes))
        constant = float(model.target_weights @ means**2) + factor*b.scale/(b.shape-.5)
        value = a*scale2*second + linear*scale*first + constant*mass
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid conditional second-moment bound")
        bounds.append(value)
        masses.append(mass)
    return dict(value=float(weights @ bounds), probability=float(weights @ masses),
                component_bounds=bounds, interval=[left, right],
                interpretation='continuous_working_model_not_source_or_quadrature_bound')
