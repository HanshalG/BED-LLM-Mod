"""Sample-validated value interpolation diagnostic, not a certified envelope."""

import math

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from .adaptive_reference import student_log_density
from .tail_risk_bound import tail_risk_bound
from .linear_risk_interval import terminal_risk_interval


def interpolation_check(nodes, training, checks, actual):
    interpolator = PchipInterpolator(nodes, training, axis=0, extrapolate=False)
    error = float(np.max(np.abs(interpolator(checks)-actual)))
    return interpolator, error


def reconstruct_values(baseline, residual, growth):
    return np.asarray(baseline)-np.asarray(residual)*growth


def approximate_root(reference, state, action, *, adaptive=False, linear_baseline=False,
                     quintic=False):
    if quintic and not (adaptive and linear_baseline):
        raise ValueError('quintic candidate requires adaptive linear residual')
    if linear_baseline and not adaptive:
        raise ValueError('linear baseline requires adaptive fitting')
    m = reference.model
    df, loc, scale2 = reference.density_parameters(state, action)
    logs = m._state(state)
    coords = reference.coordinates(df, loc, scale2, logs)
    center, scale = coords['center'], coords['scale']
    tail = None
    for radius in (4, 8, 16, 32, 64):
        reference.check()
        tail = tail_risk_bound(m, state, action, center-radius*scale, center+radius*scale)
        if tail['value'] <= 1e-6:
            break
    else:
        return dict(status='tail_failed', tail=tail)
    nodes = np.linspace(-np.arcsinh(radius), np.arcsinh(radius), 65)
    checks = (nodes[:-1]+nodes[1:])/2
    max_inner_error = 0.0

    def baseline(child):
        return np.asarray([terminal_risk_interval(m, child, a)['linear_risk']
                           for a in range(m.num_actions)])

    def values(u):
        nonlocal max_inner_error
        reference.check()
        z = np.sinh(u)
        child = m.condition(state, action, center+scale*z)
        results = [reference.terminal(child, a) for a in range(m.num_actions)]
        max_inner_error = max(max_inner_error, max(e/(1+z*z) for _, e in results))
        scores = np.asarray([v for v, _ in results])
        if linear_baseline:
            scores = baseline(child)-scores
        return scores/(1+z*z)

    diagnostics = {}
    if adaptive:
        from .adaptive_value_fit import fit_adaptive
        interpolator, nodes, diagnostics = fit_adaptive(
            values, nodes[0], nodes[-1], nonnegative=not linear_baseline, quintic=quintic)
        if interpolator is None:
            return dict(status='validation_failed', radius=radius, tail_bound=tail['value'],
                        normalized_inner_error=max_inner_error, **diagnostics)
        error = diagnostics['normalized_check_error']
    else:
        training = np.asarray([values(u) for u in nodes])
        actual = np.asarray([values(u) for u in checks])
        interpolator, error = interpolation_check(nodes, training, checks, actual)
        if not np.isfinite(training).all() or not np.isfinite(actual).all():
            raise ValueError('nonfinite normalized action values')
    result = dict(status='validation_failed', radius=radius, tail_bound=tail['value'],
                  normalized_check_error=error, normalized_inner_error=max_inner_error,
                  training_nodes=65, check_nodes=64, uniform_error_proven=False)
    result.update(diagnostics)
    if error > 2e-5 or max_inner_error > 1e-7:
        return result
    density = student_log_density(df, loc, scale2)

    def integrand(u):
        reference.check()
        z = np.sinh(u)
        scores = interpolator(u)*(1+z*z)
        if linear_baseline:
            child = m.condition(state, action, center+scale*z)
            scores = reconstruct_values(baseline(child), interpolator(u), 1+z*z)
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError('invalid interpolated action values')
        value = float(np.min(scores))
        probability = np.exp(np.logaddexp.reduce(logs+density(center+scale*z)))
        return value*probability*scale*np.cosh(u)

    integral = quad(integrand, nodes[0], nodes[-1], points=nodes[1:-1],
                    epsabs=1e-8, epsrel=1e-8, limit=100, full_output=1)
    if len(integral) != 3:
        return dict(result, status='integration_failed', reason=integral[3])
    value, quadrature_error, _ = integral
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid surrogate integral')
    result.update(status='sample_checks_passed', interior_estimate=value,
                  quadrature_error=quadrature_error,
                  tail_only_interval=[value, value+tail['value']])
    return result
