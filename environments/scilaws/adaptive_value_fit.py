"""Adaptive smooth per-action interpolation with disjoint final checks."""

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline


def fit_adaptive(values, left, right, *, nonnegative=True, quintic=False):
    nodes = np.linspace(left, right, 17)
    cache = {}

    def evaluate(u):
        key = float(u)
        if key not in cache:
            result = np.asarray(values(key), dtype=float)
            if (result.ndim != 1 or not np.isfinite(result).all()
                    or (nonnegative and np.any(result < 0))):
                raise ValueError('invalid action-value sample')
            cache[key] = result
        return cache[key]

    refinement_tolerance = 1e-5 if quintic else 2e-5
    diagnostics = dict(uniform_error_proven=False, final_disjoint=False,
                       scheme='quintic' if quintic else 'cubic',
                       refinement_tolerance=refinement_tolerance)
    for iteration in range(8):
        training = np.asarray([evaluate(u) for u in nodes])
        if quintic:
            model = make_interp_spline(nodes, training, k=5, axis=0)
            model.extrapolate = False
        else:
            model = CubicSpline(nodes, training, axis=0, extrapolate=False)
        mids = (nodes[:-1]+nodes[1:])/2
        actual = np.asarray([evaluate(u) for u in mids])
        errors = np.max(np.abs(model(mids)-actual), axis=1)
        bad = errors > refinement_tolerance
        diagnostics.update(iterations=iteration+1, training_nodes=len(nodes),
                           adaptation_evaluations=len(cache),
                           adaptation_error=float(max(errors)))
        if not np.any(bad):
            break
        if len(nodes)+int(np.sum(bad)) > 65:
            return None, nodes, dict(diagnostics, fit_status='node_cap')
        nodes = np.sort(np.concatenate((nodes, mids[bad])))
    else:
        return None, nodes, dict(diagnostics, fit_status='refinement_cap')
    fraction = (np.sqrt(5)-1)/2
    checks = np.sort(np.concatenate([
        nodes[:-1]+f*np.diff(nodes) for f in (fraction, 1-fraction)]))
    if any(float(u) in cache for u in checks):
        raise ValueError('final check overlaps fitting or adaptation')
    actual = np.asarray([evaluate(u) for u in checks])
    prediction = model(checks)
    error = float(np.max(np.abs(prediction-actual)))
    diagnostics.update(final_disjoint=True, check_nodes=len(checks),
                       normalized_check_error=error, total_evaluations=len(cache),
                       fit_status='passed' if error <= 2e-5 and (not nonnegative or np.all(prediction >= 0))
                       else 'fresh_check_failed')
    return (model if diagnostics['fit_status']=='passed' else None), nodes, diagnostics
