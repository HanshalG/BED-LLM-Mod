"""Adaptive conditional parameter moments; not a posterior coverage certificate."""
import numpy as np
from scipy.integrate import cubature

from .parameter_quadrature import IntegrationUnresolved


def adaptive_parameter_integral(lower, upper, likelihood, predict, *, output_size,
                                log_likelihood_bound, max_rows=400000):
    """Two partition checks with an analytic likelihood ceiling for stable scaling.

    Uniform independent prior in the supplied transformed box. The ceiling must
    be analytically supplied, never fitted to hidden outcomes. For Gaussian
    observations the sum of peak log densities is a valid ceiling. Returned
    moments are conditional on the supplied structure, not selection-corrected.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (lower.ndim != 1 or len(lower) not in (1, 2) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not np.all(upper > lower)):
        raise ValueError('one or two finite transformed parameter intervals required')
    for value, maximum in ((output_size, 16), (max_rows, 400000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError('invalid output size or evaluation cap')
    if (isinstance(log_likelihood_bound, bool) or not np.isscalar(log_likelihood_bound)
            or not np.isfinite(log_likelihood_bound)):
        raise ValueError('finite analytic likelihood bound required')
    rows = 0
    checks = []

    def integrand(unit):
        nonlocal rows
        if rows + len(unit) > max_rows:
            raise IntegrationUnresolved('adaptive parameter row cap')
        rows += len(unit)
        nodes = lower + unit*(upper-lower)
        ll = np.asarray(likelihood(nodes), float)
        pred = np.asarray(predict(nodes), float)
        if (ll.shape != (len(unit),) or np.isnan(ll).any() or np.isposinf(ll).any()
                or np.any(ll > log_likelihood_bound + 1e-10)
                or pred.shape != (len(unit), output_size) or not np.isfinite(pred).all()):
            raise IntegrationUnresolved('invalid callback or violated likelihood bound')
        with np.errstate(over='raise', invalid='raise'):
            density = np.exp(ll-log_likelihood_bound)
            return density[:, None]*np.column_stack((np.ones(len(unit)), pred, pred**2))

    try:
        for split in (False, True):
            result = cubature(integrand, np.zeros(len(lower)), np.ones(len(lower)),
                              rtol=1e-7, atol=1e-12, max_subdivisions=1000,
                              points=[np.full(len(lower), .5)] if split else None)
            value, error = np.asarray(result.estimate), np.asarray(result.error)
            check = {'partition': 'midpoint' if split else 'whole',
                     'status': result.status, 'integrals': value.tolist(),
                     'errors': error.tolist(), 'subdivisions': result.subdivisions,
                     'cumulative_rows': rows}
            checks.append(check)
            z = value[0]
            if (result.status != 'converged' or not np.isfinite(value).all()
                    or not np.isfinite(error).all() or z <= 0 or error[0]/z > 1e-5):
                raise IntegrationUnresolved('unresolved adaptive evidence')
            mean = value[1:1+output_size]/z
            variance = value[1+output_size:]/z-mean**2
            mean_error = (error[1:1+output_size] + np.abs(mean)*error[0])/z
            second_error = (error[1+output_size:] + np.abs(value[1+output_size:]/z)*error[0])/z
            variance_error = second_error + 2*np.abs(mean)*mean_error + mean_error**2
            check.update(estimated_mean_error=mean_error.tolist(),
                         estimated_variance_error=variance_error.tolist())
            if (np.any(mean_error > 1e-6 + 1e-3*np.abs(mean))
                    or np.any(variance_error > 1e-6 + 1e-3*np.abs(variance))):
                raise IntegrationUnresolved('unresolved normalized moments')
            if (variance < -1e-10).any():
                raise IntegrationUnresolved('negative variance')
            check.update(log_evidence=float(np.log(z)+log_likelihood_bound),
                         mean=mean.tolist(), variance=np.maximum(variance, 0).tolist())
        if (abs(checks[0]['log_evidence']-checks[1]['log_evidence']) > 1e-4
                or any(not np.allclose(checks[0][key], checks[1][key], atol=1e-6, rtol=1e-3)
                       for key in ('mean', 'variance'))):
            raise IntegrationUnresolved('partition disagreement')
        return {'status': 'agreement', 'checks': checks, 'evaluated_rows': rows,
                'interpretation': 'estimated_error_and_partition_agreement_not_coverage_certificate'}
    except (IntegrationUnresolved, FloatingPointError) as error:
        return {'status': 'unresolved', 'checks': checks, 'evaluated_rows': rows,
                'reason': str(error)}
