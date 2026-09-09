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
    scaling = 1.
    evidence_only = True

    def integrand(unit):
        nonlocal rows
        if rows + len(unit) > max_rows:
            raise IntegrationUnresolved('adaptive parameter row cap')
        rows += len(unit)
        nodes = lower + unit*(upper-lower)
        ll = np.asarray(likelihood(nodes), float)
        if (ll.shape != (len(unit),) or np.isnan(ll).any() or np.isposinf(ll).any()
                or np.any(ll > log_likelihood_bound + 1e-10)):
            raise IntegrationUnresolved('invalid callback or violated likelihood bound')
        with np.errstate(over='raise', invalid='raise'):
            density = np.exp(ll-log_likelihood_bound)/scaling
            if evidence_only:
                return density
            pred = np.asarray(predict(nodes), float)
            if pred.shape != (len(unit), output_size) or not np.isfinite(pred).all():
                raise IntegrationUnresolved('invalid prediction callback')
            return density[:, None]*np.column_stack((np.ones(len(unit)), pred, pred**2))

    try:
        # Absolute tolerances can accept an almost-zero integral without locating
        # a narrow likelihood peak. Evidence is positive, so use relative control.
        pilot = cubature(integrand, np.zeros(len(lower)), np.ones(len(lower)),
                         rtol=1e-7, atol=0., max_subdivisions=1000)
        checks.append({'partition': 'evidence_pilot', 'status': pilot.status,
                       'integrals': float(pilot.estimate), 'errors': float(pilot.error),
                       'subdivisions': pilot.subdivisions, 'cumulative_rows': rows})
        if (pilot.status != 'converged' or not np.isfinite(pilot.estimate)
                or pilot.estimate <= 0 or not np.isfinite(pilot.error)
                or pilot.error/pilot.estimate > 1e-5):
            raise IntegrationUnresolved('unresolved evidence pilot')
        scaling = float(pilot.estimate)
        evidence_only = False
        for split in (False, True):
            # Preserve discovered narrow regions; restarting globally can falsely
            # stop on a tiny absolute error before sampling the known peak.
            results = [cubature(integrand, region.a, region.b,
                                rtol=1e-7, atol=1e-9*float(np.prod(region.b-region.a)),
                                max_subdivisions=1000,
                                points=[(region.a+region.b)/2] if split else None)
                       for region in pilot.regions]
            value = np.sum([r.estimate for r in results], axis=0)
            error = np.sum([r.error for r in results], axis=0)
            status = 'converged' if all(r.status == 'converged' for r in results) else 'not_converged'
            check = {'partition': 'refined_pilot_regions' if split else 'pilot_regions',
                     'status': status, 'integrals': value.tolist(),
                     'errors': error.tolist(), 'subdivisions': sum(r.subdivisions for r in results),
                     'pilot_region_count': len(pilot.regions),
                     'cumulative_rows': rows}
            checks.append(check)
            z = value[0]
            if (status != 'converged' or not np.isfinite(value).all()
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
            check.update(log_evidence=float(np.log(z)+np.log(scaling)+log_likelihood_bound),
                         mean=mean.tolist(), variance=np.maximum(variance, 0).tolist())
        if (abs(checks[-2]['log_evidence']-checks[-1]['log_evidence']) > 1e-4
                or abs(checks[-1]['log_evidence']-(np.log(scaling)+log_likelihood_bound)) > 1e-4
                or any(not np.allclose(checks[-2][key], checks[-1][key], atol=1e-6, rtol=1e-3)
                       for key in ('mean', 'variance'))):
            raise IntegrationUnresolved('partition disagreement')
        return {'status': 'agreement', 'checks': checks, 'evaluated_rows': rows,
                'interpretation': 'estimated_error_and_partition_agreement_not_coverage_certificate'}
    except (IntegrationUnresolved, FloatingPointError) as error:
        return {'status': 'unresolved', 'checks': checks, 'evaluated_rows': rows,
                'reason': str(error)}
