"""Bounded parameter integration conditional on an independent uniform box prior.

Bounds and callback coordinates are transformed coordinates (e.g. log parameters).
Resolution agreement is diagnostic, not a rigorous coverage/error certificate.
This does not correct for selecting a structure using its fitting observations.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre


class IntegrationUnresolved(RuntimeError):
    """No usable posterior was returned within the requested contract."""


@dataclass(frozen=True)
class ParameterIntegral:
    nodes: np.ndarray
    weights: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    log_evidence: float
    orders: tuple[int, ...]
    evaluated_rows: int
    interpretation: str = 'resolution_agreement_not_coverage_certificate'


def integrate_parameters(lower, upper, log_likelihood, predict, *, output_size,
                         orders=(32, 64, 128, 256, 512), max_rows=400000,
                         log_evidence_atol=1e-4, moment_atol=1e-6,
                         moment_rtol=1e-3):
    """Require two consecutive refinements agreeing in evidence and moments.

    Explicitly supports only 1-2 independent uniform transformed parameters and
    1-16 predictive coordinates. Callbacks must be pure vectorized numeric code;
    no observations, source labels or network are accessed by this function.
    All likelihood rows count, including refinements discarded after comparison.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.ndim != 1 or len(lower) not in (1, 2) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not np.all(upper > lower)):
        raise ValueError('one or two finite increasing transformed bounds required')
    for value, name, limit in ((output_size, 'output_size', 16),
                               (max_rows, 'max_rows', 400000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f'invalid {name}')
    orders = tuple(orders)
    if (len(orders) < 3 or any(isinstance(n, bool) or not isinstance(n, int)
                             or not 2 <= n <= 512 for n in orders)
            or any(b <= a for a, b in zip(orders, orders[1:]))):
        raise ValueError('at least three increasing orders, each <=512, required')
    for value in (log_evidence_atol, moment_atol, moment_rtol):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError('tolerances must be finite positive scalars')
    previous, agreements, rows = None, 0, 0
    for index, order in enumerate(orders):
        count = order ** len(lower)
        if rows + count > max_rows:
            raise IntegrationUnresolved('parameter evaluation cap reached')
        abscissa, mass = roots_legendre(order)
        if len(lower) == 1:
            unit, prior_mass = abscissa[:, None], mass / 2
        else:
            a, b = np.meshgrid(abscissa, abscissa, indexing='ij')
            unit = np.column_stack((a.ravel(), b.ravel()))
            prior_mass = np.outer(mass / 2, mass / 2).ravel()
        nodes = lower + (unit + 1) * ((upper - lower) / 2)
        rows += count
        likelihood = np.asarray(log_likelihood(nodes), dtype=float)
        if (likelihood.shape != (count,) or np.isnan(likelihood).any()
                or np.isposinf(likelihood).any()):
            raise IntegrationUnresolved('invalid likelihood values or shape')
        logs = np.log(prior_mass) + likelihood
        evidence = float(np.logaddexp.reduce(logs))
        if not np.isfinite(evidence):
            raise IntegrationUnresolved('zero or nonfinite evidence')
        weights = np.exp(logs - evidence)
        forecasts = np.asarray(predict(nodes), dtype=float)
        if forecasts.shape != (count, output_size) or not np.isfinite(forecasts).all():
            raise IntegrationUnresolved('invalid predictions or shape')
        with np.errstate(over='raise', invalid='raise'):
            mean = weights @ forecasts
            variance = weights @ ((forecasts - mean) ** 2)
        current = (evidence, mean, variance)
        if previous is not None:
            stable = abs(evidence - previous[0]) <= log_evidence_atol
            stable = stable and all(np.allclose(a, b, atol=moment_atol, rtol=moment_rtol)
                                    for a, b in zip(current[1:], previous[1:]))
            agreements = agreements + 1 if stable else 0
        if agreements == 2:
            for array in (nodes, weights, mean, variance):
                array.setflags(write=False)
            return ParameterIntegral(nodes, weights, mean, variance, evidence,
                                     orders[:index + 1], rows)
        previous = current
    raise IntegrationUnresolved('evidence and moments did not stabilize twice')
