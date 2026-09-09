"""One-observation Bayes prediction risk on explicitly supplied cached hypotheses.

No simulator or endpoint is invoked. Callers must bind arrays to the exact frozen
initial support; matching shapes alone is insufficient provenance.
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss

from scripts.discoverphysics_dark_matter_opportunity import posterior_batch


def myopic_prediction_risk(means, targets, prior, sigma, *, order=32):
    means, targets, prior = map(lambda a: np.asarray(a, float), (means, targets, prior))
    if (means.ndim != 2 or means.shape[1] not in (1, 2) or not len(means)
            or targets.ndim != 2 or len(targets) != len(means) or not targets.shape[1]
            or prior.shape != (len(means),) or np.any(prior < 0)
            or not all(np.isfinite(a).all() for a in (means, targets, prior))
            or not np.isclose(prior.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError('invalid prediction mixture')
    if isinstance(sigma, bool) or not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('positive finite noise required')
    if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 128:
        raise ValueError('order must be an integer from 2 to 128')
    if len(means) > 256 or targets.shape[1] > 512 or len(means)*order**means.shape[1] > 500000:
        raise ValueError('quadrature workspace/work limit')
    nodes, mass = hermgauss(order)
    nodes, mass = nodes*np.sqrt(2)*sigma, mass/np.sqrt(np.pi)
    if means.shape[1] == 1:
        offsets, weights = nodes[:, None], mass
    else:
        x, y = np.meshgrid(nodes, nodes, indexing='ij')
        offsets = np.column_stack((x.ravel(), y.ravel()))
        weights = np.outer(mass, mass).ravel()
    centered = targets-prior@targets
    second = np.mean(centered**2, axis=1)
    risk = 0.
    for hypothesis, probability in enumerate(prior):
        if probability == 0:
            continue
        for start in range(0, len(offsets), 256):
            stop = start+256
            observations = means[hypothesis]+offsets[start:stop]
            posterior = posterior_batch(prior[None, :], observations, means, sigma)[0]
            variance = posterior@second-np.mean((posterior@centered)**2, axis=1)
            if not np.isfinite(variance).all() or np.min(variance) < -1e-10:
                raise ArithmeticError('invalid conditional prediction variance')
            risk += probability*float(weights[start:stop]@np.maximum(variance, 0))
    return risk
