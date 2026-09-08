"""Joint NIG posterior samples; finite approximation, never a source oracle."""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincinv, ndtri
from scipy.stats import qmc

from environments.chembench_mopen.horizon import _integer
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


@dataclass(frozen=True)
class PosteriorParticles:
    model: QuantileGaussianModel
    family_indices: np.ndarray
    noise_variances: np.ndarray
    coefficients: tuple


def sample_posterior(model, state, *, particles_per_family, rng, branch_count=16,
                     sampling='iid'):
    n = _integer(particles_per_family, 'particles_per_family', minimum=1)
    if n > 4096 or not isinstance(rng, np.random.Generator):
        raise ValueError('bounded count and explicit numpy Generator required')
    if sampling not in ('iid', 'sobol') or (sampling == 'sobol' and n & (n-1)):
        raise ValueError('valid sampling mode and power-of-two Sobol count required')
    logs = model._state(state)
    masses = np.exp(logs)
    if np.any(np.isfinite(logs) & (masses == 0)):
        raise ValueError('nonzero family mass underflow')
    active = np.flatnonzero(masses > 0)
    dimensions = sum(len(state.components[i].mean) for i in active)
    # Include coefficient Python objects and simultaneous constructor array copies.
    estimate = n*(64*dimensions + 8*len(active)*(4*model.num_actions+6*len(model.target_weights)+64))
    if estimate > 64*1024**2:
        raise ValueError('posterior sample workspace exceeds cap')
    means, targets, variances, weights, families, coefficients = [], [], [], [], [], []
    for i in active:
        b = state.components[i]
        if sampling == 'sobol':
            engine = qmc.Sobol(len(b.mean)+1, scramble=True, bits=52,
                               seed=int(rng.integers(2**32)))
            u = engine.random_base2(n.bit_length()-1)
            if np.any((u <= 0) | (u >= 1)):
                raise ValueError('Sobol endpoint cannot represent finite posterior sample')
            variance = b.scale/gammaincinv(b.shape, u[:, 0])
            z = ndtri(u[:, 1:])
        else:
            variance = b.scale / rng.gamma(b.shape, 1., n)
            z = rng.standard_normal((n, len(b.mean)))
        precision_cholesky = np.linalg.cholesky(np.asarray(b.precision))
        beta = np.asarray(b.mean) + np.sqrt(variance)[:, None] * np.linalg.solve(precision_cholesky.T, z.T).T
        means.append(beta @ model.action_features[i].T)
        targets.append(beta @ model.target_features[i].T)
        variances.append(variance)
        weights.append(np.full(n, masses[i]/n))
        families.append(np.full(n, i, dtype=int))
        coefficients.extend(tuple(float(v) for v in row) for row in beta)
    variance = np.concatenate(variances)
    family = np.concatenate(families)
    particle_model = QuantileGaussianModel(
        np.concatenate(means), np.sqrt(variance)[:, None], np.concatenate(targets),
        np.concatenate(weights), target_weights=model.target_weights,
        target_conditional_variances=variance[:, None] if model.include_observation_noise else 0.,
        branch_count=branch_count)
    variance.setflags(write=False)
    family.setflags(write=False)
    return PosteriorParticles(particle_model, family, variance, tuple(coefficients))
