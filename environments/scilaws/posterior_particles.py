"""Joint NIG posterior samples; finite approximation, never a source oracle."""
from dataclasses import dataclass

import numpy as np

from environments.chembench_mopen.horizon import _integer
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


@dataclass(frozen=True)
class PosteriorParticles:
    model: QuantileGaussianModel
    family_indices: np.ndarray
    noise_variances: np.ndarray
    coefficients: tuple


def sample_posterior(model, state, *, particles_per_family, rng, branch_count=16):
    n = _integer(particles_per_family, 'particles_per_family', minimum=1)
    if n > 4096 or not isinstance(rng, np.random.Generator):
        raise ValueError('bounded count and explicit numpy Generator required')
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
        variance = b.scale / rng.gamma(b.shape, 1., n)
        precision_cholesky = np.linalg.cholesky(np.asarray(b.precision))
        z = rng.standard_normal((n, len(b.mean)))
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
