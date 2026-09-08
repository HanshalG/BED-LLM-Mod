"""Finite-mixture one-step risk bounds without observation quadrature."""
import numpy as np

from environments.chembench_mopen.centered_risk import CenteredTargetRisk


class ParticleRiskIntervals:
    def __init__(self, model):
        self.model = model
        self.risk = CenteredTargetRisk(model)

    def actions(self, state):
        m = self.model
        weights = np.exp(m._logs(state))
        target_mean = weights @ self.risk.centered
        centered_targets = self.risk.centered-target_mean
        shifted = m.means-m.means[0]
        locations = weights @ shifted
        delta = shifted-locations
        variances = weights @ (delta**2+m.sigmas**2)
        covariances = (weights[:, None]*delta).T @ centered_targets
        prior = float(self.risk(weights[None, :])[0])
        noise = float(weights @ m.target_noise_risk)
        linear = prior-np.sum(covariances**2, axis=1)/variances
        if (not np.isfinite(variances).all() or np.any(variances <= 0)
                or not np.isfinite(linear).all() or not np.isfinite(noise)):
            raise ValueError('invalid particle risk moments')
        padding = 1e-12*max(1., abs(prior), abs(noise), float(np.max(np.abs(linear))))
        if np.any(linear < noise-2*padding):
            raise ValueError('inconsistent particle risk bounds')
        # Particle revelation removes all mean uncertainty but not fresh target noise.
        return [dict(lower=max(0., noise-padding), upper=float(value+padding),
                     noise_floor=noise, linear_risk=float(value), float_padding=padding)
                for value in linear]
