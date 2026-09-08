"""One-step Gaussian-particle risk via an exact linear-predictor control variate."""
import math
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.horizon import SearchLimitExceeded


class ParticleLinearCorrection:
    def __init__(self, model, state, *, max_seconds=5., max_states=100000,
                 max_workspace_bytes=64 * 1024 * 1024):
        if not math.isfinite(max_seconds) or max_seconds <= 0 or max_states <= 0:
            raise ValueError('invalid correction budget')
        self.started = monotonic()
        self.model, self.logs = model, model._logs(state)
        p, t, b = model.num_particles, model.targets.shape[1], model.branch_count
        # Same conservative one-level tensor allowance as centered batch search.
        required = (8 * (20*b*p + 8*p*t) + 16*p*t + 8*p
                    + model.target_conditional_variances.nbytes + model.target_noise_risk.nbytes
                    + getattr(model, '_workspace_fixed_bytes', 0))
        if required > max_workspace_bytes:
            raise SearchLimitExceeded('correction workspace budget exceeded')
        self.risk = CenteredTargetRisk(model)
        self.weights = np.exp(self.logs)
        self.mean = self.weights @ self.risk.centered
        self.prior_risk = float(self.risk(self.weights[None, :])[0])
        self.noise_floor = float(self.weights @ model.target_noise_risk)
        self.max_seconds, self.max_states = max_seconds, max_states
        self.states = 0

    def _check(self):
        if self.states > self.max_states or monotonic() - self.started > self.max_seconds:
            raise SearchLimitExceeded('correction exceeded shared budget')

    def action(self, action):
        self._check()
        m = self.model
        action = m._action(action)
        mu, sigma = m.means[:, action], m.sigmas[:, action]
        # Center the scalar observation as well as the target vector.
        shifted = mu - mu[0]
        mean_y = float(self.weights @ shifted)
        delta = shifted - mean_y
        variance = float(self.weights @ (delta**2 + sigma**2))
        covariance = (self.weights * delta) @ (self.risk.centered - self.mean)
        if not math.isfinite(variance) or variance <= 0 or not np.isfinite(covariance).all():
            raise ValueError('invalid predictive moments')
        slope = covariance / variance
        linear_risk = self.prior_risk - float(covariance @ slope)
        self.states += m.branch_count
        self._check()
        ys, posterior, masses = posterior_branches_many(m, self.logs[None, :], action,
                                                       return_weights=True)
        posterior_mean = np.exp(posterior[0]) @ self.risk.centered
        linear_mean = self.mean + ((ys[0] - mu[0]) - mean_y)[:, None] * slope
        advantage = float(masses[0] @ np.sum((posterior_mean - linear_mean)**2, axis=1))
        value = linear_risk - advantage
        self._check()
        if not np.isfinite([value, advantage, linear_risk]).all() or value < self.noise_floor - 1e-10:
            raise ValueError('invalid corrected risk; integration needs refinement')
        return dict(value=value, linear_risk=linear_risk, advantage=advantage)
