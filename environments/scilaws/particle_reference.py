"""Whole-density one-step particle reference with estimated quadrature error."""
import math
from time import monotonic

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.horizon import SearchLimitExceeded


class ParticleReference:
    def __init__(self, model, state, *, max_seconds=5., max_evaluations=100000):
        if not math.isfinite(max_seconds) or max_seconds <= 0 or max_evaluations <= 0:
            raise ValueError('invalid reference budget')
        self.start = monotonic()
        self.model = model
        self.logs = model._logs(state)
        self.risk = CenteredTargetRisk(model)
        self.bound = float(np.ptp(model.targets, axis=0)**2 @ model.target_weights/4
                           + np.max(model.target_noise_risk))
        if not math.isfinite(self.bound):
            raise ValueError('unrepresentable risk bound')
        self.max_seconds, self.max_evaluations = max_seconds, max_evaluations
        self.evaluations = 0

    def action(self, action):
        m = self.model
        action = m._action(action)
        if self.bound == 0:
            return dict(value=0., error_estimate=0., tail_bound=0., mass_error=0.)
        mu, sd = m.means[:, action], m.sigmas[:, action]
        radius = 8.
        while self.bound*math.erfc(radius/math.sqrt(2)) > 1e-9:
            radius += 1
            if radius > 38:
                raise ValueError('unrepresentable tail allowance')
        left, right = float(np.min(mu-radius*sd)), float(np.max(mu+radius*sd))
        if not math.isfinite(left) or not math.isfinite(right) or left >= right:
            raise ValueError('invalid integration domain')
        constants = self.logs-np.log(sd)-.5*math.log(2*math.pi)

        def integrand(y):
            self.evaluations += 1
            if self.evaluations > self.max_evaluations or monotonic()-self.start > self.max_seconds:
                raise SearchLimitExceeded('particle reference exceeded shared budget')
            joint = constants-.5*((y-mu)/sd)**2
            total = np.logaddexp.reduce(joint)
            posterior = np.exp(joint-total)
            risk = float(self.risk(posterior[None, :])[0])
            if risk > self.bound + 1e-10*max(1., self.bound):
                raise ValueError('risk exceeds analytic bound')
            density = np.exp(total)
            return np.array([density, density*risk/self.bound])

        values, error, info = quad_vec(integrand, left, right,
            epsabs=min(1e-8, 1e-8/self.bound), epsrel=0., norm='max',
            points=np.linspace(left, right, 17)[1:-1], limit=1000, full_output=True)
        exact_mass = float(np.exp(self.logs) @ (ndtr((right-mu)/sd)-ndtr((left-mu)/sd)))
        mass_error = abs(values[0]-exact_mass)
        if (not info.success or not np.isfinite(values).all() or not math.isfinite(error)
                or mass_error > 1e-8 or error*self.bound > 1e-7):
            raise ValueError('particle reference failed quadrature or mass check')
        if monotonic()-self.start > self.max_seconds:
            raise SearchLimitExceeded('particle reference exceeded shared time')
        tail = self.bound*math.erfc(radius/math.sqrt(2))
        return dict(value=float(values[1]*self.bound), error_estimate=float(error*self.bound),
                    tail_bound=tail, mass_error=float(mass_error))
