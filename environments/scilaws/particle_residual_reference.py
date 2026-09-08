"""Diagnostic full-density integration of the linear-predictor residual."""
import math
from time import monotonic

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.horizon import SearchLimitExceeded


class ParticleResidualReference:
    def __init__(self, model, state, *, max_seconds=5., max_evaluations=100000):
        if not math.isfinite(max_seconds) or max_seconds <= 0 or max_evaluations <= 0:
            raise ValueError('invalid diagnostic budget')
        self.start = monotonic()
        self.model, self.logs = model, model._logs(state)
        self.risk = CenteredTargetRisk(model)
        self.max_seconds, self.max_evaluations = max_seconds, max_evaluations
        self.evaluations = 0

    def check(self):
        if self.evaluations > self.max_evaluations or monotonic()-self.start > self.max_seconds:
            raise SearchLimitExceeded('residual reference exceeded shared budget')

    def action(self, action):
        self.check()
        m = self.model
        action = m._action(action)
        w = np.exp(self.logs)
        targets = self.risk.centered
        tm = w @ targets
        mu, sd = m.means[:, action], m.sigmas[:, action]
        ym = float(w @ mu)
        delta = mu-ym
        variance = float(w @ (delta**2+sd**2))
        cov = (w*delta) @ (targets-tm)
        slope = cov/variance
        linear_risk = float(self.risk(w[None, :])[0] - cov @ slope)
        maximum_target = float(np.max(np.sum((targets-tm)**2, axis=1)))
        slope_squared = float(slope @ slope)
        radius = 8.
        while True:
            probability = 2*float(ndtr(-radius))
            second = 2*(radius*math.exp(-radius**2/2)/math.sqrt(2*math.pi)+ndtr(-radius))
            # Outside the union domain implies |standardized Y|>radius for each component.
            tail = (2*maximum_target*probability
                    + 4*slope_squared*float(w @ (delta**2*probability+sd**2*second)))
            if tail <= 1e-9:
                break
            radius += 1
            if radius > 38:
                raise ValueError('cannot bound residual tails')
        left, right = float(np.min(mu-radius*sd)), float(np.max(mu+radius*sd))
        bound = max(1., 2*maximum_target+2*slope_squared*max((left-ym)**2, (right-ym)**2))
        if not np.isfinite([left, right, bound, tail, linear_risk]).all() or left >= right:
            raise ValueError('invalid residual domain')
        self.evaluations += m.branch_count
        self.check()
        ys, _ = posterior_branches_many(m, self.logs[None, :], action)
        lower, upper = float(ys.min()), float(ys.max())
        constants = self.logs-np.log(sd)-.5*math.log(2*math.pi)

        def integrand(y):
            self.evaluations += 1
            self.check()
            joint = constants-.5*((y-mu)/sd)**2
            total = np.logaddexp.reduce(joint)
            mean = np.exp(joint-total) @ targets
            residual = mean-tm-slope*(y-ym)
            value = float(residual @ residual)
            density = np.exp(total)
            return density*np.array([1., value/bound, value/bound if y < lower or y > upper else 0.])

        points = sorted(set([*np.linspace(left, right, 17)[1:-1], lower, upper]))
        values, error, info = quad_vec(integrand, left, right, points=points,
            epsabs=min(1e-8, 1e-8/bound), epsrel=0., norm='max', limit=1000, full_output=True)
        mass = float(w @ (ndtr((right-mu)/sd)-ndtr((left-mu)/sd)))
        mass_error = abs(values[0]-mass)
        self.check()
        if (not info.success or not np.isfinite(values).all() or not math.isfinite(error)
                or error*bound+tail > 1e-7 or mass_error > 1e-8):
            raise ValueError('residual reference failed accuracy check')
        return dict(value=linear_risk-float(values[1]*bound), advantage=float(values[1]*bound),
                    outer_advantage=float(values[2]*bound), error_estimate=float(error*bound),
                    tail_bound=tail, mass_error=float(mass_error), lower=lower, upper=upper)
