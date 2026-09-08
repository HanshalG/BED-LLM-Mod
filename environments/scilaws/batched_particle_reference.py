"""Joint adaptive integration of full action menus, charging every posterior."""
import math
from time import monotonic

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.horizon import SearchLimitExceeded


def integrate_actions(model, state, *, max_seconds=5., max_evaluations=100000,
                      max_workspace_bytes=64*1024*1024):
    started = monotonic()
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_evaluations <= 0:
        raise ValueError('invalid batched reference budget')
    m = model
    p, t = m.targets.shape
    a = m.num_actions
    cache_bytes = 2*1024*1024
    workspace = 16*p*t+128*a*p+64*p+cache_bytes
    if workspace > max_workspace_bytes:
        raise SearchLimitExceeded('batched reference workspace exceeded')
    logs = m._logs(state)
    bound = float(np.ptp(m.targets, axis=0)**2 @ m.target_weights/4+np.max(m.target_noise_risk))
    if not math.isfinite(bound):
        raise ValueError('invalid risk bound')
    if bound == 0:
        return dict(roots=[dict(value=0., error_estimate=0., tail_bound=0., mass_error=0.) for _ in range(a)],
                    evaluations=0, callbacks=0, seconds=monotonic()-started, workspace_bytes=workspace)
    radius = 8.
    while bound*math.erfc(radius/math.sqrt(2)) > 1e-9:
        radius += 1
        if radius > 38:
            raise ValueError('cannot bound tails')
    mu, sd = m.means.T, m.sigmas.T
    left, right = np.min(mu-radius*sd, axis=1), np.max(mu+radius*sd, axis=1)
    width = right-left
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError('invalid action domains')
    constants = logs[None, :]-np.log(sd)-.5*math.log(2*math.pi)
    risk = CenteredTargetRisk(m)
    evaluations, callbacks = 0, 0

    def check():
        if evaluations > max_evaluations or monotonic()-started > max_seconds:
            raise SearchLimitExceeded('batched reference exceeded shared budget')

    def integrand(x):
        nonlocal evaluations, callbacks
        callbacks += 1
        evaluations += a
        check()
        y = left+width*x
        joint = constants-.5*((y[:, None]-mu)/sd)**2
        peak = np.max(joint, axis=1)
        numerator = np.exp(joint-peak[:, None])
        normalizer = np.sum(numerator, axis=1)
        weights = numerator/normalizer[:, None]
        density = np.exp(peak+np.log(normalizer))*width
        values = risk(weights)
        if np.any(values > bound+1e-10*max(1., bound)):
            raise ValueError('risk exceeds bound')
        return np.concatenate((density, density*values/bound))

    values, error, info = quad_vec(integrand, 0., 1.,
        epsabs=min(1e-8, 1e-8/bound), epsrel=0., norm='max', limit=1000,
        cache_size=cache_bytes, full_output=True)
    exact_mass = (ndtr((right[:, None]-mu)/sd)-ndtr((left[:, None]-mu)/sd)) @ np.exp(logs)
    mass_error = np.abs(values[:a]-exact_mass)
    check()
    tail = bound*math.erfc(radius/math.sqrt(2))
    if (not info.success or not np.isfinite(values).all() or not math.isfinite(error)
            or np.max(mass_error) > 1e-8 or error*bound+tail > 1e-7):
        raise ValueError('batched reference failed integration/mass check')
    roots = [dict(value=float(values[a+i]*bound), error_estimate=float(error*bound),
                  tail_bound=tail, mass_error=float(mass_error[i])) for i in range(a)]
    return dict(roots=roots, evaluations=evaluations, callbacks=callbacks,
                seconds=monotonic()-started, workspace_bytes=workspace)
