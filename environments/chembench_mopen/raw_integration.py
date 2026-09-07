"""Bounded adaptive scalar integration for Gaussian predictive expectations.

QUADPACK errors are numerical estimates, not certified mathematical bounds.
Only the omitted Gaussian-tail contribution has an analytic upper bound.
"""

from dataclasses import dataclass
import math
from time import monotonic
from typing import Callable, Hashable

import numpy as np
from scipy.integrate import quad

from .horizon import SearchLimitExceeded, _integer
from .raw_belief import GaussianParticleModel


@dataclass(frozen=True)
class IntegralEstimate:
    value: float
    quadrature_error: float
    tail_bound: float
    evaluations: int


def predictive_expectation(
    model: GaussianParticleModel,
    state: Hashable,
    action: int,
    continuation: Callable[[tuple], float],
    *,
    value_bound: float,
    tolerance: float = 1e-6,
    max_evaluations: int = 10_000,
    max_seconds: float = 10.0,
) -> IntegralEstimate:
    """Integrate a bounded nonnegative continuation on raw posterior states.

    Caller must supply a valid global bound and a deterministic continuation.
    Nested callers must separately account for continuation estimation error;
    this function does not certify a recursively approximated policy value.
    """
    action = model._action(action)
    logs = model._logs(state)
    _integer(max_evaluations, "max_evaluations", minimum=1)
    if any(
        not math.isfinite(x) or x <= 0 for x in (value_bound, tolerance, max_seconds)
    ):
        raise ValueError("bound, tolerance and time must be finite and positive")
    started = monotonic()
    evaluations = 0
    # Integrate each generating component separately. This retains tiny prior
    # components and avoids silently missing narrow mixture peaks.
    radius = 8.0
    while value_bound * math.erfc(radius / math.sqrt(2)) > tolerance / 4:
        radius += 1
        if radius > 38:
            raise ValueError("requested tail tolerance is not representable")
    weights = np.exp(logs)
    values, errors = [], []
    for i, weight in enumerate(weights):
        if weight == 0:
            continue
        mean, sigma = model.means[i, action], model.sigmas[i, action]

        def integrand(z):
            nonlocal evaluations
            evaluations += 1
            if evaluations > max_evaluations or monotonic() - started > max_seconds:
                raise SearchLimitExceeded(
                    "predictive integration exceeded resource cap"
                )
            posterior = model.condition(state, action, float(mean + sigma * z))
            value = float(continuation(posterior))
            if not math.isfinite(value) or value < 0 or value > value_bound:
                raise ValueError("continuation violates its global value bound")
            return value * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

        # Seed subdivision at each likelihood peak and width, not outcome bins.
        points = sorted(
            {
                float((m + k * s - mean) / sigma)
                for m, s in zip(model.means[:, action], model.sigmas[:, action])
                for k in (-2, -1, 0, 1, 2)
                if -radius < (m + k * s - mean) / sigma < radius
            }
        )
        if len(points) >= 200:
            raise SearchLimitExceeded("too many integration breakpoints")
        result = quad(
            integrand,
            -radius,
            radius,
            points=points,
            epsabs=tolerance / 2,
            epsrel=0,
            limit=200,
            full_output=1,
        )
        if len(result) != 3:
            raise ArithmeticError("adaptive quadrature did not converge")
        value, error, _ = result
        values.append(float(weight) * value)
        errors.append(float(weight) * error)
    error = math.fsum(errors)
    tail = value_bound * math.erfc(radius / math.sqrt(2))
    if error + tail > tolerance:
        raise ArithmeticError("predictive integration missed tolerance")
    if monotonic() - started > max_seconds:
        raise SearchLimitExceeded("predictive integration exceeded time cap")
    return IntegralEstimate(math.fsum(values), error, tail, evaluations)
