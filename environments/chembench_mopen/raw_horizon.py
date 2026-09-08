"""Ordinary-horizon continuous reference with bounded nested integration.

Error accounting is conditional on the adaptive integrator's error estimates.
It is not a rigorous interval proof or a model-calibration guarantee.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math
from time import monotonic

import numpy as np

from .horizon import SearchLimitExceeded, _integer
from .raw_belief import GaussianParticleModel
from .raw_integration import predictive_expectation


@dataclass(frozen=True)
class RawPlan:
    mode: str
    requested_horizon: int
    effective_horizon: int
    action: int | None
    value: float
    root_values: tuple[tuple[int, float], ...]
    error_estimate: float
    selection_regret_estimate: float
    selection_resolved: bool
    fixed_sequence: tuple[int, ...] | None
    evaluations: int
    elapsed_seconds: float


def plan_raw_horizon(
    model: GaussianParticleModel,
    state: tuple,
    horizon: int,
    *,
    available: tuple[int, ...] | None = None,
    mode: str = "adaptive",
    tolerance: float = 1e-5,
    max_evaluations: int = 200_000,
    max_seconds: float = 30,
    cache_size: int = 1024,
) -> RawPlan:
    """No repeats; full menu; same fixed squared loss for every continuation.

    Each level gets tolerance/horizon for integration. A min is nonexpansive
    in uniform action-value error, so at most one local error per remaining
    level propagates to the root. Root choice regret is at most twice that
    estimated error. Inner estimates are clipped to the analytic risk range
    before integration; projection cannot increase error to the true value.
    All numerical calls share a single global evaluation/time budget.
    """
    horizon = _integer(horizon, "horizon")
    _integer(max_evaluations, "max_evaluations", minimum=1)
    _integer(cache_size, "cache_size", minimum=1)
    if mode not in ("adaptive", "open_loop"):
        raise ValueError("invalid mode")
    if any(not math.isfinite(v) or v <= 0 for v in (tolerance, max_seconds)):
        raise ValueError("tolerance and max_seconds must be finite and positive")
    menu = tuple(range(model.num_actions)) if available is None else tuple(available)
    menu = tuple(sorted(model._action(a) for a in menu))
    if len(set(menu)) != len(menu):
        raise ValueError("duplicate available actions")
    model._logs(state)
    state = tuple(state)
    depth = min(horizon, len(menu))
    if depth > 3:
        raise SearchLimitExceeded("raw reference supports at most three steps")
    bound = float(np.ptp(model.targets, axis=0) ** 2 @ model.target_weights / 4
                  + np.max(model.target_noise_risk))
    if not math.isfinite(bound):
        raise ValueError("unrepresentable target risk range")
    started = monotonic()
    evaluations = 0
    local_tolerance = tolerance / max(depth, 1)

    def check():
        if evaluations >= max_evaluations or monotonic() - started >= max_seconds:
            raise SearchLimitExceeded("raw horizon exceeded global resource budget")

    def expect(belief, action, continuation):
        check()
        if bound == 0:
            return 0.0
        # Truly uninformative observations do not change this fixed model.
        if np.all(model.means[:, action] == model.means[0, action]) and np.all(
            model.sigmas[:, action] == model.sigmas[0, action]
        ):
            return continuation(belief)

        def counted(posterior):
            nonlocal evaluations
            check()
            evaluations += 1
            return float(np.clip(continuation(posterior), 0, bound))

        result = predictive_expectation(
            model,
            belief,
            action,
            counted,
            value_bound=bound,
            tolerance=local_tolerance,
            max_evaluations=max_evaluations,
            max_seconds=max(max_seconds - (monotonic() - started), 1e-12),
        )
        return result.value

    @lru_cache(maxsize=cache_size)
    def choose(belief, actions, remaining):
        check()
        if remaining == 0:
            return model.risk(belief)
        return min(action_value(belief, actions, remaining, a) for a in actions)

    def action_value(belief, actions, remaining, action):
        rest = tuple(a for a in actions if a != action)
        return expect(
            belief, action, lambda posterior: choose(posterior, rest, remaining - 1)
        )

    @lru_cache(maxsize=cache_size)
    def sequence_value(belief, sequence):
        check()
        if not sequence:
            return model.risk(belief)
        return expect(
            belief,
            sequence[0],
            lambda posterior: sequence_value(posterior, sequence[1:]),
        )

    try:
        fixed = None
        values = {}
        if not depth:
            value, action = model.risk(state), None
        elif mode == "adaptive":
            values = {a: action_value(state, menu, depth, a) for a in menu}
            value, action = min((v, a) for a, v in values.items())
        else:
            best = None
            for sequence in permutations(menu, depth):
                candidate = sequence_value(state, sequence)
                values[sequence[0]] = min(values.get(sequence[0], math.inf), candidate)
                if best is None or (candidate, sequence) < best:
                    best = candidate, sequence
            value, fixed = best
            action = fixed[0]
        check()
        error = tolerance if depth and bound else 0.0
        alternatives = [v for a, v in values.items() if a != action]
        gap = min(alternatives) - value if alternatives else math.inf
        return RawPlan(
            mode,
            horizon,
            depth,
            action,
            value,
            tuple(sorted(values.items())),
            error,
            max(0.0, 2 * error - gap) if alternatives else 0.0,
            gap > 2 * error,
            fixed,
            evaluations,
            monotonic() - started,
        )
    finally:
        choose.cache_clear()
        sequence_value.cache_clear()
