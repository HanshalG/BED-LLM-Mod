"""Memory-bounded batch implementation of the scalar quantile Bellman search."""

from dataclasses import dataclass
from itertools import permutations
import math
from time import monotonic

import numpy as np
from scipy.special import ndtr, ndtri

from .horizon import SearchLimitExceeded, _integer
from .quantile_belief import QuantileGaussianModel


@dataclass(frozen=True)
class BatchPlan:
    action: int | None
    value: float
    root_values: tuple[tuple[int, float], ...]
    requested_horizon: int
    effective_horizon: int
    mode: str
    fixed_sequence: tuple[int, ...] | None
    processed_states: int
    elapsed_seconds: float


def posterior_branches_many(model, logs, action, *, return_weights=False):
    """Same 64 bisections and full scalar likelihood as QuantileGaussianModel."""
    action = model._action(action)
    logs = np.asarray(logs, dtype=float)
    if logs.ndim != 2 or logs.shape[1] != model.num_particles or not len(logs):
        raise ValueError("expected nonempty (beliefs, particles)")
    if (
        np.isnan(logs).any()
        or np.isposinf(logs).any()
        or not np.allclose(np.logaddexp.reduce(logs, axis=1), 0, atol=1e-12, rtol=0)
    ):
        raise ValueError("expected normalized log weights")
    weights = np.exp(logs)
    means, sigmas = model.means[:, action], model.sigmas[:, action]
    if hasattr(model, "quadrature_rule"):
        rules = [model.quadrature_rule(row, action) for row in logs]
        quantiles = np.stack([rule[0] for rule in rules])
        integration_weights = np.stack([rule[1] for rule in rules])
    else:
        quantiles = np.broadcast_to(model._quantiles, (len(logs), model.branch_count))
        integration_weights = np.broadcast_to(
            model._quadrature_weights, quantiles.shape
        )
    if (
        not np.isfinite(quantiles).all()
        or np.any(quantiles < 0)
        or np.any(quantiles > 1)
    ):
        raise ValueError("invalid integration probabilities")
    # Composite rules can round an interior tail node to an endpoint. Keep its
    # mass but evaluate at the closest representable interior probability.
    quantiles = np.clip(quantiles, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    component_q = means[None, :, None] + sigmas[None, :, None] * ndtri(
        quantiles[:, None, :]
    )
    active = weights[:, :, None] > 0
    low = np.min(np.where(active, component_q, np.inf), axis=1)
    high = np.max(np.where(active, component_q, -np.inf), axis=1)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("quantiles exceed numerical range")

    def cdf(y):
        return np.sum(
            weights[:, :, None]
            * ndtr((y[:, None, :] - means[None, :, None]) / sigmas[None, :, None]),
            axis=1,
        )

    for _ in range(64):
        middle = low / 2 + high / 2
        below = cdf(middle) < quantiles
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    observations = low / 2 + high / 2
    if np.max(np.abs(cdf(observations) - quantiles)) > 1e-8:
        raise ArithmeticError("predictive quantile inversion failed")
    with np.errstate(over="ignore", invalid="ignore"):
        likelihoods = (
            -0.5 * ((observations[:, :, None] - means) / sigmas) ** 2
            - np.log(sigmas)
            - 0.5 * math.log(2 * math.pi)
        )
    if not np.isfinite(likelihoods).all():
        raise ValueError("likelihood exceeds numerical range")
    posterior = logs[:, None, :] + likelihoods
    posterior -= np.logaddexp.reduce(posterior, axis=2)[:, :, None]
    return (
        (observations, posterior, integration_weights)
        if return_weights
        else (observations, posterior)
    )


def plan_batched(
    model: QuantileGaussianModel,
    state,
    horizon,
    *,
    available=None,
    mode="adaptive",
    batch_size=64,
    max_states=5_000_000,
    max_seconds=60,
    max_workspace_bytes=64 * 1024 * 1024,
):
    horizon = _integer(horizon, "horizon")
    for name, value in [
        ("batch_size", batch_size),
        ("max_states", max_states),
        ("max_workspace_bytes", max_workspace_bytes),
    ]:
        _integer(value, name, minimum=1)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if mode not in ("adaptive", "open_loop"):
        raise ValueError("invalid mode")
    state = model._logs(state)[None, :]
    menu = (
        tuple(range(model.num_actions))
        if available is None
        else tuple(sorted(model._action(a) for a in available))
    )
    if len(set(menu)) != len(menu):
        raise ValueError("duplicate actions")
    depth = min(horizon, len(menu))
    if depth > 3:
        raise SearchLimitExceeded("batch reference supports at most depth three")
    # Conservative tensor allowance across all live recursion levels. Includes
    # targets in terminal risk work, not only posterior tensors. No disk cache.
    row_bytes = (
        8
        * max(depth, 1)
        * (
            20 * model.branch_count * model.num_particles
            + 8 * model.num_particles * model.targets.shape[1]
        )
    )
    batch_size = min(batch_size, max_workspace_bytes // row_bytes)
    if batch_size < 1:
        raise SearchLimitExceeded("workspace budget too small for one belief")
    started = monotonic()
    processed = 0

    def check(count=0):
        nonlocal processed
        processed += count
        if processed > max_states or monotonic() - started > max_seconds:
            raise SearchLimitExceeded("batch horizon exceeded global resource budget")

    def terminal(logs):
        weights = np.exp(logs)
        forecasts = weights @ model.targets
        deviations = model.targets[None, :, :] - forecasts[:, None, :]
        return np.sum(weights * ((deviations**2) @ model.target_weights), axis=1)

    def integrate(logs, action, continuation):
        check()
        _, posterior, integration_weights = posterior_branches_many(
            model, logs, action, return_weights=True
        )
        check()
        values = continuation(posterior.reshape(-1, model.num_particles))
        return np.sum(
            values.reshape(len(logs), model.branch_count) * integration_weights, axis=1
        )

    def value(logs, actions, remaining, sequence=None):
        output = np.empty(len(logs))
        for start in range(0, len(logs), batch_size):
            chunk = logs[start : start + batch_size]
            check(len(chunk))
            if not remaining:
                result = terminal(chunk)
            elif sequence is not None:
                result = integrate(
                    chunk,
                    sequence[0],
                    lambda child: value(child, (), remaining - 1, sequence[1:]),
                )
            else:
                result = np.full(len(chunk), np.inf)
                for action in actions:
                    rest = tuple(a for a in actions if a != action)
                    candidate = integrate(
                        chunk, action, lambda child: value(child, rest, remaining - 1)
                    )
                    result = np.minimum(result, candidate)
            output[start : start + len(chunk)] = result
        return output

    roots = {}
    fixed = None
    if not depth:
        chosen, value_at_root = None, float(terminal(state)[0])
    elif mode == "adaptive":
        for action in menu:
            rest = tuple(a for a in menu if a != action)
            roots[action] = float(
                integrate(state, action, lambda child: value(child, rest, depth - 1))[0]
            )
        value_at_root, chosen = min((v, a) for a, v in roots.items())
    else:
        best = None
        for sequence in permutations(menu, depth):
            candidate = float(value(state, menu, depth, sequence)[0])
            roots[sequence[0]] = min(roots.get(sequence[0], math.inf), candidate)
            if best is None or (candidate, sequence) < best:
                best = candidate, sequence
        value_at_root, fixed = best
        chosen = fixed[0]
    check()
    return BatchPlan(
        chosen,
        value_at_root,
        tuple(sorted(roots.items())),
        horizon,
        depth,
        mode,
        fixed,
        processed,
        monotonic() - started,
    )
