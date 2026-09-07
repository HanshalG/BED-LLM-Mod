"""Bounded full-budget evaluation of the deployed one-step Bayes policy."""

from dataclasses import dataclass
import math
from time import monotonic

import numpy as np

from .batch_horizon import posterior_branches_many
from .horizon import SearchLimitExceeded, _integer


@dataclass(frozen=True)
class MyopicPolicyValue:
    value: float
    measurement_budget: int
    root_action: int | None
    root_one_step_values: tuple
    processed_states: int
    elapsed_seconds: float


def evaluate_myopic_policy(
    model,
    state,
    budget,
    *,
    available=None,
    batch_size=64,
    max_states=8_000_000,
    max_seconds=60,
    max_workspace_bytes=64 * 1024**2,
):
    """Select by one-step risk, but score only after the common real budget.

    This is policy evaluation, not a replacement for full contingent planning.
    Each decision uses the entire remaining design menu and current posterior.
    """
    budget = _integer(budget, "budget")
    for name, value in [
        ("batch_size", batch_size),
        ("max_states", max_states),
        ("max_workspace_bytes", max_workspace_bytes),
    ]:
        _integer(value, name, minimum=1)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("invalid time cap")
    menu = (
        tuple(range(model.num_actions))
        if available is None
        else tuple(sorted(model._action(a) for a in available))
    )
    if len(menu) != len(set(menu)) or budget > len(menu):
        raise ValueError("duplicate menu or insufficient distinct designs")
    if budget > 3:
        raise SearchLimitExceeded(
            "policy reference supports at most three measurements"
        )
    state = model._logs(state)[None, :]
    fixed = 8 * model.num_particles**2 + getattr(model, "_workspace_fixed_bytes", 0)
    row_bytes = 8 * max(1, budget) * 20 * model.num_particles * model.branch_count
    batch_size = min(batch_size, (max_workspace_bytes - fixed) // row_bytes)
    if batch_size < 1:
        raise SearchLimitExceeded("insufficient policy workspace")
    started, processed = monotonic(), 0
    root_action, root_values = None, ()

    def check(count=0):
        nonlocal processed
        processed += count
        if processed > max_states or monotonic() - started > max_seconds:
            raise SearchLimitExceeded(
                "policy evaluation exceeded global resource budget"
            )

    distances = np.empty((model.num_particles, model.num_particles))
    for index in range(model.num_particles):
        check()
        distances[index] = (
            model.targets - model.targets[index]
        ) ** 2 @ model.target_weights
    if not np.isfinite(distances).all():
        raise ValueError("unrepresentable target distances")

    def risk(logs):
        check(len(logs))
        weights = np.exp(logs)
        return 0.5 * np.sum(weights * (weights @ distances), axis=1)

    def branches(logs, action):
        check(len(logs) * model.branch_count)
        _, posterior, masses = posterior_branches_many(
            model, logs, action, return_weights=True
        )
        check()
        return posterior, masses

    def walk(logs, actions, remaining, is_root=False):
        nonlocal root_action, root_values
        if not remaining:
            return risk(logs)
        values = np.empty(len(logs))
        for start in range(0, len(logs), batch_size):
            chunk = logs[start : start + batch_size]
            check(len(chunk))
            immediate = np.empty((len(chunk), len(actions)))
            for index, action in enumerate(actions):
                posterior, masses = branches(chunk, action)
                immediate[:, index] = np.sum(
                    risk(posterior.reshape(-1, model.num_particles)).reshape(
                        len(chunk), model.branch_count
                    )
                    * masses,
                    axis=1,
                )
            chosen = np.argmin(immediate, axis=1)
            if is_root:
                root_action = actions[int(chosen[0])]
                root_values = tuple(zip(actions, map(float, immediate[0])))
            if remaining == 1:
                values[start : start + len(chunk)] = immediate[
                    np.arange(len(chunk)), chosen
                ]
                continue
            for index, action in enumerate(actions):
                selected = np.flatnonzero(chosen == index)
                if not len(selected):
                    continue
                posterior, masses = branches(chunk[selected], action)
                rest = tuple(a for a in actions if a != action)
                future = walk(
                    posterior.reshape(-1, model.num_particles), rest, remaining - 1
                )
                values[start + selected] = np.sum(
                    future.reshape(len(selected), model.branch_count) * masses, axis=1
                )
        return values

    value = float(walk(state, menu, budget, is_root=True)[0])
    check()
    return MyopicPolicyValue(
        value, budget, root_action, root_values, processed, monotonic() - started
    )
