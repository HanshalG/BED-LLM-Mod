"""Registry that maps ``(task_name, method_name)`` pairs to BED builders.

The existing :mod:`main` dispatches based on ``config.task == "location_finding"``
with a hand-rolled if/else.  The registry inverts this: each environment
registers itself together with the methods it supports, and ``main`` just
looks up the right builder.

Adding a third environment becomes: write a new ``Environment`` and ``Method``,
register them here, done — no changes to ``main`` or the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .environment import Environment
from .method import Method


# A builder takes the legacy ``Config`` plus the *questioner* LLM and the
# *answerer* LLM (or None for environments that don't have one) and returns a
# fully-constructed Environment.
EnvironmentBuilder = Callable[[Any, Any, Any], Environment]

# A method builder takes the legacy ``Config`` and returns a constructed Method.
MethodBuilder = Callable[[Any], Method]


@dataclass(frozen=True)
class EnvironmentRecord:
    name: str
    build: EnvironmentBuilder


@dataclass(frozen=True)
class MethodRecord:
    name: str
    build: MethodBuilder


_ENVIRONMENTS: dict[str, EnvironmentRecord] = {}
_METHODS: dict[tuple[str, str], MethodRecord] = {}


def register_environment(name: str, build: EnvironmentBuilder) -> None:
    """Register an environment factory under the given task name."""
    if name in _ENVIRONMENTS:
        raise ValueError(f"environment {name!r} is already registered")
    _ENVIRONMENTS[name] = EnvironmentRecord(name=name, build=build)


def register_method(env_name: str, method_name: str, build: MethodBuilder) -> None:
    """Register a method factory for the given environment."""
    key = (env_name, method_name)
    if key in _METHODS:
        raise ValueError(f"method {method_name!r} for env {env_name!r} is already registered")
    _METHODS[key] = MethodRecord(name=method_name, build=build)


def build_environment(env_name: str, config: Any, questioner: Any, answerer: Any) -> Environment:
    """Look up the registered builder for ``env_name`` and call it."""
    record = _ENVIRONMENTS.get(env_name)
    if record is None:
        raise KeyError(
            f"no environment registered for {env_name!r}; "
            f"known environments: {sorted(_ENVIRONMENTS)}"
        )
    return record.build(config, questioner, answerer)


def build_method(env_name: str, method_name: str, config: Any) -> Method:
    """Look up the registered builder for ``(env_name, method_name)`` and call it."""
    record = _METHODS.get((env_name, method_name))
    if record is None:
        raise KeyError(
            f"no method {method_name!r} registered for env {env_name!r}; "
            f"known methods for this env: "
            f"{sorted(name for env, name in _METHODS if env == env_name)}"
        )
    return record.build(config)


def list_environments() -> list[str]:
    return sorted(_ENVIRONMENTS)


def list_methods(env_name: str) -> list[str]:
    return sorted(name for env, name in _METHODS if env == env_name)


def clear_registry() -> None:
    """Reset the registry; for tests only."""
    _ENVIRONMENTS.clear()
    _METHODS.clear()
