"""Environment-agnostic EIG method dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from core import ActionScore, BeliefState, Environment, Method
from methods.continuous_eig import ContinuousEIG
from methods.eig_binary import EIGBinary


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


def _has_observation_labels(environment: Any) -> bool:
    labels = getattr(environment, "observation_labels", None)
    return isinstance(labels, (tuple, list)) and len(labels) == 2


def _has_predictive_means(environment: Any) -> bool:
    return callable(getattr(environment, "predictive_means", None))


def build_eig_method(config: Any, environment: Environment | None = None) -> Method:
    """Construct the appropriate EIG module for ``config`` / ``environment``."""
    if environment is not None and callable(getattr(environment, "build_eig_method", None)):
        return environment.build_eig_method(config)  # type: ignore[attr-defined]
    if environment is not None and _has_predictive_means(environment):
        noise_sd = float(getattr(config, "location_noise_sd", 0.5))
        quadrature_order = int(getattr(config, "location_eig_quadrature_order", 15))
        search_depth = int(getattr(config, "location_search_depth", 1))
        return ContinuousEIG(
            noise_sd=noise_sd,
            quadrature_order=quadrature_order,
            search_depth=search_depth,
        )
    if environment is not None and _has_observation_labels(environment):
        labels = tuple(getattr(environment, "observation_labels"))
        search_depth = int(getattr(config, "search_depth", 1))
        return EIGBinary(observation_labels=(labels[0], labels[1]), search_depth=search_depth)
    task = getattr(config, "task", None)
    if task == "animals":
        return EIGBinary(
            observation_labels=("Yes", "No"),
            search_depth=int(getattr(config, "search_depth", 1)),
        )
    if task == "location_finding":
        noise_sd = float(getattr(config, "location_noise_sd", 0.5))
        quadrature_order = int(getattr(config, "location_eig_quadrature_order", 15))
        search_depth = int(getattr(config, "location_search_depth", 1))
        return ContinuousEIG(
            noise_sd=noise_sd,
            quadrature_order=quadrature_order,
            search_depth=search_depth,
        )
    return EIGBinary(observation_labels=("Yes", "No"))


@dataclass
class EIG(Method[H, A, O, S]):
    """Dispatch to binary or continuous EIG based on environment capabilities."""

    _delegate: Method[H, A, O, S]

    @property
    def name(self) -> str:
        return "EIG"

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        return self._delegate.select_action(
            candidates,
            belief_state,
            environment,
            model,
            history,
            config,
        )
