"""Register the built-in environments and methods with :mod:`core.registry`."""

from __future__ import annotations

from typing import Any

from .registry import (
    clear_registry,
    register_environment,
    register_method,
)


_REGISTERED = False

_GLOBAL_METHODS = (
    "EIG",
    "naive",
    "Naive",
    "naive+belief",
    "StrategyEIG",
    "StrategyEIG+root",
)

_ANIMALS_ONLY_METHODS = (
    "Entropy",
    "split",
)


def register_defaults(force: bool = False) -> None:
    """Idempotently install the built-in environment/method registrations."""
    global _REGISTERED
    if _REGISTERED and not force:
        return
    if force:
        clear_registry()

    register_environment("animals", _build_animals_environment)
    register_environment("location_finding", _build_location_environment)
    register_environment("hyperbolic_discounting", _build_hyperbolic_environment)

    for env_name in ("animals", "location_finding", "hyperbolic_discounting"):
        for method_name in _GLOBAL_METHODS:
            register_method(env_name, method_name, _build_global_method(method_name))

    for method_name in _ANIMALS_ONLY_METHODS:
        register_method("animals", method_name, _build_animals_only_method(method_name))

    _REGISTERED = True


def _build_animals_environment(config: Any, questioner: Any, answerer: Any) -> Any:
    from environments.animals import AnimalsBEDEnvironment

    return AnimalsBEDEnvironment(config=config, answerer=answerer)


def _build_location_environment(config: Any, questioner: Any, answerer: Any) -> Any:
    from environments.location_finding import LocationBEDEnvironment

    return LocationBEDEnvironment(config=config)


def _build_hyperbolic_environment(config: Any, questioner: Any, answerer: Any) -> Any:
    from environments.hyperbolic_discounting import HyperbolicBEDEnvironment

    return HyperbolicBEDEnvironment(config=config)


def _build_global_method(method_name: str):
    def build(config: Any) -> Any:
        if method_name == "EIG":
            from methods.eig import build_eig_method

            return build_eig_method(config)
        if method_name in {"naive", "Naive", "naive+belief"}:
            from methods import Naive

            canonical = "naive" if method_name == "Naive" else method_name
            return Naive(method_name=canonical)
        if method_name in {"StrategyEIG", "StrategyEIG+root"}:
            from methods import StrategyEIG

            return StrategyEIG(fixed_root=method_name == "StrategyEIG+root")
        raise KeyError(method_name)

    return build


def _build_animals_only_method(method_name: str):
    def build(config: Any) -> Any:
        from methods.animals_special import AnimalsEntropy, AnimalsSplit

        if method_name == "Entropy":
            return AnimalsEntropy()
        if method_name == "split":
            return AnimalsSplit()
        raise KeyError(method_name)

    return build
