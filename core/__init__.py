"""Core abstractions for Bayesian Experimental Design (BED) with LLMs.

This package contains environment-agnostic infrastructure:

- :class:`BeliefState` — generic weighted-hypothesis container
- :class:`Environment` — abstract interface every BED environment must implement
- :class:`Method` — abstract action-selection strategy (e.g. EIG, Naive, StrategyEIG)
- :class:`BEDRunner` — the shared trial/round loop that wires them together

Concrete environments (animals, location_finding, ...) implement the
:class:`Environment` ABC and are dispatched through :class:`BEDRunner`.
"""

from .belief import BeliefState
from .environment import Environment
from .method import Method, ActionScore
from .bed_runner import BEDRunner, RoundResult, TrialResult, RunResult
from .config import (
    AnimalsConfig,
    BaseConfig,
    HyperbolicConfig,
    LocationConfig,
    animals_view,
    base_view,
    hyperbolic_view,
    location_view,
)
from .experiment_summary import ExperimentSummary
from .registry import (
    build_environment,
    build_method,
    list_environments,
    list_methods,
    register_environment,
    register_method,
)

__all__ = [
    "BeliefState",
    "Environment",
    "Method",
    "ActionScore",
    "BEDRunner",
    "RoundResult",
    "TrialResult",
    "RunResult",
    "AnimalsConfig",
    "BaseConfig",
    "LocationConfig",
    "HyperbolicConfig",
    "animals_view",
    "base_view",
    "hyperbolic_view",
    "location_view",
    "build_environment",
    "build_method",
    "list_environments",
    "list_methods",
    "register_environment",
    "register_method",
    "ExperimentSummary",
]
