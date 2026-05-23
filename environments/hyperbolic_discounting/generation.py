"""LLM generation helpers (re-exported from runner)."""

from .runner import (
    choose_design_naive,
    estimate_params_naive,
    generate_hyperbolic_candidates,
    generate_hyperbolic_hypotheses,
)

__all__ = [
    "choose_design_naive",
    "estimate_params_naive",
    "generate_hyperbolic_candidates",
    "generate_hyperbolic_hypotheses",
]
