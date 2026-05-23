"""Continuous-observation EIG helpers for hyperbolic temporal discounting."""

from .runner import (
    _expected_information_gain_from_means as expected_information_gain_from_means,
    _quadrature_nodes as quadrature_nodes,
    expected_information_gain,
    score_candidate_designs,
)

__all__ = [
    "expected_information_gain",
    "expected_information_gain_from_means",
    "quadrature_nodes",
    "score_candidate_designs",
]
