"""Continuous-observation EIG helpers for location finding."""

from .runner import (
    _expected_information_gain_batch_from_means as expected_information_gain_batch_from_means,
    _expected_information_gain_from_means as expected_information_gain_from_means,
    _normal_logpdf_array as normal_logpdf_array,
    _posterior_probabilities_after_values as posterior_probabilities_after_values,
    _quadrature_nodes as quadrature_nodes,
    expected_information_gain,
    score_candidate_locations,
)

__all__ = [
    "expected_information_gain",
    "expected_information_gain_batch_from_means",
    "expected_information_gain_from_means",
    "normal_logpdf_array",
    "posterior_probabilities_after_values",
    "quadrature_nodes",
    "score_candidate_locations",
]

