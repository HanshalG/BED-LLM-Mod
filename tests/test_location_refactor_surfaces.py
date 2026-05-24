"""Regression tests for the location-finding package surfaces."""

from __future__ import annotations


def test_location_finding_package_exports_new_surfaces():
    from environments.location_finding import beliefs, eig, generation, parsing, physics, strategy, types

    assert types.LocationObservation.__name__ == "LocationObservation"
    assert callable(parsing.parse_source_hypotheses)
    assert callable(physics.signal_intensity_for_hypothesis)
    assert callable(beliefs.build_location_posterior)
    assert callable(generation.generate_location_candidates)
    assert callable(eig.score_candidate_locations)
    assert callable(strategy.choose_location_with_strategy_rollouts)
