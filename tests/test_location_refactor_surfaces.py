"""Regression tests for the location_finding module split."""

from __future__ import annotations

import subprocess
import sys


def test_top_level_location_finding_reexports_new_package_surfaces():
    import location_finding as legacy
    from environments.location_finding import beliefs, eig, generation, parsing, physics, runner, strategy, types

    assert legacy.LocationObservation is types.LocationObservation
    assert legacy.LocationBeliefState is types.LocationBeliefState
    assert legacy.parse_source_hypotheses is parsing.parse_source_hypotheses
    assert legacy.signal_intensity_for_hypothesis is physics.signal_intensity_for_hypothesis
    assert legacy.build_location_posterior is beliefs.build_location_posterior
    assert legacy.generate_location_candidates is generation.generate_location_candidates
    assert legacy.score_candidate_locations is eig.score_candidate_locations
    assert legacy.choose_location_with_strategy_rollouts is strategy.choose_location_with_strategy_rollouts
    assert legacy.run_location_finding is runner.run_location_finding


def test_main_via_runner_import_is_heavy_dependency_safe():
    completed = subprocess.run(
        [sys.executable, "-c", "import main_via_runner; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"

