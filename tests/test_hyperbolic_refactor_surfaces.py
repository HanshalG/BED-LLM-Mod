"""Facade import tests for hyperbolic_discounting."""

import hyperbolic_discounting
from environments.hyperbolic_discounting import (
    HyperbolicBEDEnvironment,
    HyperbolicParams,
    build_hyperbolic_posterior,
    generate_hyperbolic_hypotheses,
    score_candidate_designs,
)


def test_facade_exports_runner_symbols():
    assert hyperbolic_discounting.HyperbolicParams is HyperbolicParams
    assert callable(build_hyperbolic_posterior)
    assert callable(generate_hyperbolic_hypotheses)
    assert callable(score_candidate_designs)


def test_package_exports_adapter():
    assert HyperbolicBEDEnvironment.__name__ == "HyperbolicBEDEnvironment"
