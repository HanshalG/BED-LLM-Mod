from __future__ import annotations

import json

import numpy as np

from scripts.discoverphysics_dark_matter_grounded_policy import (
    build_branches,
    hidden_map_family,
    parse_refresh,
    relative_reduction,
    stratified_bootstrap_difference,
    weighted_two_means,
)
from tests.test_discoverphysics_dark_matter_executable_support import (
    _valid_payload,
)


def _weighted_hypotheses() -> list[dict]:
    hypotheses = _valid_payload()["hypotheses"]
    for hypothesis in hypotheses:
        probability = hypothesis.pop("probability")
        hypothesis["weight"] = int(round(probability * 100))
    return hypotheses


def test_weighted_two_means_separates_two_clouds():
    values = np.array(
        [[-2.0, 0.0], [-1.8, 0.1], [2.0, 0.0], [2.1, -0.1]]
    )
    centers, assignments = weighted_two_means(
        values,
        np.full(4, 0.25),
    )

    assert centers.shape == (2, 2)
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]


def test_build_branches_returns_normalized_objective_posteriors():
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    root_means = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    )
    means = {
        root["action_id"]: root_means for root in __import__(
            "scripts.discoverphysics_dark_matter_grounded_policy",
            fromlist=["ROOTS"],
        ).ROOTS
    }

    branches = build_branches(means, prior)

    assert set(branches) == {"A", "B", "C", "D"}
    for root_branches in branches.values():
        assert np.isclose(
            sum(branch["probability"] for branch in root_branches),
            1.0,
        )
        assert all(
            np.isclose(sum(branch["posterior_probabilities"]), 1.0)
            for branch in root_branches
        )


def test_refresh_parser_reuses_weighted_support_grammar():
    response = json.dumps(
        {
            "hypotheses": _weighted_hypotheses(),
            "continuation_action": "r2.5_a1",
        }
    )

    refresh = parse_refresh(
        response,
        root_action_id="center",
        label="refresh",
    )

    assert len(refresh["hypotheses"]) == 8
    assert refresh["continuation_action"] == "r2.5_a1"


def test_hidden_family_is_fresh_weighted_96_map_endpoint():
    maps, regions, prior = hidden_map_family()

    assert maps.shape == (96, 10, 2)
    assert len(regions) == 96
    assert np.isclose(prior.sum(), 1.0)
    assert {
        region: float(prior[np.asarray(regions) == region].sum())
        for region in ("NE", "NW", "SW", "SE")
    } == {"NE": 0.4, "NW": 0.3, "SW": 0.2, "SE": 0.1}


def test_stratified_bootstrap_and_relative_reduction():
    regions = [
        region
        for region in ("NE", "NW", "SW", "SE")
        for _ in range(6)
    ]
    differences = np.ones(24)

    lower, upper = stratified_bootstrap_difference(
        differences,
        regions,
    )

    assert np.isclose(lower, 1.0)
    assert np.isclose(upper, 1.0)
    assert np.isclose(relative_reduction(2.0, 1.5), 0.25)
