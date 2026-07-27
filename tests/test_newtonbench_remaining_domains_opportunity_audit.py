from __future__ import annotations

import math

import numpy as np

from scripts.newtonbench_remaining_domains_opportunity_audit import (
    DOMAIN_SPECS,
    generate_action_bank,
    generate_action_manifest,
    mixed_quadrature_likelihood_tables,
    select_development_candidate,
)
from scripts.newtonbench_sound_speed_opportunity_audit import (
    quadrature_likelihood_tables,
)


def test_action_manifest_is_deterministic_and_bounded() -> None:
    first = generate_action_manifest()
    second = generate_action_manifest()
    assert first == second
    assert len(first["domains"]) == 10

    for domain, spec in DOMAIN_SPECS.items():
        record = first["domains"][domain]
        for bank_name in ("development_actions", "confirmation_actions"):
            actions = record[bank_name]
            assert len(actions) == 32
            for action in actions:
                for name, lower, upper, _scale in spec["parameters"]:
                    assert lower <= action[name] <= upper


def test_action_bank_uses_all_latin_hypercube_bins() -> None:
    parameters = (("x", 1.0, 33.0, "linear"),)
    actions = generate_action_bank(parameters, seed=7)
    bins = {
        min(31, int((action["x"] - 1.0) / (32.0 / 32.0)))
        for action in actions
    }
    assert bins == set(range(32))


def test_mixed_builder_matches_finite_gaussian_builder() -> None:
    means = np.asarray(
        [
            [1.0, 2.0],
            [1.5, 3.0],
            [-0.5, 4.0],
        ]
    )
    expected = quadrature_likelihood_tables(
        means, noise_level=0.1, quadrature_order=5
    )
    actual = mixed_quadrature_likelihood_tables(
        means, noise_level=0.1, quadrature_order=5
    )
    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_array, expected_array)


def test_mixed_builder_preserves_exact_invalid_category() -> None:
    means = np.asarray([[math.nan], [math.nan], [2.0]])
    tables, sources, weights = mixed_quadrature_likelihood_tables(
        means, noise_level=0.1, quadrature_order=3
    )
    assert sources.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert math.isclose(float(weights.sum()), 3.0)
    assert np.all(tables[0, :6, :2] == 0.0)
    assert np.all(np.isneginf(tables[0, :6, 2]))
    assert np.all(np.isneginf(tables[0, 6:, :2]))
    assert np.all(np.isfinite(tables[0, 6:, 2]))


def test_candidate_selection_uses_frozen_ordering() -> None:
    domains = [
        {
            "domain": "m3_fourier_law",
            "module_index": 3,
            "development_strata": [
                {
                    "noise_level": 0.1,
                    "strict_opportunity": True,
                    "diagnostics": {
                        "primary_depth_two_margin_nats": 0.03,
                        "primary_immediate_margin_nats": 0.04,
                    },
                }
            ],
        },
        {
            "domain": "m0_gravity",
            "module_index": 0,
            "development_strata": [
                {
                    "noise_level": 0.01,
                    "strict_opportunity": True,
                    "diagnostics": {
                        "primary_depth_two_margin_nats": 0.03,
                        "primary_immediate_margin_nats": 0.04,
                    },
                }
            ],
        },
    ]
    assert select_development_candidate(domains) == ("m0_gravity", 0.01)
