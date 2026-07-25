from __future__ import annotations

import numpy as np

from scripts.newtonbench_snell_opportunity_audit import (
    ACTION_BANK_SHA256,
    generate_action_bank,
    mixed_likelihood_tables,
)
from scripts.newtonbench_sound_speed_opportunity_audit import (
    compact_json_sha256,
    expected_entropies_for_priors,
)


def test_snell_action_bank_matches_preregistered_hash() -> None:
    actions = generate_action_bank()

    assert len(actions) == 32
    assert compact_json_sha256(actions) == ACTION_BANK_SHA256
    assert all(
        1.0 <= action["refractive_index_1"] <= 1.5 for action in actions
    )
    assert all(
        1.0 <= action["refractive_index_2"] <= 1.5 for action in actions
    )
    assert all(0.0 <= action["incidence_angle"] <= 90.0 for action in actions)


def test_mixed_table_separates_invalid_from_finite_outcomes() -> None:
    means = np.asarray([[np.nan], [10.0], [12.0]])
    tables, source_indices, weights = mixed_likelihood_tables(
        means, noise_level=0.01, quadrature_order=3
    )

    invalid_rows = source_indices == 0
    finite_rows = source_indices != 0
    assert np.all(tables[0, invalid_rows, 0] == 0.0)
    assert np.all(np.isneginf(tables[0, invalid_rows, 1:]))
    assert np.all(np.isneginf(tables[0, finite_rows, 0]))
    assert np.all(np.isfinite(tables[0, finite_rows, 1:]))
    np.testing.assert_allclose(weights.reshape(3, 3).sum(axis=1), 1.0)


def test_impossible_zero_weight_branches_do_not_produce_nan_entropy() -> None:
    means = np.asarray([[np.nan], [10.0], [12.0]])
    tables, source_indices, weights = mixed_likelihood_tables(
        means, noise_level=0.01, quadrature_order=3
    )
    priors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )

    expected = expected_entropies_for_priors(
        priors, tables[0], source_indices, weights
    )

    assert np.all(np.isfinite(expected))
    assert expected[0] == 0.0
    assert expected[1] >= 0.0
