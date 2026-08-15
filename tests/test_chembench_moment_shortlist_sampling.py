from __future__ import annotations

import numpy as np

from scripts.chembench_moment_shortlist_sampling import (
    SHORTLIST_SIZE,
    evaluate_shortlist_case,
    moment_proxy_gains,
    select_shortlist,
)


def test_moment_proxy_prioritizes_target_correlated_action() -> None:
    latent = np.asarray([-2.0, -1.0, 1.0, 2.0])
    action_means = np.column_stack((latent, np.asarray([1.0, -1.0, -1.0, 1.0])))
    sigmas = np.full_like(action_means, 0.1)
    targets = latent[:, None]
    gains = moment_proxy_gains(
        action_means, sigmas, targets, np.full(4, 0.25)
    )
    assert gains[0] > gains[1]


def test_shortlist_breaks_proxy_ties_by_assay_index() -> None:
    positions = select_shortlist(
        np.ones(10), (11, 3, 9, 1, 8, 7, 6, 5, 4, 2), SHORTLIST_SIZE
    )
    assert positions == (3, 9, 1, 8, 7, 6, 5, 4)


def test_shortlist_case_measures_regret_against_all_actions() -> None:
    risks = np.zeros((4, SHORTLIST_SIZE, 256))
    risks[:, 0, :] = -1.0
    result = evaluate_shortlist_case(
        full_reference=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        replicate_outcome_risks=risks,
        shortlist_positions=(1, 2, 3, 4, 5, 6, 7, 8),
        full_action_indices=tuple(range(9)),
        root_risk=10.0,
        component_action_risks={
            "bank_1": np.arange(9, dtype=float),
            "bank_2": np.arange(9, dtype=float),
        },
        component_root_risks={"bank_1": 10.0, "bank_2": 10.0},
    )
    payload = result["estimates"]["256"][0]
    assert payload["selected_action_index"] == 1
    assert payload["normalized_top_one_regret"] == 0.1
