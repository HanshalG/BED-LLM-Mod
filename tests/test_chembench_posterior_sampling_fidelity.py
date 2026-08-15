from __future__ import annotations

import numpy as np

from scripts.chembench_posterior_sampling_fidelity import (
    NUM_REPLICATES,
    REFERENCE_OUTCOMES,
    SAMPLE_COUNTS,
    evaluate_case_samples,
    evaluate_gates,
)


def test_case_samples_use_disjoint_nested_prefixes() -> None:
    action_indices = (1, 2, 3)
    outcome_risks = np.vstack(
        (
            np.arange(REFERENCE_OUTCOMES, dtype=float),
            10_000.0 + np.arange(REFERENCE_OUTCOMES, dtype=float),
            20_000.0 + np.arange(REFERENCE_OUTCOMES, dtype=float),
        )
    )
    result = evaluate_case_samples(
        reference_action_risks=np.asarray([0.0, 1.0, 2.0]),
        outcome_risks=outcome_risks,
        action_indices=action_indices,
        root_risk=10.0,
        component_action_risks={
            "bank_1": np.asarray([0.0, 1.0, 2.0]),
            "bank_2": np.asarray([0.0, 2.0, 1.0]),
        },
        component_root_risks={"bank_1": 10.0, "bank_2": 10.0},
    )
    assert set(result["estimates"]) == {str(value) for value in SAMPLE_COUNTS}
    assert all(len(items) == NUM_REPLICATES for items in result["estimates"].values())
    assert result["estimates"]["32"][0]["estimated_action_risks"][0] == 15.5
    assert result["estimates"]["32"][1]["estimated_action_risks"][0] == 271.5
    assert result["ensemble_1024"]["estimated_action_risks"][0] == 511.5


def _payload(selected: int = 1, *, rho: float = 0.95, regret: float = 0.005):
    return {
        "spearman": rho,
        "selected_action_index": selected,
        "normalized_top_one_regret": regret,
        "component_bank_regret": {"bank_1": regret, "bank_2": regret},
    }


def _case(index: int) -> dict:
    estimates = {
        str(count): [_payload() for _ in range(NUM_REPLICATES)]
        for count in SAMPLE_COUNTS
    }
    return {
        "finite_and_reproducible": True,
        "estimates": estimates,
        "ensemble_1024": _payload(),
    }


def test_sampling_gate_passes_all_four_256_replicates() -> None:
    gates = evaluate_gates([_case(index) for index in range(36)])
    assert gates["pass"]
    assert gates["replicate_selected_action_agreement"]["256"]["mean"] == 1.0


def test_sampling_gate_fails_one_bad_256_replicate() -> None:
    cases = [_case(index) for index in range(36)]
    for case in cases[:5]:
        case["estimates"]["256"][2] = _payload(rho=0.1, regret=0.1)
    gates = evaluate_gates(cases)
    assert not gates["pass"]
    assert not gates["conditions"]["all_four_256_replicates_pass"]
