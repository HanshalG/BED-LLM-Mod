from __future__ import annotations

from scripts.tau2_account_only_v4 import (
    ACCOUNT_HYPOTHESES,
    ACCOUNT_WORLD_IDS,
    add_account_only_gates,
)
from scripts.tau2_account_prerequisite_ranking_gate import ROOT_ACTIONS


def _payload(d2_lookup: float = 1.0) -> dict:
    actions = [
        {
            "action_id": action,
            "predicted_d1_information": 0.0,
            "predicted_d2_information": (
                d2_lookup if action == "customer_lookup" else 0.0
            ),
        }
        for action in ROOT_ACTIONS
    ]
    return {
        "schema_version": 2,
        "status": "passed",
        "protocol": {"stage": "serving_smoke"},
        "summary": {
            "gates": {
                "base": True,
                "mean_signature_coverage_at_least_4": True,
                "all_pass": True,
            }
        },
        "records": [
            {
                "official_signature_coverage": 4,
                "actions": actions,
            },
            {
                "official_signature_coverage": 4,
                "actions": actions,
            },
        ],
    }


def test_account_only_support_is_four_official_worlds() -> None:
    assert len(ACCOUNT_HYPOTHESES) == 4
    assert len(ACCOUNT_WORLD_IDS) == 4
    assert {row["device_roaming"] for row in ACCOUNT_HYPOTHESES} == {"off"}


def test_account_only_gates_select_zero_eig_setup_at_depth_two() -> None:
    payload = add_account_only_gates(_payload())
    assert payload["status"] == "passed"
    assert payload["summary"]["d1_lookup_selected_count"] == 0
    assert payload["summary"]["d2_lookup_selected_count"] == 2


def test_account_only_gates_reject_missing_setup_value() -> None:
    payload = add_account_only_gates(_payload(d2_lookup=0.0))
    assert payload["status"] == "gate_failed"
