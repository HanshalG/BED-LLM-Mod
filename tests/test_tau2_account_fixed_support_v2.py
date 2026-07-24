from __future__ import annotations

from scripts.tau2_account_fixed_support_v2 import FIXED_HYPOTHESES
from scripts.tau2_account_prerequisite_ranking_gate import (
    ROOT_ACTIONS,
    official_signature_coverage,
    rollout_messages,
)


def test_fixed_support_is_complete_and_unique() -> None:
    assert len(FIXED_HYPOTHESES) == 6
    assert len(
        {
            (
                row["data_allowance"],
                row["account_roaming"],
                row["device_roaming"],
            )
            for row in FIXED_HYPOTHESES
        }
    ) == 6
    assert official_signature_coverage(FIXED_HYPOTHESES) == 6


def test_fixed_support_rollout_hides_scores_and_simulator_outputs() -> None:
    messages = rollout_messages(
        FIXED_HYPOTHESES,
        "customer_lookup",
        "No carrier service.",
    )
    text = messages[-1]["content"]
    assert "line_details" in text
    assert "data_allowance" in text
    assert "information gain" not in text.casefold()
    assert "1.329" not in text
    assert set(ROOT_ACTIONS).issuperset({"customer_lookup", "network_status"})
