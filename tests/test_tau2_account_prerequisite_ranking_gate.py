from __future__ import annotations

import json

import pytest

from scripts.tau2_account_prerequisite_ranking_gate import (
    HYPOTHESIS_COUNT,
    ROOT_ACTIONS,
    best_two_step_information,
    information_gain,
    official_signature_coverage,
    parse_hypotheses,
    parse_rollout,
)


def _hypotheses() -> list[dict[str, str]]:
    signatures = [
        ("available", "enabled", "off"),
        ("available", "disabled", "on"),
        ("available", "disabled", "off"),
        ("exhausted", "enabled", "off"),
        ("exhausted", "disabled", "on"),
        ("exhausted", "disabled", "off"),
        ("unknown", "enabled", "off"),
        ("available", "unknown", "unknown"),
    ]
    return [
        {
            "id": f"h{index + 1}",
            "description": f"account mechanism {index + 1}",
            "data_allowance": signature[0],
            "account_roaming": signature[1],
            "device_roaming": signature[2],
        }
        for index, signature in enumerate(signatures)
    ]


def test_parse_hypotheses_and_coverage() -> None:
    parsed = parse_hypotheses(json.dumps({"hypotheses": _hypotheses()}))
    assert len(parsed) == HYPOTHESIS_COUNT
    assert official_signature_coverage(parsed) == 6


def test_unknown_hypothesis_does_not_cover_official_worlds() -> None:
    rows = _hypotheses()
    for row in rows:
        for field in (
            "data_allowance",
            "account_roaming",
            "device_roaming",
        ):
            row[field] = "unknown"
    # Keep the signatures syntactically distinct without adding real coverage.
    rows[1]["data_allowance"] = "available"
    rows[2]["account_roaming"] = "enabled"
    rows[3]["device_roaming"] = "off"
    rows[4]["data_allowance"] = "exhausted"
    rows[5]["account_roaming"] = "disabled"
    rows[6]["device_roaming"] = "on"
    rows[7].update(data_allowance="available", account_roaming="unknown")
    assert official_signature_coverage(rows) == 0


def test_abnormal_signature_requires_exact_abnormal_fields() -> None:
    rows = _hypotheses()
    rows[3]["data_allowance"] = "unknown"
    assert official_signature_coverage(rows) == 5


def test_parse_hypotheses_rejects_bad_line_status() -> None:
    rows = _hypotheses()
    rows[0]["data_allowance"] = "depleted"
    with pytest.raises(ValueError, match="invalid data_allowance"):
        parse_hypotheses(json.dumps({"hypotheses": rows}))


def test_exact_setup_information_uses_unlocked_line_details() -> None:
    worlds = [
        "available_enabled_off",
        "available_disabled_on",
        "available_disabled_off",
        "exhausted_enabled_off",
        "exhausted_disabled_on",
        "exhausted_disabled_off",
    ]
    line_results = [
        "available_enabled",
        "available_disabled",
        "available_disabled",
        "exhausted_enabled",
        "exhausted_disabled",
        "exhausted_disabled",
    ]
    observations = {}
    for world, line_result in zip(worlds, line_results, strict=True):
        observations[world] = {
            action: (
                "device roaming on"
                if action == "network_status" and "disabled_on" in world
                else "shared"
            )
            for action in ROOT_ACTIONS
        }
        observations[world].update(
            {
                "customer_lookup": "C1001,L1001,L1002,L1003",
                "line_details": line_result,
                "data_usage": (
                    "exhausted" if world.startswith("exhausted") else "available"
                ),
                "customer_bills": "clear",
            }
        )
    assert information_gain(
        [observations[world]["customer_lookup"] for world in worlds]
    ) == 0.0
    value, branch_actions = best_two_step_information(
        observations, "customer_lookup"
    )
    assert value == pytest.approx(1.329661348855)
    assert set(branch_actions.values()) == {"line_details"}


def test_parse_rollout_scores_customer_lookup_then_line_details() -> None:
    hypotheses = _hypotheses()
    outcomes = [
        "available enabled",
        "available disabled",
        "available disabled",
        "exhausted enabled",
        "exhausted disabled",
        "exhausted disabled",
        "available enabled",
        "available unknown",
    ]
    payload = {
        "root_predictions": [
            {"hypothesis_id": row["id"], "outcome": "customer C1001, lines known"}
            for row in hypotheses
        ],
        "branches": [
            {
                "root_outcome": "customer C1001, lines known",
                "followup_action": "line_details",
                "followup_predictions": [
                    {"hypothesis_id": row["id"], "outcome": outcome}
                    for row, outcome in zip(hypotheses, outcomes, strict=True)
                ],
            }
        ],
    }
    parsed = parse_rollout(
        json.dumps(payload),
        hypotheses=hypotheses,
        root_action="customer_lookup",
    )
    assert parsed["predicted_d1_information"] == 0.0
    assert parsed["predicted_d2_information"] > 1.0


def test_parse_rollout_rejects_line_details_without_lookup() -> None:
    hypotheses = _hypotheses()
    payload = {
        "root_predictions": [
            {"hypothesis_id": row["id"], "outcome": "same"}
            for row in hypotheses
        ],
        "branches": [
            {
                "root_outcome": "same",
                "followup_action": "line_details",
                "followup_predictions": [
                    {"hypothesis_id": row["id"], "outcome": "active"}
                    for row in hypotheses
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid rollout branch"):
        parse_rollout(
            json.dumps(payload),
            hypotheses=hypotheses,
            root_action="status_bar",
        )
