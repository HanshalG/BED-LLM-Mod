from __future__ import annotations

import json
import math

import pytest

from scripts.tau2_mms_prerequisite_ranking_gate import (
    HYPOTHESIS_COUNT,
    ROOT_ACTIONS,
    best_two_step_information,
    information_gain,
    official_signature_coverage,
    parse_hypotheses,
    parse_rollout,
)


def _hypotheses() -> list[dict[str, str]]:
    rows = []
    signatures = [
        ("faulty", "normal", "normal", "normal", "normal"),
        ("normal", "faulty", "normal", "normal", "normal"),
        ("normal", "normal", "faulty", "normal", "normal"),
        ("normal", "normal", "normal", "faulty", "normal"),
        ("normal", "normal", "normal", "normal", "faulty"),
        ("normal", "normal", "normal", "faulty", "faulty"),
        ("unknown", "normal", "normal", "normal", "normal"),
        ("normal", "unknown", "normal", "normal", "normal"),
    ]
    for index, signature in enumerate(signatures):
        rows.append(
            {
                "id": f"h{index + 1}",
                "description": f"mechanism {index + 1}",
                "network_mode": signature[0],
                "wifi_calling": signature[1],
                "mmsc_apn": signature[2],
                "sms_permission": signature[3],
                "storage_permission": signature[4],
            }
        )
    return rows


def test_information_gain_matches_six_world_partition() -> None:
    partition = ["normal", "normal", "normal", "sms", "storage", "both"]
    assert information_gain(partition) == pytest.approx(1.242453324894)


def test_prerequisite_action_beats_two_direct_binary_checks() -> None:
    worlds = ["network", "wifi", "apn", "sms", "storage", "both"]
    observations = {}
    for world in worlds:
        observations[world] = {
            action: (
                f"{action}:fault"
                if (
                    (action == "network_mode" and world == "network")
                    or (action == "wifi_calling" and world == "wifi")
                    or (action == "apn_settings" and world == "apn")
                )
                else f"{action}:normal"
            )
            for action in ROOT_ACTIONS
        }
        observations[world]["installed_apps"] = "messaging,browser"
        observations[world]["messaging_permissions"] = (
            world if world in {"sms", "storage", "both"} else "all"
        )

    setup_value, branch_actions = best_two_step_information(
        observations,
        "installed_apps",
    )
    direct_value, _ = best_two_step_information(observations, "network_mode")

    assert setup_value == pytest.approx(1.242453324894)
    assert direct_value == pytest.approx(0.867563228481)
    assert setup_value - direct_value == pytest.approx(0.374890096413)
    assert set(branch_actions.values()) == {"messaging_permissions"}


def test_parse_hypotheses_and_signature_coverage() -> None:
    text = json.dumps({"hypotheses": _hypotheses()})
    parsed = parse_hypotheses(text)
    assert len(parsed) == HYPOTHESIS_COUNT
    assert official_signature_coverage(parsed) == 6


def test_signature_coverage_treats_unknown_normal_fields_as_wildcards() -> None:
    rows = _hypotheses()
    for row in rows:
        for field in (
            "network_mode",
            "wifi_calling",
            "mmsc_apn",
            "sms_permission",
            "storage_permission",
        ):
            if row[field] == "normal":
                row[field] = "unknown"
    assert official_signature_coverage(rows) == 6


def test_parse_hypotheses_rejects_invalid_status() -> None:
    rows = _hypotheses()
    rows[0]["network_mode"] = "broken"
    with pytest.raises(ValueError, match="invalid network_mode"):
        parse_hypotheses(json.dumps({"hypotheses": rows}))


def test_parse_rollout_scores_branch_specific_followup() -> None:
    hypotheses = _hypotheses()
    root_predictions = [
        {"hypothesis_id": row["id"], "outcome": "messaging,browser"}
        for row in hypotheses
    ]
    followup_outcomes = [
        "all",
        "all",
        "all",
        "sms missing",
        "storage missing",
        "both missing",
        "all",
        "all",
    ]
    payload = {
        "root_predictions": root_predictions,
        "branches": [
            {
                "root_outcome": "messaging,browser",
                "followup_action": "messaging_permissions",
                "followup_predictions": [
                    {"hypothesis_id": row["id"], "outcome": outcome}
                    for row, outcome in zip(
                        hypotheses,
                        followup_outcomes,
                        strict=True,
                    )
                ],
            }
        ],
    }
    parsed = parse_rollout(
        json.dumps(payload),
        hypotheses=hypotheses,
        root_action="installed_apps",
    )
    assert parsed["predicted_d1_information"] == 0.0
    assert parsed["predicted_d2_information"] == pytest.approx(
        information_gain(followup_outcomes)
    )
    assert math.isfinite(parsed["predicted_d2_information"])


def test_parse_rollout_rejects_permission_without_setup() -> None:
    hypotheses = _hypotheses()
    payload = {
        "root_predictions": [
            {"hypothesis_id": row["id"], "outcome": "normal"}
            for row in hypotheses
        ],
        "branches": [
            {
                "root_outcome": "normal",
                "followup_action": "messaging_permissions",
                "followup_predictions": [
                    {"hypothesis_id": row["id"], "outcome": "all"}
                    for row in hypotheses
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid rollout branch"):
        parse_rollout(
            json.dumps(payload),
            hypotheses=hypotheses,
            root_action="network_status",
        )
