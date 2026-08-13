from __future__ import annotations

import math

from scripts import tau2_native_prerequisite_opportunity as audit


def test_canonical_response_removes_identifiers_not_semantic_state():
    left = audit.canonical_response(
        {"customer_id": "C1001", "line_id": "L1002", "phone": "555-123-2002", "roaming_enabled": True}
    )
    right = audit.canonical_response(
        {"customer_id": "C9876", "line_id": "L4433", "phone": "777-888-9999", "roaming_enabled": True}
    )
    assert left == right
    assert "roaming_enabled" in left
    assert "true" in left


def test_native_followup_is_legal_only_after_prerequisite():
    assert audit.MMS_UNLOCKED_ACTION in audit.legal_followups(
        "mms_home", "installed_apps"
    )
    assert audit.MMS_UNLOCKED_ACTION not in audit.legal_followups(
        "mms_home", "mms_probe"
    )
    assert set(audit.MOBILE_UNLOCKED_ACTIONS).issubset(
        audit.legal_followups("mobile_abroad", "customer_lookup")
    )
    assert not set(audit.MOBILE_UNLOCKED_ACTIONS).intersection(
        audit.legal_followups("mobile_abroad", "network_status")
    )


def test_information_gain_matches_balanced_binary_partition():
    assert math.isclose(
        audit.information_gain(["a", "a", "b", "b"]),
        math.log(2),
        abs_tol=1e-12,
    )


def test_exact_depth_two_values_zero_information_setup_action():
    roots = audit.MMS_ROOT_ACTIONS
    observations = []
    for permission in ("sms", "storage", "both", "none"):
        row = {action: "same" for action in roots}
        row[audit.MMS_UNLOCKED_ACTION] = permission
        observations.append(row)
    metrics = audit.episode_metrics("mms_home", "0" * 64, observations)
    assert metrics["greedy_first_action"] == "wifi_calling"
    assert metrics["depth_two_first_action"] == "installed_apps"
    assert metrics["depth_two_changes_first_action"] is True
    assert metrics["depth_two_selects_native_prerequisite"] is True
    assert math.isclose(metrics["horizon_gain_nats"], math.log(4), abs_tol=1e-12)


def test_depth_two_ties_break_lexically():
    observations = []
    for value in ("a", "b", "a", "b"):
        row = {action: value for action in audit.MMS_ROOT_ACTIONS}
        row[audit.MMS_UNLOCKED_ACTION] = value
        observations.append(row)
    metrics = audit.episode_metrics("mms_home", "1" * 64, observations)
    assert metrics["greedy_first_action"] == max(audit.MMS_ROOT_ACTIONS)
    assert metrics["depth_two_first_action"] == max(audit.MMS_ROOT_ACTIONS)
    assert metrics["horizon_gain_nats"] == 0.0


def test_result_schema_cannot_contain_private_source_fields(tmp_path):
    row = {
        "episode_sha256": "2" * 64,
        "family": "mms_home",
        "state_count": 4,
        "prior_entropy_nats": math.log(4),
        "maximum_root_information_nats": 0.0,
        "maximum_root_information_fraction": 0.0,
        "greedy_first_action": "apn_settings",
        "greedy_two_step_information_nats": 0.0,
        "depth_two_first_action": "installed_apps",
        "depth_two_information_nats": math.log(4),
        "depth_two_changes_first_action": True,
        "depth_two_selects_native_prerequisite": True,
        "horizon_gain_nats": math.log(4),
    }
    assert not {"task_id", "faults", "responses", "ticket"}.intersection(row)
