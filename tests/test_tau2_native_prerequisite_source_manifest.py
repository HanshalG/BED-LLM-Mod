from __future__ import annotations

import json

import pytest

from scripts import tau2_native_prerequisite_source_manifest as source


def task_id(issue: str, faults: list[str], index: int) -> str:
    return f"[{issue}]{'|'.join(faults)}[PERSONA:P{index}]"


def test_task_key_ignores_persona_but_preserves_physical_faults():
    left = source.task_key("[mms_issue]a|b[PERSONA:Easy]")
    right = source.task_key("[mms_issue]b|a[PERSONA:Hard]")
    assert left == right == ("mms_issue", frozenset({"a", "b"}))


def test_episode_key_defines_native_target_subsystems():
    assert source.episode_key(
        "mms_issue",
        frozenset({"airplane_mode_on", "break_app_sms_permission"}),
    ) == (
        "mms_home",
        frozenset({"airplane_mode_on"}),
        frozenset({"break_app_sms_permission"}),
    )
    assert source.episode_key(
        "mobile_data_issue",
        frozenset({"bad_vpn", "data_usage_exceeded", "user_abroad_roaming_disabled_off"}),
    ) == (
        "mobile_abroad",
        frozenset({"bad_vpn"}),
        frozenset({"data_usage_exceeded", "user_abroad_roaming_disabled_off"}),
    )
    assert source.episode_key("service_issue", frozenset({"airplane_mode_on"})) is None


def test_prior_world_inventory_is_exact_and_unique():
    worlds = source.prior_worlds()
    assert len(worlds) == source.EXPECTED_PRIOR
    assert sum(issue == "mms_issue" for issue, _ in worlds) == 84
    assert sum(issue == "mobile_data_issue" for issue, _ in worlds) == 6


def test_frozen_public_manifest_contains_hashes_not_task_or_fault_ids():
    manifest = json.loads(
        (source.OUTPUT_DIR / "SOURCE_MANIFEST.json").read_text()
    )
    text = source.canonical_json(manifest)
    assert "[mms_issue]" not in text
    assert "break_app_sms_permission" not in text
    assert "user_abroad_roaming_disabled_off" not in text
    assert manifest["task_ids_serialized"] is False
    assert manifest["fault_names_serialized"] is False
    assert manifest["public_source_and_ticket_templates_inspected"] is True
    assert manifest["selected_task_identities_serialized"] is False
    assert manifest["selected_initialization_actions_serialized"] is False
    assert manifest["selected_tool_responses_opened"] is False


def test_duplicate_physical_fault_sets_fail_closed():
    tasks = [
        "[mms_issue]a|b[PERSONA:Easy]",
        "[mms_issue]b|a[PERSONA:Hard]",
    ]
    with pytest.raises(ValueError, match="physical fault sets"):
        source.select_episodes(tasks, [])


def test_protocol_declares_no_model_or_endpoint_authority():
    text = source.PROTOCOL.read_text()
    assert "model response" in text
    assert "authorizes only a separately frozen mechanics interface" in text
    assert "confirmation" in text
