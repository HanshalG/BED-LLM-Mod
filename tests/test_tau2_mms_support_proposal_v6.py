import json
import math
from pathlib import Path

import pytest

from scripts.tau2_mms_prerequisite_ranking_gate import (
    _load_tau2_tasks,
    _task_index,
    exact_action_values,
    exact_observations_for_backbone,
)
from scripts.tau2_mms_support_proposal_v6 import (
    FORMAL_PROMPTS,
    HYPOTHESIS_COUNT,
    PERMISSION_WORLDS,
    SMOKE_PROMPTS,
    common_observed_history,
    compatible_permission_worlds,
    parse_support,
    proposed_world_support,
    support_messages,
)


def _row(
    index,
    *,
    network_mode="normal",
    wifi_calling="normal",
    mmsc_apn="normal",
    sms_permission="unknown",
    storage_permission="unknown",
):
    return {
        "id": f"h{index}",
        "description": f"distinct mechanism {index}",
        "network_mode": network_mode,
        "wifi_calling": wifi_calling,
        "mmsc_apn": mmsc_apn,
        "sms_permission": sms_permission,
        "storage_permission": storage_permission,
    }


def _valid_payload():
    return {
        "hypotheses": [
            _row(1, sms_permission="faulty", storage_permission="normal"),
            _row(2, sms_permission="normal", storage_permission="faulty"),
            _row(3, sms_permission="faulty", storage_permission="faulty"),
            _row(4, network_mode="faulty", sms_permission="faulty"),
            _row(5, wifi_calling="faulty", storage_permission="faulty"),
            _row(6, mmsc_apn="faulty", sms_permission="faulty"),
        ]
    }


def test_support_parser_and_compatibility_filter_recover_three_worlds():
    hypotheses = parse_support(json.dumps(_valid_payload()))
    support, annotated = proposed_world_support(hypotheses)
    assert support == list(PERMISSION_WORLDS)
    assert sum(bool(row["compatible_worlds"]) for row in annotated) == 3


def test_permission_world_projection_requires_concrete_fault():
    assert compatible_permission_worlds(
        _row(1, sms_permission="unknown", storage_permission="unknown")
    ) == ()
    assert compatible_permission_worlds(
        _row(1, sms_permission="normal", storage_permission="normal")
    ) == ()
    assert compatible_permission_worlds(
        _row(
            1,
            network_mode="faulty",
            sms_permission="faulty",
            storage_permission="normal",
        )
    ) == ()


def test_support_parser_rejects_duplicate_signatures():
    payload = _valid_payload()
    payload["hypotheses"][5] = {
        **payload["hypotheses"][4],
        "id": "h6",
        "description": "another description",
    }
    with pytest.raises(ValueError, match="signatures"):
        parse_support(json.dumps(payload))


def test_support_prompt_exposes_history_but_no_future_score():
    history = {"status_bar": "airplane mode off"}
    messages = support_messages(SMOKE_PROMPTS[0], history)
    text = messages[-1]["content"].lower()
    assert "airplane mode off" in text
    assert "do not identify a true hypothesis" in text
    assert "calculate a score" in text
    assert "messaging_permissions" not in text


def test_exact_permission_worlds_have_zero_d1_and_ln3_setup():
    t3 = Path("external/T3")
    observations = exact_observations_for_backbone(
        t3,
        _task_index(_load_tau2_tasks(t3)),
        (),
    )
    observations = {world: observations[world] for world in PERMISSION_WORLDS}
    history = common_observed_history(observations)
    assert len(history) == 7
    values = exact_action_values(observations)
    assert all(
        values[action]["exact_d1_information"] == pytest.approx(0.0)
        for action in values
    )
    assert values["installed_apps"]["exact_d2_information"] == pytest.approx(
        math.log(3)
    )
    assert set(
        values["installed_apps"]["exact_d2_branch_actions"].values()
    ) == {"messaging_permissions"}
    assert all(
        values[action]["exact_d2_information"] == pytest.approx(0.0)
        for action in values
        if action != "installed_apps"
    )


def test_split_sizes_and_prompt_uniqueness_are_frozen():
    assert HYPOTHESIS_COUNT == 6
    assert len(SMOKE_PROMPTS) == 2
    assert len(FORMAL_PROMPTS) == 12
    assert len(set((*SMOKE_PROMPTS, *FORMAL_PROMPTS))) == 14
