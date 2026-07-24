import json
import math

import pytest

from scripts.tau2_account_joint_likelihood_v5 import (
    ACCOUNT_HYPOTHESES,
    ALL_ACTIONS,
    FORMAL_PROMPTS,
    SMOKE_PROMPTS,
    _confirmation_summary,
    _ranking_summary,
    joint_likelihood_messages,
    parse_joint_likelihood,
    predicted_action_values,
    sequence_information,
)


def _response(overrides=None):
    overrides = overrides or {}
    actions = []
    for action_id in ALL_ACTIONS:
        predictions = []
        for hypothesis in ACCOUNT_HYPOTHESES:
            hypothesis_id = hypothesis["id"]
            outcome = overrides.get(
                (action_id, hypothesis_id),
                f"same {action_id}",
            )
            predictions.append(
                {"hypothesis_id": hypothesis_id, "outcome": outcome}
            )
        actions.append({"action_id": action_id, "predictions": predictions})
    return json.dumps({"actions": actions})


def _account_table():
    overrides = {}
    for hypothesis in ACCOUNT_HYPOTHESES:
        hypothesis_id = hypothesis["id"]
        overrides["line_details", hypothesis_id] = (
            f"{hypothesis['data_allowance']} {hypothesis['account_roaming']}"
        )
        overrides["data_usage", hypothesis_id] = hypothesis["data_allowance"]
    return parse_joint_likelihood(_response(overrides))


def test_joint_prompt_contains_no_scores_or_official_outputs():
    messages = joint_likelihood_messages(SMOKE_PROMPTS[0])
    text = messages[-1]["content"].lower()
    assert "information gain" in text
    assert "do not report" in text
    assert "official" not in text
    assert "preferred policy" in text
    assert "line_details" in text


def test_joint_parser_normalizes_and_preserves_complete_order():
    parsed = parse_joint_likelihood(
        _response({("status_bar", "h1"): "  AIRPLANE   MODE  "})
    )
    assert parsed["observations"]["h1"]["status_bar"] == "airplane mode"
    assert [row["action_id"] for row in parsed["actions"]] == list(ALL_ACTIONS)


def test_joint_parser_rejects_missing_or_reordered_actions():
    payload = json.loads(_response())
    payload["actions"] = payload["actions"][:-1]
    with pytest.raises(ValueError, match="every action"):
        parse_joint_likelihood(json.dumps(payload))

    payload = json.loads(_response())
    payload["actions"][0], payload["actions"][1] = (
        payload["actions"][1],
        payload["actions"][0],
    )
    with pytest.raises(ValueError, match="order changed"):
        parse_joint_likelihood(json.dumps(payload))


def test_joint_table_recovers_zero_information_setup_and_full_unlock():
    observations = _account_table()["observations"]
    values = predicted_action_values(observations)
    assert all(
        values[action]["predicted_d1_information"] == pytest.approx(0.0)
        for action in values
    )
    assert values["customer_lookup"]["predicted_d2_information"] == pytest.approx(
        math.log(4)
    )
    branch_actions = values["customer_lookup"]["predicted_d2_branch_actions"]
    assert set(branch_actions.values()) == {"line_details"}
    assert all(
        values[action]["predicted_d2_information"] == pytest.approx(0.0)
        for action in values
        if action != "customer_lookup"
    )


def test_sequence_information_matches_manual_partition():
    observations = _account_table()["observations"]
    assert sequence_information(
        observations, "customer_lookup", "line_details"
    ) == pytest.approx(math.log(4))
    assert sequence_information(
        observations, "status_bar", "network_status"
    ) == pytest.approx(0.0)


def test_smoke_summary_requires_both_joint_mechanism_passes():
    action_rows = [
        {
            "action_id": action,
            "predicted_d2_information": (
                math.log(4) if action == "customer_lookup" else 0.0
            ),
            "exact_d2_information": (
                math.log(4) if action == "customer_lookup" else 0.0
            ),
        }
        for action in (
            "status_bar",
            "network_status",
            "speed_test",
            "payment_request",
            "sim_status",
            "customer_lookup",
        )
    ]
    records = [
        {
            "all_root_actions_equivalent": True,
            "line_details_unique_outcomes": 4,
            "d1_selected_action": "status_bar",
            "d2_selected_action": "customer_lookup",
            "lookup_followup_action": "line_details",
            "actions": action_rows,
            "actions_by_id": {row["action_id"]: row for row in action_rows},
        }
        for _ in SMOKE_PROMPTS
    ]
    summary = _ranking_summary(
        records,
        {"physical_requests": 2, "reasoning_tokens": 0},
        stage="serving_smoke",
    )
    assert summary["gates"]["all_pass"] is True


def test_formal_prompts_are_distinct_from_v4_wording():
    assert len(FORMAL_PROMPTS) == 12
    assert len(set(FORMAL_PROMPTS)) == 12


def test_confirmation_summary_checks_truth_and_entropy_controls():
    records = []
    for _ in range(24):
        records.append(
            {
                "d2_root_action": "customer_lookup",
                "d1_final_entropy_nats": math.log(4),
                "d2_final_entropy_nats": 0.0,
                "random_final_entropy_nats": math.log(4),
                "d1_truth_log_posterior": -math.log(4),
                "d2_truth_log_posterior": 0.0,
                "random_truth_log_posterior": -math.log(4),
                "d1_entropy_auc_nats": 2 * math.log(4),
                "d2_entropy_auc_nats": 1.5 * math.log(4),
            }
        )
    summary = _confirmation_summary(
        records,
        {"physical_requests": 24, "reasoning_tokens": 0},
    )
    assert summary["gates"]["all_pass"] is True
    assert summary["mean_d1_minus_d2_final_entropy_nats"] == pytest.approx(
        math.log(4)
    )
