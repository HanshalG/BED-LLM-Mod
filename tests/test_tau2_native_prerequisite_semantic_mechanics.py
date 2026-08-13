from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts import tau2_native_prerequisite_semantic_mechanics as mechanics
from scripts import tau2_native_prerequisite_semantic_verify as verifier
from scripts import tau2_native_prerequisite_semantic_aug13_execute as wrapper


def synthetic_public(family: str = "mms_home", count: int = 4):
    fields = mechanics.action_fields(family)
    return {
        "episode_index": 0,
        "family": family,
        "worlds": [
            {"world_id": f"w{index}", "description": f"world {index}"}
            for index in range(count)
        ],
        "actions": [
            {"action_id": action, "observable_fields": values}
            for action, values in fields.items()
        ],
        "legal_unlock": "installed_apps unlocks messaging_permissions",
    }


def first_value(options):
    return options[0]


def synthetic_payload(public, *, native_distinct=True, confidence=0.9):
    fields = mechanics.action_fields(public["family"])
    worlds = []
    for world_index, world in enumerate(public["worlds"]):
        predictions = {}
        for action, action_fields in fields.items():
            row = {name: first_value(values) for name, values in action_fields.items()}
            if native_distinct and action == "messaging_permissions":
                row = {
                    "sms": world_index in (0, 2),
                    "storage": world_index in (0, 1),
                    "phone": True,
                }
            row["confidence"] = confidence
            predictions[action] = row
        worlds.append({"world_id": world["world_id"], "predictions": predictions})
    return {"worlds": worlds}


def test_response_parser_rejects_reordered_worlds():
    public = synthetic_public()
    payload = synthetic_payload(public)
    payload["worlds"].reverse()
    with pytest.raises(ValueError, match="world order"):
        mechanics.parse_response(json.dumps(payload), public)


def test_likelihood_rows_are_normalized_and_confidence_bound():
    public = synthetic_public()
    parsed = mechanics.parse_response(json.dumps(synthetic_payload(public)), public)
    tables = mechanics.likelihood_tables(parsed)
    for table in tables.values():
        for row in table.values():
            assert math.isclose(sum(row.values()), 1.0, abs_tol=1e-12)
            assert max(row.values()) == pytest.approx(0.9)


def test_hidden_state_leakage_has_positive_total_variation():
    public = synthetic_public()
    payload = synthetic_payload(public, native_distinct=False)
    payload["worlds"][0]["predictions"]["status_bar"]["data_enabled"] = False
    payload["worlds"][1]["predictions"]["status_bar"]["data_enabled"] = True
    parsed = mechanics.parse_response(json.dumps(payload), public)
    table = mechanics.likelihood_tables(parsed)["status_bar"]
    categories = list(table["w0"])
    tv = 0.5 * sum(abs(table["w0"][key] - table["w1"][key]) for key in categories)
    assert tv > 0.15


def test_realized_native_observation_obeys_bayes_update():
    public = synthetic_public()
    parsed = mechanics.parse_response(json.dumps(synthetic_payload(public)), public)
    table = mechanics.likelihood_tables(parsed)["messaging_permissions"]
    outcome = mechanics.signature_key(
        parsed["worlds"][2]["predictions"]["messaging_permissions"]["signature"]
    )
    posterior = mechanics.posterior([0.25] * 4, table, outcome)
    assert posterior[2] == max(posterior)
    assert posterior[2] > 0.8


def test_delayed_native_information_changes_first_action():
    public = synthetic_public()
    parsed = mechanics.parse_response(json.dumps(synthetic_payload(public)), public)
    plan = mechanics.semantic_plan("mms_home", parsed)
    assert plan["greedy_first_action"] == "wifi_calling"
    assert plan["depth_two_first_action"] == "installed_apps"
    assert plan["horizon_gain_nats"] > 1.0


def test_prompt_omits_forbidden_endpoint_language():
    public = synthetic_public()
    text = json.dumps(mechanics.messages_for_episode(public)).casefold()
    for forbidden in ("task_id", "repair action", "information gain", "preferred action"):
        assert forbidden not in text


def test_response_schema_is_strict_and_closed():
    schema = mechanics.response_format("mms_home", 4)
    root = schema["json_schema"]
    assert root["strict"] is True
    assert root["schema"]["additionalProperties"] is False
    item = root["schema"]["properties"]["worlds"]["items"]
    assert item["additionalProperties"] is False


def test_adapter_payload_explicitly_disables_reasoning():
    adapter = object.__new__(mechanics.NonReasoningSeededAdapter)
    adapter._per_request_seed = type("Local", (), {"value": 123})()
    adapter.model_name = mechanics.MODEL_ID
    adapter.max_tokens = mechanics.MAX_TOKENS
    adapter.spec = type("Spec", (), {"reasoning_max_tokens": None, "reasoning_effort": None})()
    adapter.thinking = False
    adapter.seed = None
    adapter.request_seed = 0
    payload = mechanics.NonReasoningSeededAdapter._payload(
        adapter,
        [{"role": "user", "content": "x"}],
        0.0,
        1,
        response_format=mechanics.response_format("mms_home", 4),
    )
    assert payload["reasoning"] == {"enabled": False, "exclude": True}
    assert payload["seed"] == 123


def test_spearman_matches_perfect_order():
    assert mechanics.spearman([0, 1, 2, 3], [10, 11, 12, 13]) == pytest.approx(1.0)


def perfect_raw_bank():
    episodes = mechanics.mechanics_episodes()
    public = [mechanics.public_episode(episode, index) for index, episode in enumerate(episodes)]
    observed = mechanics.official_observations(episodes)
    responses = []
    for episode_public, rows in zip(public, observed, strict=True):
        payload = {"worlds": []}
        for index, observations in enumerate(rows):
            predictions = {
                action: dict(signature, confidence=0.95)
                for action, signature in observations.items()
            }
            payload["worlds"].append(
                {"world_id": f"w{index}", "predictions": predictions}
            )
        responses.append(json.dumps(payload))
    return {
        "model_id": mechanics.MODEL_ID,
        "seeds": list(mechanics.MODEL_SEEDS),
        "responses": responses,
    }


def test_independent_replay_matches_producer_on_perfect_bank():
    bank = perfect_raw_bank()
    independent = verifier.replay(bank)
    episodes = mechanics.mechanics_episodes()
    public = [mechanics.public_episode(episode, index) for index, episode in enumerate(episodes)]
    parsed = [
        mechanics.parse_response(raw, row)
        for raw, row in zip(bank["responses"], public, strict=True)
    ]
    observed = mechanics.official_observations(episodes)
    produced = mechanics.score_responses(episodes, parsed, observed)
    assert independent == produced
    assert independent["gates"]["all_semantic_and_planning_gates_pass"] is True


def write_verified_fixture(tmp_path: Path):
    bank = perfect_raw_bank()
    replayed = verifier.replay(bank)
    private = tmp_path / "private"
    private.mkdir()
    (private / "RAW_RESPONSES.json").write_text(json.dumps(bank))
    ordering = {
        "all_raw_responses_banked": True,
        "official_mechanics_loaded_after_raw_bank": True,
    }
    privacy = {
        "selected_task_ids_in_prompts": False,
        "source_fault_ids_in_prompts": False,
        "raw_tool_responses_in_prompts": False,
        "repair_or_endpoint_outcomes_in_prompts": False,
    }
    (private / "ORDERING.json").write_text(json.dumps(ordering))
    (private / "PROMPT_PRIVACY.json").write_text(json.dumps(privacy))
    usage = {
        "adapter_requests": 6,
        "http_attempts": 6,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.01,
    }
    serving = {
        "exact_six_accepted_requests": True,
        "exact_six_http_attempts": True,
        "zero_retries": True,
        "zero_reasoning_tokens": True,
        "zero_forced_exits": True,
        "within_stage_cap": True,
    }
    result = {
        **replayed,
        "status": "semantic_mechanics_pass",
        "authorizes": "prospective_development_protocol_only",
        "usage": usage,
        "serving_gates": serving,
        "ordering": ordering,
        "privacy": privacy,
        "development_confirmation_reserve_opened": False,
        "repair_or_task_success_endpoints_opened": False,
    }
    (tmp_path / "RESULT.json").write_text(json.dumps(result))
    return bank, result


def test_independent_verifier_accepts_exact_fixture(tmp_path):
    write_verified_fixture(tmp_path)
    assert verifier.verify(tmp_path)["all_pass"] is True


@pytest.mark.parametrize("tamper", ["raw", "metric", "ordering", "usage"])
def test_independent_verifier_rejects_tamper(tmp_path, tamper):
    bank, result = write_verified_fixture(tmp_path)
    if tamper == "raw":
        payload = json.loads(bank["responses"][0])
        payload["worlds"][0]["predictions"]["wifi_calling"]["enabled"] = not payload["worlds"][0]["predictions"]["wifi_calling"]["enabled"]
        bank["responses"][0] = json.dumps(payload)
        (tmp_path / "private/RAW_RESPONSES.json").write_text(json.dumps(bank))
    elif tamper == "metric":
        result["metrics"]["typed_signature_accuracy"] -= 0.1
        (tmp_path / "RESULT.json").write_text(json.dumps(result))
    elif tamper == "ordering":
        result["ordering"]["official_mechanics_loaded_after_raw_bank"] = False
        (tmp_path / "RESULT.json").write_text(json.dumps(result))
    else:
        result["usage"]["adapter_requests"] = 5
        (tmp_path / "RESULT.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {
        "total_credits_usd": credits,
        "total_usage_usd": usage,
        "balance_usd": credits - usage,
    }


def catalog(prompt="0.00000008", completion="0.00000018"):
    return {
        "data": [
            {
                "id": mechanics.MODEL_ID,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "response_format"],
                "pricing": {"prompt": prompt, "completion": completion},
            }
        ]
    }


def test_catalog_validation_covers_frozen_request():
    row = wrapper.validate_catalog(catalog())
    assert row["id"] == mechanics.MODEL_ID
    assert row["covered_prompt_tokens_at_live_price"] >= 12_000


def test_catalog_validation_rejects_price_exposure():
    with pytest.raises(RuntimeError, match="reservation"):
        wrapper.validate_catalog(catalog(prompt="0.000001", completion="0.000001"))


def test_prior_spend_counts_account_wide_usage():
    assert wrapper.prior_spend(live(wrapper.OPENING_USAGE_USD + 1.25)) == pytest.approx(1.25)


def test_live_validation_rejects_negative_boundary_delta():
    with pytest.raises(RuntimeError, match="below"):
        wrapper.validate_live(live(wrapper.OPENING_USAGE_USD - 0.01))


def test_initial_ledger_rejects_usage_race():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64}}
    with pytest.raises(RuntimeError, match="raced"):
        wrapper.initial_ledger(
            ready,
            live(wrapper.OPENING_USAGE_USD + wrapper.DAILY_CAP_USD),
        )


def test_reconcile_uses_larger_of_posted_and_local():
    ready = {"bindings": {"execution_binding_sha256": "b" * 64}}
    ledger = wrapper.initial_ledger(ready, live())
    updated = wrapper.reconcile(
        ledger,
        status="semantic_mechanics_pass",
        local_cost=0.02,
        live=live(wrapper.OPENING_USAGE_USD + 0.01),
    )
    assert updated["recorded_actual_spend_usd"] == pytest.approx(0.02)
