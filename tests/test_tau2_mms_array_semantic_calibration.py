from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from scripts import tau2_mms_array_semantic_calibration as array
from scripts import tau2_mms_array_semantic_verify as verifier
from scripts import tau2_mms_array_semantic_aug13_execute as wrapper
from scripts import tau2_native_prerequisite_semantic_mechanics as base


def perfect_payload(public, observations):
    worlds = []
    for index, truth in enumerate(observations):
        actions = []
        for action in public["actions"]:
            signature = truth[action["action_id"]]
            actions.append(
                {
                    "action_id": action["action_id"],
                    "fields": [
                        {
                            "field_id": field["field_id"],
                            "value": (
                                str(signature[field["field_id"]]).lower()
                                if isinstance(signature[field["field_id"]], bool)
                                else str(signature[field["field_id"]])
                            ),
                        }
                        for field in action["fields"]
                    ],
                    "confidence": 0.95,
                }
            )
        worlds.append({"world_id": f"w{index}", "actions": actions})
    return {"worlds": worlds}


def test_fresh_manifest_reconstructs_six_mms_reserve_episodes():
    episodes = array.selected_episodes()
    assert [row["family"] for row in episodes] == ["mms_abroad"] * 3 + ["mms_home"] * 3
    assert all(len(row["worlds"]) == 4 for row in episodes)
    assert len({array.episode_hash(row) for row in episodes}) == 6


def test_array_parser_accepts_perfect_typed_payload():
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    public = array.public_episode(episodes[0], 0)
    parsed = array.parse(json.dumps(perfect_payload(public, observed[0])), public)
    assert len(parsed["worlds"]) == 4
    assert list(parsed["worlds"][0]["predictions"]) == list(base.MMS_ACTION_FIELDS)


def test_array_parser_rejects_reordered_action_rows():
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    public = array.public_episode(episodes[0], 0)
    payload = perfect_payload(public, observed[0])
    payload["worlds"][0]["actions"][0], payload["worlds"][0]["actions"][1] = (
        payload["worlds"][0]["actions"][1], payload["worlds"][0]["actions"][0]
    )
    with pytest.raises(ValueError, match="action order"):
        array.parse(json.dumps(payload), public)


def test_array_parser_rejects_reordered_field_rows():
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    public = array.public_episode(episodes[0], 0)
    payload = perfect_payload(public, observed[0])
    network = payload["worlds"][0]["actions"][4]["fields"]
    network[0], network[1] = network[1], network[0]
    with pytest.raises(ValueError, match="field order"):
        array.parse(json.dumps(payload), public)


def test_perfect_fresh_cohort_passes_calibration_gates():
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    parsed = []
    for index, (episode, truth) in enumerate(zip(episodes, observed, strict=True)):
        public = array.public_episode(episode, index)
        parsed.append(array.parse(json.dumps(perfect_payload(public, truth)), public))
    scores = base.score_responses(episodes, parsed, observed)
    gates = array.calibration_gates(scores)
    assert gates["all_calibration_gates_pass"] is True
    assert scores["metrics"]["cell_count"] == 216


def test_schema_uses_ordered_arrays_and_closed_objects():
    schema = array.response_format()["json_schema"]
    assert schema["strict"] is True
    root = schema["schema"]
    assert root["additionalProperties"] is False
    world = root["properties"]["worlds"]["items"]
    assert world["properties"]["actions"]["type"] == "array"
    assert world["additionalProperties"] is False


def perfect_bank():
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    responses = []
    for index, (episode, truth) in enumerate(zip(episodes, observed, strict=True)):
        public = array.public_episode(episode, index)
        responses.append(json.dumps(perfect_payload(public, truth)))
    return {"model_id": array.MODEL_ID, "seeds": list(array.MODEL_SEEDS), "responses": responses}


def test_independent_replay_matches_producer_math():
    bank = perfect_bank()
    replayed = verifier.replay(bank)
    episodes = array.selected_episodes()
    observed = base.official_observations(episodes)
    parsed = []
    for index, (episode, truth, raw) in enumerate(zip(episodes, observed, bank["responses"], strict=True)):
        del truth
        parsed.append(array.parse(raw, array.public_episode(episode, index)))
    scores = base.score_responses(episodes, parsed, observed)
    produced = {
        "metrics": scores["metrics"],
        "episode_metrics": scores["episode_metrics"],
        "calibration_gates": array.calibration_gates(scores),
    }
    assert replayed == produced


def write_fixture(tmp_path: Path):
    bank = perfect_bank()
    replayed = verifier.replay(bank)
    private = tmp_path / "private"; private.mkdir()
    ordering = {"all_raw_responses_banked": True, "official_calibration_loaded_after_raw_bank": True}
    privacy = {"selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False, "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False}
    (private / "RAW_RESPONSES.json").write_text(json.dumps(bank))
    (private / "ORDERING.json").write_text(json.dumps(ordering))
    (private / "PROMPT_PRIVACY.json").write_text(json.dumps(privacy))
    usage = {"adapter_requests": 6, "http_attempts": 6, "retry_count": 0, "provider_error_retries": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "run_cost_usd": 0.01}
    serving = {"exact_six_accepted_requests": True, "exact_six_http_attempts": True, "zero_retries": True, "zero_provider_error_retries": True, "zero_reasoning_tokens": True, "zero_forced_exits": True, "within_stage_cap": True}
    result = {**replayed, "status": "mms_array_semantic_pass", "authorizes": "prospective_paired_development_protocol_only", "usage": usage, "serving_gates": serving, "ordering": ordering, "privacy": privacy, "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}
    (tmp_path / "RESULT.json").write_text(json.dumps(result))
    return bank, result


def test_independent_verifier_accepts_exact_fixture(tmp_path):
    write_fixture(tmp_path)
    assert verifier.verify(tmp_path)["all_pass"] is True


@pytest.mark.parametrize("tamper", ["raw", "metric", "ordering", "usage"])
def test_independent_verifier_rejects_tamper(tmp_path, tamper):
    bank, result = write_fixture(tmp_path)
    if tamper == "raw":
        payload = json.loads(bank["responses"][0]); payload["worlds"][0]["actions"][0], payload["worlds"][0]["actions"][1] = payload["worlds"][0]["actions"][1], payload["worlds"][0]["actions"][0]; bank["responses"][0] = json.dumps(payload); (tmp_path / "private/RAW_RESPONSES.json").write_text(json.dumps(bank))
    elif tamper == "metric":
        result["metrics"]["typed_signature_accuracy"] -= 0.1; (tmp_path / "RESULT.json").write_text(json.dumps(result))
    elif tamper == "ordering":
        result["ordering"]["official_calibration_loaded_after_raw_bank"] = False; (tmp_path / "RESULT.json").write_text(json.dumps(result))
    else:
        result["usage"]["adapter_requests"] = 5; (tmp_path / "RESULT.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": credits - usage}


def catalog():
    return {
        "data": [
            {
                "id": array.MODEL_ID,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "structured_outputs"],
                "pricing": {"prompt": "0.00000008", "completion": "0.00000018"},
            }
        ]
    }


def test_preflight_accepts_bound_pristine_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "daily-result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", tmp_path / "daily-failure.json")
    ready = wrapper.preflight(
        now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc),
        live_reader=lambda: live(wrapper.OPENING_USAGE_USD + 0.003),
        catalog_reader=catalog,
    )
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["model_calls_made"] == 0
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(0.003)


def test_initial_ledger_retains_predecessor_cost_when_posting_lags():
    ready = {
        "bindings": {
            "execution_binding_sha256": "a" * 64,
            "predecessor": {
                "failure_sha256": "b" * 64,
                "ledger_sha256": "c" * 64,
                "recorded_spend_usd": 0.002086228884,
            },
        }
    }
    ledger = wrapper.initial_ledger(ready, live())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.002086228884)


def test_initial_ledger_uses_larger_live_account_spend():
    ready = {
        "bindings": {
            "execution_binding_sha256": "a" * 64,
            "predecessor": {
                "failure_sha256": "b" * 64,
                "ledger_sha256": "c" * 64,
                "recorded_spend_usd": 0.002,
            },
        }
    }
    ledger = wrapper.initial_ledger(ready, live(wrapper.OPENING_USAGE_USD + 0.5))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.5)


def test_reconcile_adds_local_cost_to_execution_opening():
    ready = {
        "bindings": {
            "execution_binding_sha256": "a" * 64,
            "predecessor": {
                "failure_sha256": "b" * 64,
                "ledger_sha256": "c" * 64,
                "recorded_spend_usd": 0.002,
            },
        }
    }
    ledger = wrapper.initial_ledger(ready, live(wrapper.OPENING_USAGE_USD + 0.002))
    updated = wrapper.reconcile(ledger, status="mms_array_semantic_pass", local_cost=0.003, live=live(wrapper.OPENING_USAGE_USD + 0.004))
    assert updated["recorded_actual_spend_usd"] == pytest.approx(0.005)
