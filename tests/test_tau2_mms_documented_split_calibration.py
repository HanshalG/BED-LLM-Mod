from __future__ import annotations

import json
from pathlib import Path
import threading
from datetime import datetime, timezone

import pytest

from scripts import tau2_mms_documented_split_calibration as documented
from scripts import tau2_mms_documented_split_verify as verifier
from scripts import tau2_native_prerequisite_semantic_mechanics as base
from scripts import tau2_mms_documented_split_aug13_execute as wrapper


def perfect_responses():
    episodes = documented.selected_episodes()
    observations = base.official_observations(episodes)
    roots = []
    natives = []
    for truth in observations:
        roots.append(json.dumps({"actions": [{"action_id": action, "all_worlds_same": len({documented.canonical_json(world[action]) for world in truth}) == 1, "confidence": .95} for action in documented.ROOTS]}))
        natives.append(json.dumps({"action_id": documented.NATIVE, "world_groups": documented.canonical_groups([world[documented.NATIVE] for world in truth]), "confidence": .95}))
    return roots + natives


class FakeAdapter(documented.DocumentedSplitAdapter):
    def __init__(self, partial_path: Path, responses, fail_seed=None):
        self.partial_path = partial_path
        self.responses = responses
        self.fail_seed = fail_seed
        self._per_request_seed = threading.local()
        self.concurrency = 2

    def _complete_request(self, messages, temperature, n, max_tokens, response_format=None):
        del messages, temperature, n, max_tokens, response_format
        seed = self._per_request_seed.value
        if seed == self.fail_seed:
            raise RuntimeError("provider failure")
        seeds = documented.ROOT_SEEDS + documented.NATIVE_SEEDS
        return [self.responses[seeds.index(seed)]]

    def usage_snapshot(self):
        return {"adapter_requests": 12, "http_attempts": 12, "retry_count": 0, "provider_error_retries": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "adapter_cost_usd": .01, "adapter_prompt_tokens": 1, "adapter_completion_tokens": 1}


def test_documented_prompts_include_public_contract_without_outcomes():
    episode = documented.public_episode(documented.selected_episodes()[0], 0)
    root = documented.root_messages(episode)
    native = documented.native_messages(episode)
    assert "check_network_status" in root[1]["content"]
    assert "check_app_permissions" in native[1]["content"]
    assert "sms, storage, and phone" in native[1]["content"]
    assert "task_id" not in documented.canonical_json(root + native)
    assert tuple(row["action_id"] for row in documented.source_audit.ROOT_TOOL_CONTRACT) == documented.ROOTS


def test_documented_codecs_accept_perfect_responses():
    responses = perfect_responses()
    assert len(documented.parse_root(responses[0])) == 8
    assert documented.parse_native(responses[6])["groups"] == [0, 1, 2, 3]


def test_documented_root_codec_rejects_native_payload():
    with pytest.raises(ValueError, match="root action array"):
        documented.parse_root(perfect_responses()[6])


def test_documented_native_codec_rejects_noncanonical_labels():
    payload = json.loads(perfect_responses()[6])
    payload["world_groups"] = [1, 2, 3, 0]
    with pytest.raises(ValueError, match="native groups"):
        documented.parse_native(json.dumps(payload))


def test_perfect_documented_scoring_passes_all_gates():
    responses = perfect_responses()
    replayed = verifier.replay({"model_id": documented.MODEL_ID, "requests": verifier.expected_requests(), "responses": responses})
    assert replayed["calibration_gates"]["all_calibration_gates_pass"] is True


def test_independent_replay_matches_producer_math():
    responses = perfect_responses()
    episodes = documented.selected_episodes()
    observations = base.official_observations(episodes)
    produced = documented.score(episodes, [documented.parse_root(responses[index]) for index in range(6)], [documented.parse_native(responses[6 + index]) for index in range(6)], observations)
    replayed = verifier.replay({"model_id": documented.MODEL_ID, "requests": verifier.expected_requests(), "responses": responses})
    assert replayed == produced


def test_native_collapse_fails_semantic_and_planning_gates():
    responses = perfect_responses()
    for index in range(6, 12):
        payload = json.loads(responses[index])
        payload["world_groups"] = [0, 0, 0, 0]
        responses[index] = json.dumps(payload)
    replayed = verifier.replay({"model_id": documented.MODEL_ID, "requests": verifier.expected_requests(), "responses": responses})
    assert replayed["calibration_gates"]["all_six_native_partitions_exact"] is False
    assert replayed["calibration_gates"]["all_calibration_gates_pass"] is False


def test_full_documented_transaction_and_verifier_pass(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    result = documented.run(output_dir=tmp_path, adapter=adapter, daily_budget_status={"rehearsal": True})
    verification = verifier.verify(tmp_path, output_path=tmp_path / "VERIFICATION.json")
    assert result["status"] == "mms_documented_split_pass"
    assert verification["all_pass"] is True


def test_partial_bank_binds_prompt_kind_episode_and_seed(tmp_path):
    episodes = documented.selected_episodes()
    adapter = FakeAdapter(tmp_path / "partial.json", perfect_responses())
    adapter.run_requests(documented.build_requests(episodes))
    partial = json.loads((tmp_path / "partial.json").read_text())
    assert verifier.validate_partial(partial) == perfect_responses()
    partial["requests"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="partial identity"):
        verifier.validate_partial(partial)


def test_provider_failure_banks_completed_subset_and_fails_closed(tmp_path):
    adapter = FakeAdapter(tmp_path / "partial.json", perfect_responses(), fail_seed=documented.NATIVE_SEEDS[-1])
    with pytest.raises(RuntimeError, match="provider failure"):
        adapter.run_requests(documented.build_requests(documented.selected_episodes()))
    partial = json.loads((tmp_path / "partial.json").read_text())
    assert partial["complete"] is False
    assert len(partial["completed"]) == 11
    with pytest.raises(ValueError, match="incomplete"):
        verifier.validate_partial(partial)


def test_verifier_rejects_prompt_hash_and_result_tamper(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    documented.run(output_dir=tmp_path, adapter=adapter)
    raw_path = tmp_path / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["requests"][0]["prompt_sha256"] = "f" * 64
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="raw bank"):
        verifier.verify(tmp_path)


def test_verifier_rejects_privacy_and_metric_tamper(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    documented.run(output_dir=tmp_path, adapter=adapter)
    privacy_path = tmp_path / "private/PROMPT_PRIVACY.json"
    privacy = json.loads(privacy_path.read_text())
    privacy["raw_tool_responses_in_prompts"] = True
    privacy_path.write_text(json.dumps(privacy))
    with pytest.raises(ValueError, match="verification failed"):
        verifier.verify(tmp_path)

    privacy["raw_tool_responses_in_prompts"] = False
    privacy_path.write_text(json.dumps(privacy))
    result_path = tmp_path / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["root_exact_count"] = 0
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="metrics differs"):
        verifier.verify(tmp_path)


def test_verifier_rejects_ordering_tamper(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    documented.run(output_dir=tmp_path, adapter=adapter)
    ordering_path = tmp_path / "private/ORDERING.json"
    ordering = json.loads(ordering_path.read_text())
    ordering["official_calibration_loaded_after_complete_bank"] = False
    ordering_path.write_text(json.dumps(ordering))
    with pytest.raises(ValueError, match="verification failed"):
        verifier.verify(tmp_path)


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": credits - usage}


def catalog(prompt="0.00000008", completion="0.00000018"):
    return {"data": [{"id": documented.MODEL_ID, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "structured_outputs"], "pricing": {"prompt": prompt, "completion": completion}}]}


def test_documented_catalog_reserves_against_root_output_cap():
    model = wrapper.validate_catalog(catalog())
    assert model["covered_prompt_tokens_at_live_price"] == pytest.approx(47_975.0)


def test_documented_catalog_rejects_insufficient_request_coverage():
    with pytest.raises(RuntimeError, match="no longer covers prompt"):
        wrapper.validate_catalog(catalog(prompt="0.000001", completion="0.000004"))


def test_predecessor_requires_verified_semantic_null(monkeypatch, tmp_path):
    result = json.loads(wrapper.PREDECESSOR_RESULT.read_text())
    ledger = json.loads(wrapper.PREDECESSOR_LEDGER.read_text())
    result["verification"]["all_pass"] = False
    result_path = tmp_path / "result.json"
    ledger_path = tmp_path / "ledger.json"
    report_path = tmp_path / "report.md"
    result_path.write_text(json.dumps(result))
    ledger_path.write_text(json.dumps(ledger))
    report_path.write_text("terminal\n")
    monkeypatch.setattr(wrapper, "PREDECESSOR_RESULT", result_path)
    monkeypatch.setattr(wrapper, "PREDECESSOR_LEDGER", ledger_path)
    monkeypatch.setattr(wrapper, "PREDECESSOR_REPORT", report_path)
    with pytest.raises(RuntimeError, match="predecessor malformed"):
        wrapper.validate_predecessor()


def test_initial_ledger_retains_verified_predecessor_spend():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"result_sha256": "b" * 64, "ledger_sha256": "c" * 64, "report_sha256": "d" * 64, "recorded_spend_usd": .011792256999996198}}}
    ledger = wrapper.initial_ledger(ready, live())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(.011792256999996198)
    assert ledger["stage"]["maximum_http_attempts"] == 12


def test_initial_ledger_rejects_account_usage_race():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"result_sha256": "b" * 64, "ledger_sha256": "c" * 64, "report_sha256": "d" * 64, "recorded_spend_usd": .011792256999996198}}}
    with pytest.raises(RuntimeError, match="usage raced"):
        wrapper.initial_ledger(ready, live(wrapper.OPENING_USAGE_USD + 4.95))


def test_reconcile_uses_larger_of_posted_and_local_cost():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"result_sha256": "b" * 64, "ledger_sha256": "c" * 64, "report_sha256": "d" * 64, "recorded_spend_usd": .011792256999996198}}}
    ledger = wrapper.initial_ledger(ready, live(wrapper.OPENING_USAGE_USD + .02))
    updated = wrapper.reconcile(ledger, status="mms_documented_split_null", local_cost=.01, live=live(wrapper.OPENING_USAGE_USD + .025))
    assert updated["recorded_actual_spend_usd"] == pytest.approx(.03)


def test_preflight_accepts_bound_pristine_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", tmp_path / "failure.json")
    ready = wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=lambda: live(wrapper.OPENING_USAGE_USD + .02), catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(.02)


def test_preflight_refuses_existing_terminal(monkeypatch, tmp_path):
    terminal = tmp_path / "failure.json"
    terminal.write_text("{}\n")
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", terminal)
    with pytest.raises(RuntimeError, match="already terminal"):
        wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=live, catalog_reader=catalog)
