from __future__ import annotations

import json
from pathlib import Path
import threading
from datetime import datetime, timezone

import pytest

from scripts import tau2_mms_split_partition_calibration as split
from scripts import tau2_mms_split_partition_verify as verifier
from scripts import tau2_native_prerequisite_semantic_mechanics as base
from scripts import tau2_mms_split_partition_aug13_execute as wrapper


def perfect_responses():
    episodes = split.selected_episodes(); observations = base.official_observations(episodes); roots = []; natives = []
    for truth in observations:
        roots.append(json.dumps({"actions": [{"action_id": action, "all_worlds_same": len({split.canonical_json(world[action]) for world in truth}) == 1, "confidence": .95} for action in split.ROOTS]}))
        natives.append(json.dumps({"action_id": split.NATIVE, "world_groups": split.canonical_groups([world[split.NATIVE] for world in truth]), "confidence": .95}))
    return roots + natives


class FakeAdapter(split.SplitAdapter):
    def __init__(self, partial_path: Path, responses):
        self.partial_path = partial_path; self.responses = responses; self._per_request_seed = threading.local(); self.concurrency = 2
    def _complete_request(self, messages, temperature, n, max_tokens, response_format=None):
        del messages, temperature, n, max_tokens, response_format
        seed = self._per_request_seed.value
        seeds = split.ROOT_SEEDS + split.NATIVE_SEEDS
        return [self.responses[seeds.index(seed)]]
    def usage_snapshot(self):
        return {"adapter_requests": 12, "http_attempts": 12, "retry_count": 0, "provider_error_retries": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "adapter_cost_usd": .01, "adapter_prompt_tokens": 1, "adapter_completion_tokens": 1}


def requests_for(episodes):
    public = [split.public_episode(row, i) for i, row in enumerate(episodes)]; rows = []
    for i, row in enumerate(public): rows.append({"episode_index": i, "kind": "root", "seed": split.ROOT_SEEDS[i], "messages": split.root_messages(row), "response_format": split.root_response_format(), "max_tokens": split.ROOT_MAX_TOKENS})
    for i, row in enumerate(public): rows.append({"episode_index": i, "kind": "native", "seed": split.NATIVE_SEEDS[i], "messages": split.native_messages(row), "response_format": split.native_response_format(), "max_tokens": split.NATIVE_MAX_TOKENS})
    return rows


def test_separate_codecs_accept_perfect_responses():
    responses = perfect_responses()
    assert len(split.parse_root(responses[0])) == 8
    assert split.parse_native(responses[6])["groups"] == [0, 1, 2, 3]


def test_root_codec_rejects_native_payload():
    with pytest.raises(ValueError, match="root action array"):
        split.parse_root(perfect_responses()[6])


def test_native_codec_rejects_noncanonical_labels():
    payload = json.loads(perfect_responses()[6]); payload["world_groups"] = [1, 2, 3, 0]
    with pytest.raises(ValueError, match="native groups"):
        split.parse_native(json.dumps(payload))


def test_perfect_scoring_passes_all_gates():
    episodes = split.selected_episodes(); observations = base.official_observations(episodes); responses = perfect_responses(); result = split.score(episodes, [split.parse_root(responses[i]) for i in range(6)], [split.parse_native(responses[6+i]) for i in range(6)], observations)
    assert result["calibration_gates"]["all_calibration_gates_pass"] is True


def test_independent_replay_matches_producer_math():
    responses = perfect_responses(); raw = {"model_id": split.MODEL_ID, "requests": verifier.expected_requests(), "responses": responses}; replayed = verifier.replay(raw); episodes = split.selected_episodes(); observations = base.official_observations(episodes); produced = split.score(episodes, [split.parse_root(responses[i]) for i in range(6)], [split.parse_native(responses[6+i]) for i in range(6)], observations)
    assert replayed == produced


def test_full_producer_and_verifier_transaction_passes(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses()); result = split.run(output_dir=tmp_path, adapter=adapter, daily_budget_status={"rehearsal": True}); verification = verifier.verify(tmp_path, output_path=tmp_path / "VERIFICATION.json")
    assert result["status"] == "mms_split_partition_pass"
    assert verification["all_pass"] is True


def test_partial_bank_binds_kind_episode_and_seed(tmp_path):
    adapter = FakeAdapter(tmp_path / "partial.json", perfect_responses()); adapter.run_requests(requests_for(split.selected_episodes())); partial = json.loads((tmp_path / "partial.json").read_text()); assert verifier.validate_partial(partial) == perfect_responses()
    partial["completed"][0]["kind"] = "native"
    with pytest.raises(ValueError, match="partial row"):
        verifier.validate_partial(partial)


def test_native_semantic_tamper_fails_gate():
    responses = perfect_responses(); payload = json.loads(responses[6]); payload["world_groups"] = [0, 0, 0, 0]; responses[6] = json.dumps(payload); replayed = verifier.replay({"model_id": split.MODEL_ID, "requests": verifier.expected_requests(), "responses": responses})
    assert replayed["calibration_gates"]["all_calibration_gates_pass"] is False


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": credits - usage}


def catalog():
    return {"data": [{"id": split.MODEL_ID, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "structured_outputs"], "pricing": {"prompt": "0.00000008", "completion": "0.00000018"}}]}


def test_split_catalog_reserves_against_larger_root_output():
    model = wrapper.validate_catalog(catalog())
    assert model["covered_prompt_tokens_at_live_price"] == pytest.approx(47_975.0)


def test_initial_ledger_retains_partition_predecessor_spend():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"failure_sha256": "b" * 64, "ledger_sha256": "c" * 64, "recorded_spend_usd": .010897319000008565}}}
    ledger = wrapper.initial_ledger(ready, live())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(.010897319000008565)
    assert ledger["stage"]["maximum_http_attempts"] == 12


def test_preflight_accepts_bound_pristine_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", tmp_path / "failure.json")
    ready = wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=lambda: live(wrapper.OPENING_USAGE_USD + .012), catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(.012)


def test_preflight_refuses_existing_terminal(monkeypatch, tmp_path):
    terminal = tmp_path / "failure.json"
    terminal.write_text("{}\n")
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", terminal)
    with pytest.raises(RuntimeError, match="already terminal"):
        wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=live, catalog_reader=catalog)
