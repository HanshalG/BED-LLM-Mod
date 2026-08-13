from __future__ import annotations

import json
from pathlib import Path
import threading
from datetime import datetime, timezone

import pytest

from scripts import tau2_mms_partition_semantic_calibration as partition
from scripts import tau2_mms_partition_semantic_verify as verifier
from scripts import tau2_mms_array_semantic_calibration as first
from scripts import tau2_mms_checkpointed_semantic_calibration as second
from scripts import tau2_native_prerequisite_semantic_mechanics as base
from scripts import tau2_mms_partition_semantic_aug13_execute as wrapper


def perfect_responses():
    episodes = partition.selected_episodes(); observations = base.official_observations(episodes); responses = []
    for truth in observations:
        rows = []
        for action in base.MMS_ACTION_FIELDS:
            rows.append({"action_id": action, "world_groups": partition.canonical_groups([world[action] for world in truth]), "confidence": .95})
        responses.append(json.dumps({"actions": rows}))
    return responses


class FakeAdapter(partition.CheckpointingAdapter):
    def __init__(self, partial_path: Path, responses):
        self.partial_path = partial_path; self.responses = responses; self._per_request_seed = threading.local(); self.concurrency = 2
    def _complete_request(self, messages, temperature, n, max_tokens, response_format=None):
        del messages, temperature, n, max_tokens, response_format
        return [self.responses[partition.MODEL_SEEDS.index(self._per_request_seed.value)]]
    def usage_snapshot(self):
        return {"adapter_requests": 6, "http_attempts": 6, "retry_count": 0, "provider_error_retries": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "adapter_cost_usd": .01, "adapter_prompt_tokens": 1, "adapter_completion_tokens": 1}


def test_partition_cohort_is_fresh_and_exactly_structured():
    episodes = partition.selected_episodes(); prior = first.selected_episodes() + second.selected_episodes()
    assert [row["family"] for row in episodes] == ["mms_abroad"] * 3 + ["mms_home"] * 3
    assert {partition.episode_hash(row) for row in episodes}.isdisjoint({partition.episode_hash(row) for row in prior})


def test_parser_rejects_noncanonical_group_labels():
    public = partition.public_episode(partition.selected_episodes()[0], 0); payload = json.loads(perfect_responses()[0]); payload["actions"][0]["world_groups"] = [1, 1, 1, 1]
    with pytest.raises(ValueError, match="not canonical"):
        partition.parse(json.dumps(payload), public)


def test_perfect_partition_math_passes_all_gates():
    episodes = partition.selected_episodes(); observations = base.official_observations(episodes)
    parsed = [partition.parse(raw, partition.public_episode(episodes[index], index)) for index, raw in enumerate(perfect_responses())]
    scored = partition.score(episodes, parsed, observations)
    assert scored["calibration_gates"]["all_calibration_gates_pass"] is True
    assert scored["metrics"]["pair_relation_count"] == 324


def test_independent_replay_exactly_matches_producer():
    responses = perfect_responses(); bank = {"model_id": partition.MODEL_ID, "seeds": list(partition.MODEL_SEEDS), "responses": responses}
    replayed = verifier.replay(bank); episodes = partition.selected_episodes(); observations = base.official_observations(episodes)
    parsed = [partition.parse(raw, partition.public_episode(episodes[index], index)) for index, raw in enumerate(responses)]
    assert replayed == partition.score(episodes, parsed, observations)


def test_full_producer_and_independent_verifier_pass(tmp_path):
    adapter = FakeAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    result = partition.run(output_dir=tmp_path, adapter=adapter, daily_budget_status={"rehearsal": True})
    verification = verifier.verify(tmp_path, output_path=tmp_path / "VERIFICATION.json")
    assert result["status"] == "mms_partition_semantic_pass"
    assert verification["all_pass"] is True


def test_independent_replay_rejects_partition_tamper():
    responses = perfect_responses(); payload = json.loads(responses[0]); payload["actions"][0]["world_groups"] = [0, 1, 0, 0]; responses[0] = json.dumps(payload)
    replayed = verifier.replay({"model_id": partition.MODEL_ID, "seeds": list(partition.MODEL_SEEDS), "responses": responses})
    assert replayed["calibration_gates"]["all_calibration_gates_pass"] is False


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": credits - usage}


def catalog():
    return {"data": [{"id": partition.MODEL_ID, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "structured_outputs"], "pricing": {"prompt": "0.00000008", "completion": "0.00000018"}}]}


def test_partition_catalog_uses_own_output_cap():
    model = wrapper.validate_catalog(catalog())
    assert model["covered_prompt_tokens_at_live_price"] == pytest.approx(69375.0)


def test_initial_ledger_retains_predecessor_spend():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"result_sha256": "b" * 64, "ledger_sha256": "c" * 64, "recorded_spend_usd": .009814359}}}
    assert wrapper.initial_ledger(ready, live())["recorded_actual_spend_usd"] == pytest.approx(.009814359)


def test_preflight_accepts_bound_pristine_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run"); monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json"); monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "result.json"); monkeypatch.setattr(wrapper, "DAILY_FAILURE", tmp_path / "failure.json")
    ready = wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=lambda: live(wrapper.OPENING_USAGE_USD + .011), catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(.011)
