from __future__ import annotations

import json
from pathlib import Path
import threading
from datetime import datetime, timezone

import pytest

from scripts import tau2_mms_checkpointed_semantic_calibration as checkpointed
from scripts import tau2_mms_checkpointed_semantic_verify as verifier
from scripts import tau2_mms_array_semantic_calibration as prior
from scripts import tau2_native_prerequisite_semantic_mechanics as base
from scripts import tau2_mms_checkpointed_semantic_aug13_execute as wrapper


def perfect_responses():
    episodes = checkpointed.selected_episodes()
    observed = base.official_observations(episodes)
    responses = []
    for index, truth in enumerate(observed):
        public = checkpointed.public_episode(episodes[index], index)
        worlds = []
        for world_index, signatures in enumerate(truth):
            actions = []
            for action in public["actions"]:
                signature = signatures[action["action_id"]]
                actions.append({
                    "action_id": action["action_id"],
                    "fields": [{"field_id": field["field_id"], "value": str(signature[field["field_id"]]).lower() if isinstance(signature[field["field_id"]], bool) else str(signature[field["field_id"]])} for field in action["fields"]],
                    "confidence": 0.95,
                })
            worlds.append({"world_id": f"w{world_index}", "actions": actions})
        responses.append(json.dumps({"worlds": worlds}))
    return responses


class FakeCheckpointAdapter(checkpointed.CheckpointingAdapter):
    def __init__(self, partial_path: Path, responses, fail_index=None):
        self.partial_path = partial_path
        self.responses = responses
        self.fail_index = fail_index
        self._per_request_seed = threading.local()
        self.concurrency = 2

    def _complete_request(self, messages, temperature, n, max_tokens, response_format=None):
        del messages, temperature, n, max_tokens, response_format
        seed = self._per_request_seed.value
        index = checkpointed.MODEL_SEEDS.index(seed)
        if index == self.fail_index:
            raise RuntimeError("provider-error response persisted after retries")
        return [self.responses[index]]

    def usage_snapshot(self):
        return {"adapter_requests": 6, "http_attempts": 6, "retry_count": 0, "provider_error_retries": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "adapter_cost_usd": 0.01, "adapter_prompt_tokens": 1, "adapter_completion_tokens": 1}


def test_fresh_cohort_is_disjoint_and_balanced():
    fresh = checkpointed.selected_episodes()
    old = prior.selected_episodes()
    assert [row["family"] for row in fresh] == ["mms_abroad"] * 3 + ["mms_home"] * 3
    assert {checkpointed.episode_hash(row) for row in fresh}.isdisjoint({prior.episode_hash(row) for row in old})


def test_partial_checkpoint_survives_one_provider_error(tmp_path):
    responses = perfect_responses()
    adapter = FakeCheckpointAdapter(tmp_path / "partial.json", responses, fail_index=5)
    with pytest.raises(RuntimeError, match="provider-error"):
        adapter.chat_complete_seeded_messages_batched_structured(
            [[{"role": "user", "content": "x"}]] * 6,
            checkpointed.MODEL_SEEDS, temperature=0.0,
            response_format=checkpointed.response_format(), max_new_tokens=6000,
        )
    bank = json.loads((tmp_path / "partial.json").read_text())
    assert bank["complete"] is False
    assert [row["index"] for row in bank["completed"]] == [0, 1, 2, 3, 4]
    assert verifier.validate_partial_bank(bank, require_complete=False) == responses[:5]
    with pytest.raises(ValueError, match="incomplete"):
        verifier.validate_partial_bank(bank, require_complete=True)


def test_complete_checkpoint_is_ordered_despite_completion_order(tmp_path):
    responses = perfect_responses()
    adapter = FakeCheckpointAdapter(tmp_path / "partial.json", responses)
    actual = adapter.chat_complete_seeded_messages_batched_structured(
        [[{"role": "user", "content": "x"}]] * 6,
        checkpointed.MODEL_SEEDS, temperature=0.0,
        response_format=checkpointed.response_format(), max_new_tokens=6000,
    )
    assert actual == responses
    bank = json.loads((tmp_path / "partial.json").read_text())
    assert bank["complete"] is True
    assert verifier.validate_partial_bank(bank, require_complete=True) == responses


def test_independent_replay_passes_perfect_fresh_bank():
    replayed = verifier.replay({"model_id": checkpointed.MODEL_ID, "seeds": list(checkpointed.MODEL_SEEDS), "responses": perfect_responses()})
    assert replayed["calibration_gates"]["all_calibration_gates_pass"] is True
    assert replayed["metrics"]["cell_count"] == 216


def test_full_producer_verifier_transaction_passes(tmp_path):
    adapter = FakeCheckpointAdapter(tmp_path / "private/PARTIAL_RAW_RESPONSES.json", perfect_responses())
    result = checkpointed.run(output_dir=tmp_path, adapter=adapter, daily_budget_status={"rehearsal": True})
    verification = verifier.verify(tmp_path, output_path=tmp_path / "VERIFICATION.json")
    assert result["status"] == "mms_checkpointed_semantic_pass"
    assert verification["all_pass"] is True


def live(usage=wrapper.OPENING_USAGE_USD, credits=245.0):
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": credits - usage}


def catalog():
    return {"data": [{"id": checkpointed.MODEL_ID, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "structured_outputs"], "pricing": {"prompt": "0.00000008", "completion": "0.00000018"}}]}


def test_initial_ledger_retains_chained_predecessor_spend():
    ready = {"bindings": {"execution_binding_sha256": "a" * 64, "predecessor": {"failure_sha256": "b" * 64, "ledger_sha256": "c" * 64, "recorded_spend_usd": 0.005350987795992523}}}
    ledger = wrapper.initial_ledger(ready, live())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.005350987795992523)


def test_preflight_accepts_bound_pristine_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(wrapper, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(wrapper, "DAILY_RESULT", tmp_path / "daily-result.json")
    monkeypatch.setattr(wrapper, "DAILY_FAILURE", tmp_path / "daily-failure.json")
    ready = wrapper.preflight(now=datetime(2026, 8, 13, 9, tzinfo=timezone.utc), live_reader=lambda: live(wrapper.OPENING_USAGE_USD + 0.006), catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(0.006)
