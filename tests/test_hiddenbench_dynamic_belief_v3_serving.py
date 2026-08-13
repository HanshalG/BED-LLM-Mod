from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts import hiddenbench_dynamic_belief_v3_serving as serving
from scripts import hiddenbench_dynamic_belief_v3_verify as verifier


PRIOR = [0.3020790364377281, 0.3311198463454728, 0.3668011172167991]
MATRICES = {
    "Q1": [[0.3948265467869598, 0.19010130093847924, 0.41507215227456107], [0.1245820178848054, 0.49403480485986645, 0.3813831772553282], [0.2737094582833043, 0.5585085648866983, 0.16778197682999738]],
    "Q2": [[0.4580280716730286, 0.435056056747435, 0.10691587157953637], [0.3412699457891834, 0.28578983611772907, 0.37294021809308764], [0.5101154216353196, 0.056363861222295725, 0.43352071714238466]],
    "Q3": [[0.28823382651828927, 0.3820814015489915, 0.3296847719327192], [0.20728029576944182, 0.3079090554284516, 0.4848106488021066], [0.5855785358224683, 0.046663042907709375, 0.3677584212698224]],
    "Q4": [[0.3907260906212814, 0.49231781267330077, 0.11695609670541787], [0.2848881483615605, 0.353874093577153, 0.3612377580612865], [0.18880576853306097, 0.2549500443683205, 0.5562441870986184]],
}
REFRESHED = {
    "Q1": {"C1": [0.4270021267006022, 0.13231773616888345, 0.44068013713051435], "C2": [0.09156163633190591, 0.4349319134379601, 0.47350645023013394], "C3": [0.3611073114151607, 0.4695388667668736, 0.1693538218179656]},
    "Q2": {"C1": [0.22290836708797698, 0.25140300512564545, 0.5256886277863775], "C2": [0.4396737050116886, 0.5054539777376014, 0.05487231725071003], "C3": [0.08272522975669545, 0.3362113177196227, 0.581063452523682]},
    "Q3": {"C1": [0.1908126065623114, 0.13014438960155714, 0.6790430038361315], "C2": [0.4588484333521725, 0.492319969017205, 0.04883159763062254], "C3": [0.4888409844440297, 0.2712922071260089, 0.23986680842996144]},
    "Q4": {"C1": [0.46424936835477865, 0.3316029858304472, 0.20414764581477407], "C2": [0.4159408661768352, 0.34360425057848804, 0.24045488324467673], "C3": [0.08957839477602306, 0.4514493343192668, 0.4589722709047102]},
}


def synthetic_views() -> dict[str, Any]:
    planner = []
    router = []
    for task_index in range(4):
        slot = f"T{task_index + 1}"
        planner.append({"slot": slot, "description": f"Synthetic task {task_index + 1}", "shared_facts": [{"id": "S1", "text": "Shared synthetic fact"}], "options": [{"id": f"O{i}", "text": f"Synthetic option {i}"} for i in range(1, 4)]})
        router.append({"slot": slot, "description": f"Synthetic task {task_index + 1}", "private_facts": [{"id": f"F{i}", "text": f"Synthetic private fact {i}"} for i in range(1, 5)]})
    return {"planner": planner, "router": router}


def root_response() -> str:
    return json.dumps({"prior": [{"option_id": f"O{i + 1}", "probability": value} for i, value in enumerate(PRIOR)], "queries": [{"query_id": query_id, "request": f"Request synthetic evidence {index}", "target_dimension": f"synthetic dimension {index}", "channels": [{"channel_id": channel_id, "description": f"{query_id} synthetic pattern {c}"} for c, channel_id in enumerate(("C1", "C2", "C3"), 1)], "likelihoods": [{"option_id": f"O{o + 1}", "channels": [{"channel_id": channel_id, "probability": probability} for channel_id, probability in zip(("C1", "C2", "C3"), MATRICES[query_id][o], strict=True)]} for o in range(3)]} for index, query_id in enumerate(("Q1", "Q2", "Q3", "Q4"), 1)]})


def refresh_response() -> str:
    return json.dumps({"branches": [{"query_id": query_id, "channel_id": channel_id, "belief": [{"option_id": f"O{i + 1}", "probability": probability} for i, probability in enumerate(REFRESHED[query_id][channel_id])]} for query_id in ("Q1", "Q2", "Q3", "Q4") for channel_id in ("C1", "C2", "C3")]})


def routing_response() -> str:
    return json.dumps({"tasks": [{"slot": slot, "mappings": [{"query_id": query_id, "fact_id": f"F{index}", "channel_id": f"C{min(index, 3)}"} for index, query_id in enumerate(("Q1", "Q2", "Q3", "Q4"), 1)]} for slot in ("T1", "T2", "T3", "T4")]})


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_complete_seeded_messages_batched_structured(self, batch_messages: Sequence[list[dict[str, str]]], seeds: Sequence[int], *, temperature: float, response_formats: Sequence[dict[str, Any]], max_new_tokens: int) -> list[str]:
        assert len(batch_messages) == len(seeds) == len(response_formats)
        assert temperature == 0.0
        assert max_new_tokens == serving.MAX_TOKENS
        for messages, seed, response_format in zip(batch_messages, seeds, response_formats, strict=True):
            self.calls.append({"messages": messages, "seed": seed, "response_format": response_format})
        if len(self.calls) <= 4:
            return [root_response() for _ in batch_messages]
        if len(self.calls) <= 8:
            return [refresh_response() for _ in batch_messages]
        return [routing_response() for _ in batch_messages]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"adapter_requests": len(self.calls), "http_attempts": len(self.calls), "retry_count": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "forced_final_requests": 0, "adapter_cost_usd": 0.003}


def test_real_adapter_and_exact_payload_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    adapter = serving.build_adapter("payload-rehearsal", tmp_path)
    assert adapter.max_retries == 0
    assert adapter.backoff_seconds == 1.0
    assert adapter.reasoning_enabled is False
    assert adapter.max_tokens == 8192
    adapter._request_seed.value = serving.MODEL_SEEDS[0]
    try:
        payload = adapter._payload(serving.root_messages(synthetic_views()["planner"][0]), 0.0, 1, serving.MAX_TOKENS, response_format=serving.root_response_format(("O1", "O2", "O3")))
    finally:
        del adapter._request_seed.value
    assert payload["model"] == serving.MODEL_ID
    assert payload["seed"] == serving.MODEL_SEEDS[0]
    assert payload["max_tokens"] == 8192
    assert payload["reasoning"] == {"enabled": False, "exclude": True}
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_full_ten_response_label_free_transaction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(serving, "load_views", lambda source_path: synthetic_views())
    adapter = FakeAdapter()
    result = serving.run_serving(source_path=tmp_path / "unused", output_dir=tmp_path / "run", adapter=adapter)
    assert result["status"] == "serving_pass"
    assert result["authorizes"] == "endpoint_only"
    assert all(result["gates"].values())
    assert result["semantic"]["changed_first_query_tasks"] == 4
    assert all(task["myopic_first_query_id"] == "Q2" for task in result["semantic"]["tasks"])
    assert all(task["dynamic"]["first_query_id"] == "Q3" for task in result["semantic"]["tasks"])
    assert [call["seed"] for call in adapter.calls] == list(serving.MODEL_SEEDS)
    assert all(call["response_format"]["json_schema"]["strict"] for call in adapter.calls)
    prompts = json.dumps([call["messages"] for call in adapter.calls])
    assert "correct_answer" not in prompts
    assert result["registered_answers_opened"] is False
    raw = json.loads((tmp_path / "run/private/RAW_RESPONSES.json").read_text())
    assert {key: len(value) for key, value in raw.items()} == {"roots": 4, "refreshes": 4, "router": 1, "auditor": 1}
    verification = verifier.verify(tmp_path / "run")
    assert verification["status"] == "verification_pass"
    assert verification["registered_answers_loaded"] is False


def test_independent_verifier_rejects_metric_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(serving, "load_views", lambda source_path: synthetic_views())
    run_dir = tmp_path / "run"
    serving.run_serving(
        source_path=tmp_path / "unused", output_dir=run_dir, adapter=FakeAdapter()
    )
    result_path = run_dir / "LABEL_FREE_RESULT.json"
    result = json.loads(result_path.read_text())
    result["semantic"]["tasks"][0]["dynamic"]["scores"]["Q1"] += 0.01
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="verification failed"):
        verifier.verify(run_dir)
