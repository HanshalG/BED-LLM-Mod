from __future__ import annotations

import json

import pytest

from scripts import number_game_qwen_first_link_confirmation64 as confirm
from scripts import number_game_qwen_first_link_serving_smoke_v2 as smoke


class FakeSmokeAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        del temperature, block_size, response_format, max_new_tokens
        self.requests += len(batch_messages)
        responses = []
        for case in smoke.serving_cases():
            hypotheses = []
            for index in range(24):
                expression = f"(n + {index}) % {index + 2} == 0"
                for number, label in case["observations"]:
                    operator = "or" if label else "and"
                    comparison = "==" if label else "!="
                    expression = (
                        f"({expression}) {operator} n {comparison} {number}"
                    )
                hypotheses.append(
                    {
                        "name": f"synthetic_{index}",
                        "expression": expression,
                    }
                )
            responses.append(json.dumps({"hypotheses": hypotheses}))
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.02,
        }


def test_fresh_seed_and_budget_constants() -> None:
    assert len(confirm.TREE_SEEDS) == 64
    assert len(confirm.TARGET_SEEDS) == 64
    assert confirm.EXPECTED_REQUESTS == 3_712
    assert confirm.BOOTSTRAP_SAMPLES == 20_000
    assert smoke.MODEL_ID == "qwen/qwen3.7-plus"
    assert smoke.MODEL_SEED == 60_001


def test_v2_smoke_gates_linked_retained_supports(tmp_path) -> None:
    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="test-qwen-first-link-v2",
        adapter=FakeSmokeAdapter(),
    )

    assert result["status"] == "passed"
    assert result["protocol"]["support_update"] == "retained_rejuvenation"
    assert min(result["generated_valid_counts"][2:]) >= 4
    assert min(result["merged_first_valid_counts"]) >= 8
    assert min(result["merged_second_valid_counts"]) >= 4


def test_smoke_validator_requires_exact_linked_interface(tmp_path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": {
                    "interface_version": smoke.INTERFACE_VERSION,
                    "model": smoke.MODEL_ID,
                    "expected_requests": 10,
                    "efficacy_used_for_authorization": False,
                },
                "gates": {"all_pass": True},
            }
        )
    )
    assert confirm.validate_smoke_result(path)["status"] == "passed"

    payload = json.loads(path.read_text())
    payload["protocol"]["model"] = "wrong/model"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model changed"):
        confirm.validate_smoke_result(path)


def test_first_link_summary_uses_selected_root_pair(monkeypatch) -> None:
    monkeypatch.setattr(confirm, "BOOTSTRAP_SAMPLES", 100)
    trees = []
    for index in range(12):
        predicted = 0.01 + index * 0.001
        realized = 0.02 + index * 0.002
        trees.append(
            {
                "tree_seed": index,
                "selection": {
                    "crossfit_depth_three_root": 1,
                    "myopic_root": 2,
                    "crossfit_depth_three_brier": {
                        "1": 0.1,
                        "2": 0.1 + predicted,
                    },
                },
                "per_root_endpoint_brier": {
                    "1": 0.2,
                    "2": 0.2 + realized,
                },
            }
        )

    summary = confirm.summarize_first_link(trees)

    assert summary["root_differences"] == 12
    assert summary["wins"] == 12
    assert summary["losses"] == 0
    assert summary["score_to_realized_advantage_spearman"] == pytest.approx(
        1.0
    )
