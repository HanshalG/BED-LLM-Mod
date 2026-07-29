from __future__ import annotations

import json

from scripts import number_game_gptmini_planner_serving_smoke as smoke


class FakeAdapter:
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
            "adapter_prompt_tokens": 100,
            "adapter_completion_tokens": 200,
        }


def test_exact_ten_smoke_is_transport_only(tmp_path) -> None:
    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="test-gptmini-serving",
        adapter=FakeAdapter(),
    )

    assert result["status"] == "passed"
    assert result["protocol"]["model"] == "openai/gpt-5.4-mini"
    assert result["protocol"]["expected_requests"] == 10
    assert result["protocol"]["efficacy_used_for_authorization"] is False
    assert result["usage"]["adapter_requests"] == 10
    assert result["gates"]["all_pass"] is True
    assert (tmp_path / "private" / "RAW_RESPONSES.json").exists()
