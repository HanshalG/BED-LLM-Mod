from __future__ import annotations

import json

from scripts import number_game_gptmini_planner_serving_smoke_v2 as smoke


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
        }


def test_v2_smoke_uses_linked_retained_supports(tmp_path) -> None:
    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="test-gptmini-serving-v2",
        adapter=FakeAdapter(),
    )

    assert result["status"] == "passed"
    assert result["protocol"]["support_update"] == "retained_rejuvenation"
    assert result["usage"]["adapter_requests"] == 10
    assert min(result["generated_valid_counts"][2:]) >= 4
    assert min(result["merged_first_valid_counts"]) >= 8
    assert min(result["merged_second_valid_counts"]) >= 4
    assert result["gates"]["all_pass"] is True


def test_verified_prompt_requires_literal_substitution() -> None:
    prompt = smoke.verified_history_messages(
        ((10, True), (20, False))
    )[1]["content"]

    assert "Literally substitute every observed integer" in prompt
    assert "n=10: the expression MUST evaluate to True" in prompt
    assert "n=20: the expression MUST evaluate to False" in prompt
