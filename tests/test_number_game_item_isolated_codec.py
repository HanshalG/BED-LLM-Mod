from __future__ import annotations

import json

import pytest

from scripts import number_game_item_isolated_codec as codec
from scripts import number_game_qwen_item_isolated_serving_smoke as smoke


def hypotheses_for(observations=()):
    hypotheses = []
    for index in range(24):
        expression = f"(n + {index}) % {index + 2} == 0"
        for number, label in observations:
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
    return hypotheses


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
        return [
            json.dumps({"hypotheses": hypotheses_for(case["observations"])})
            for case in smoke.serving_cases()
        ]

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


def test_item_isolated_parser_preserves_strict_json() -> None:
    response = json.dumps({"hypotheses": hypotheses_for()})

    support, diagnostic = codec.parse_proposals_item_isolated(response)

    assert diagnostic["codec_mode"] == "strict_json"
    assert diagnostic["complete_item_count"] == 24
    assert len(support) >= 16


def test_item_isolated_parser_salvages_complete_items_only() -> None:
    complete = hypotheses_for()[:7]
    response = (
        '{"hypotheses":['
        + ",".join(json.dumps(item) for item in complete)
        + ',{"name":"broken","expression":"n % 2'
    )

    support, diagnostic = codec.parse_proposals_item_isolated(response)

    assert diagnostic["codec_mode"] == "complete_item_salvage"
    assert diagnostic["complete_item_count"] == 7
    assert diagnostic["rejected"]["missing_or_incomplete"] == 17
    assert len(support) == 7


def test_item_isolated_parser_rejects_without_complete_items() -> None:
    with pytest.raises(ValueError, match="no complete hypothesis items"):
        codec.parse_proposals_item_isolated('{"hypotheses":[{"name":"')


def test_item_isolated_smoke_keeps_live_gate_strict(tmp_path) -> None:
    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="test-item-isolated-smoke",
        adapter=FakeAdapter(),
    )

    assert result["status"] == "passed"
    assert result["usage"]["adapter_requests"] == 10
    assert result["gates"]["all_ten_live_responses_are_strict_json"] is True
    assert result["protocol"]["semantic_repair"] is False
