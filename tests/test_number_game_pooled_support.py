from __future__ import annotations

import json

from scripts import number_game_qwen_pooled_support_serving_smoke as smoke
from scripts.number_game_pooled_support import (
    PooledStructuredAdapter,
    encode_pooled_responses,
    parse_pooled_proposals,
)


def hypotheses_for(observations=(), *, offset=0):
    hypotheses = []
    for index in range(24):
        modulus = index + 2 + offset
        expression = f"(n + {index}) % {modulus} == 0"
        for number, label in observations:
            operator = "or" if label else "and"
            comparison = "==" if label else "!="
            expression = (
                f"({expression}) {operator} n {comparison} {number}"
            )
        hypotheses.append(
            {
                "name": f"synthetic_{offset}_{index}",
                "expression": expression,
            }
        )
    return hypotheses


class FakeAdapter:
    def __init__(self, offset):
        self.offset = offset
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
        cases = [
            smoke.serving_cases()[index]
            for index in smoke.HISTORY_INDICES
        ]
        return [
            json.dumps(
                {
                    "hypotheses": hypotheses_for(
                        case["observations"],
                        offset=self.offset,
                    )
                }
            )
            for case in cases
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.01,
        }


def test_parse_pooled_proposals_unions_extensions() -> None:
    response = encode_pooled_responses(
        [
            json.dumps({"hypotheses": hypotheses_for(offset=0)}),
            json.dumps({"hypotheses": hypotheses_for(offset=30)}),
        ]
    )

    support, diagnostic = parse_pooled_proposals(response)

    assert diagnostic["pool_size"] == 2
    assert diagnostic["draw_novel_contributions"][0] >= 16
    assert diagnostic["draw_novel_contributions"][1] >= 2
    assert len(support) > max(diagnostic["draw_valid_counts"])


def test_pooled_adapter_uses_both_independent_adapters(tmp_path) -> None:
    first = FakeAdapter(0)
    second = FakeAdapter(30)
    adapter = PooledStructuredAdapter([first, second])

    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="unused",
        adapter=adapter,
    )

    assert result["usage"]["adapter_requests"] == 10
    assert first.requests == 5
    assert second.requests == 5
