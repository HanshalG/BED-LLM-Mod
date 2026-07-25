from __future__ import annotations

import json
from pathlib import Path

from helpers import load_config
from scripts.zendo_structured_output_serving_smoke import (
    EXPECTED_REQUESTS,
    TASKS,
    run_smoke,
)
from scripts.zendo_path_dependent_belief_gate import validate_rule


def _response() -> str:
    rules = [
        {
            "op": "exists",
            "predicate": {
                "op": "attribute",
                "attribute": attribute,
                "value": value,
            },
        }
        for attribute, values in (
            ("color", ("blue", "red", "green")),
            ("size", ("small", "medium", "large")),
            ("orientation", ("upright", "left", "right")),
        )
        for value in values
    ]
    rules.extend(
        {
            "op": "count",
            "predicate": {"op": "any"},
            "comparison": "eq",
            "value": value,
        }
        for value in range(1, 4)
    )
    return json.dumps(
        {
            "hypotheses": [
                {
                    "id": f"H{index:02d}",
                    "rule_text": f"rule {index}",
                    "rule": validate_rule(rule),
                }
                for index, rule in enumerate(rules, start=1)
            ]
        }
    )


class _FakeStructuredAdapter:
    def __init__(self) -> None:
        self.requests = 0
        self.response_format = None

    def chat_complete_messages_batched_structured(
        self,
        messages,
        *,
        response_format,
        **_kwargs,
    ):
        self.requests += len(messages)
        self.response_format = response_format
        return [_response()] * len(messages)

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.01,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def test_structured_serving_smoke_uses_exact_three_calls(
    tmp_path: Path,
) -> None:
    adapter = _FakeStructuredAdapter()
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    payload = run_smoke(
        config,
        source_dir=Path(
            "external/doing-experiments-and-revising-rules"
        ),
        raw_checkpoint_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )
    assert len(TASKS) == EXPECTED_REQUESTS == adapter.requests == 3
    assert adapter.response_format["type"] == "json_schema"
    assert payload["status"] == "passed"
    assert payload["protocol"]["scientific_endpoints_evaluated"] is False
