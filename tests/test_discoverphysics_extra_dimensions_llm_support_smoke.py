from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helpers import Config
from scripts.discoverphysics_extra_dimensions_llm_support_smoke import (
    NUM_HYPOTHESES,
    force_features,
    parse_support,
    response_format,
    run_smoke,
    transformed_means,
)


def support_text(offset: float = 0.0) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {
                    "label": f"law_{index}_{offset:g}",
                    "weight": index + 1,
                    "log_amplitude": -4.8 + 0.25 * index + offset,
                    "long_exponent": 0.6 + 0.14 * index,
                    "short_exponent": 0.8 + 0.15 * index,
                    "transition_radius": 0.4 + 0.55 * index,
                    "transition_width": 0.2 + 0.05 * (index % 5),
                    "screening_rate": 0.02 * index,
                }
                for index in range(NUM_HYPOTHESES)
            ]
        }
    )


class FakeStructuredModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens,
    ):
        del temperature, block_size, response_format, max_new_tokens
        start = self.requests
        self.requests += len(batch_messages)
        return [
            support_text(0.001 * (start + index))
            for index in range(len(batch_messages))
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.1,
            "adapter_prompt_tokens": 1000,
            "adapter_completion_tokens": 1000,
        }


def test_schema_is_strict_and_support_compiles() -> None:
    schema = response_format()["json_schema"]
    assert schema["strict"]
    assert not schema["schema"]["additionalProperties"]
    hypotheses = parse_support(support_text())

    assert len(hypotheses) == NUM_HYPOTHESES
    assert np.isclose(
        sum(item["probability"] for item in hypotheses),
        1.0,
    )
    assert force_features(hypotheses).shape == (NUM_HYPOTHESES, 96)
    assert transformed_means(hypotheses).shape == (4, NUM_HYPOTHESES)


def test_smoke_uses_exactly_ten_structured_requests(tmp_path: Path) -> None:
    model = FakeStructuredModel()
    result = run_smoke(
        Config(),
        raw_path=tmp_path / "raw.json",
        model=model,
    )

    assert model.requests == 10
    assert result["usage"]["physical_requests"] == 10
    assert result["gates"]["exact_10_physical_requests"]
    assert result["gates"]["zero_reasoning_tokens"]
    assert (tmp_path / "raw.json").exists()
