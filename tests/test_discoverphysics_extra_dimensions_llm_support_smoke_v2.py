from __future__ import annotations

from pathlib import Path

from helpers import Config
from scripts.discoverphysics_extra_dimensions_llm_support_smoke import (
    NUM_HYPOTHESES,
)
from scripts.discoverphysics_extra_dimensions_llm_support_smoke_v2 import (
    parse_support_records,
    run_smoke,
)


def support_records(offset: float = 0.0) -> str:
    lines = []
    for index in range(NUM_HYPOTHESES):
        values = (
            f"H{index + 1:02d}",
            f"law_{index}_{str(offset).replace('.', '_')}",
            str(index + 1),
            f"{-4.8 + 0.25 * index + offset:.6f}",
            f"{0.6 + 0.14 * index:.6f}",
            f"{0.8 + 0.15 * index:.6f}",
            f"{0.4 + 0.55 * index:.6f}",
            f"{0.2 + 0.05 * (index % 5):.6f}",
            f"{0.02 * index:.6f}",
        )
        lines.append("|".join(values))
    return "\n".join(lines)


class FakeRecordModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        max_new_tokens,
    ):
        del temperature, block_size, max_new_tokens
        start = self.requests
        self.requests += len(batch_messages)
        return [
            support_records(0.001 * (start + index))
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


def test_record_parser_is_strict_and_compiles() -> None:
    parsed = parse_support_records(support_records())

    assert len(parsed) == NUM_HYPOTHESES
    assert abs(sum(item["probability"] for item in parsed) - 1.0) < 1e-12

    malformed = support_records().replace("H03|", "H04|", 1)
    try:
        parse_support_records(malformed)
    except ValueError as exc:
        assert "H03" in str(exc)
    else:
        raise AssertionError("out-of-order record IDs must be rejected")


def test_record_smoke_uses_exactly_ten_requests_and_checkpoints(
    tmp_path: Path,
) -> None:
    model = FakeRecordModel()
    result = run_smoke(
        Config(),
        raw_path=tmp_path / "raw.json",
        model=model,
    )

    assert model.requests == 10
    assert result["usage"]["physical_requests"] == 10
    assert result["gates"]["exact_10_physical_requests"]
    assert result["gates"]["zero_reasoning_tokens"]
    assert result["protocol"]["response_format"] == "chat_fixed_pipe_records"
    assert (tmp_path / "raw.json").exists()
