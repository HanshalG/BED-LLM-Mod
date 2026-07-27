import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.hotpot_shared_comparative_v3 import EXPOSED_CONFIRMATION_IDS
from scripts.hotpot_shared_comparative_v4 import (
    _run,
    myopic_rank_row_messages,
    parse_myopic_rank_rows,
    v4_serving_tasks,
)


def test_parse_myopic_rank_rows() -> None:
    assert parse_myopic_rank_rows(
        "R1|2\nR2|1\nR3|4\nR4|3"
    ) == [1, 0, 3, 2]


@pytest.mark.parametrize(
    "text",
    [
        "R1|2\nR2|1\nR3|4",
        "R1|2\nR2|2\nR3|4\nR4|3",
        "R2|2\nR1|1\nR3|4\nR4|3",
        "R1 2\nR2|1\nR3|4\nR4|3",
    ],
)
def test_parse_myopic_rank_rows_rejects_bad_shapes(text: str) -> None:
    with pytest.raises(ValueError):
        parse_myopic_rank_rows(text)


def test_myopic_prompt_has_no_standalone_order_codec() -> None:
    messages = myopic_rank_row_messages(
        [f"hypothesis {index}" for index in range(8)],
        ["A", "B", "C", "D"],
    )
    assert "R1|k, R2|k, R3|k, R4|k" in messages[0]["content"]
    assert "no header" in messages[0]["content"]
    payload = json.loads(messages[-1]["content"])
    assert [row["root_id"] for row in payload["root_titles"]] == [
        "R1",
        "R2",
        "R3",
        "R4",
    ]


class _DryRunAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, messages, **kwargs):
        del kwargs
        outputs = []
        for message_list in messages:
            self.requests += 1
            system = message_list[0]["content"]
            payload = json.loads(message_list[-1]["content"])
            if "Generate eight distinct" in system:
                outputs.append(
                    "\n".join(
                        f"H{index}|initial hypothesis {index}"
                        for index in range(1, 9)
                    )
                )
            elif "Refresh an open-world" in system:
                title = payload["revealed_article"]["title"]
                outputs.append(
                    "\n".join(
                        f"H{index}|{title} refreshed hypothesis {index}"
                        for index in range(1, 9)
                    )
                )
            elif "Rank four candidate first articles" in system:
                outputs.append("R1|2\nR2|1\nR3|3\nR4|4")
            elif "Compare four complete two-article" in system:
                outputs.append(
                    "R1|T1|2\nR2|T1|1\nR3|T1|3\nR4|T1|4"
                )
            elif "Answer using only the supplied articles" in system:
                outputs.append("ANSWER|dry run answer\nCONFIDENCE|50")
            else:
                raise AssertionError(f"unexpected prompt: {system}")
        return outputs

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


TRAIN_SHARDS = (
    Path("external/hotpotqa/train/0000.parquet"),
    Path("external/hotpotqa/train/0001.parquet"),
)


@pytest.mark.skipif(
    not all(path.exists() for path in TRAIN_SHARDS),
    reason="frozen HotpotQA train shards are not installed",
)
def test_v4_real_file_dry_run_uses_second_exposed_row(tmp_path) -> None:
    tasks, cohort_rows = v4_serving_tasks(TRAIN_SHARDS)
    assert [str(task["row"]["id"]) for task in tasks] == [
        EXPOSED_CONFIRMATION_IDS[1]
    ]
    config = load_config(
        "configs/config_hotpot_shared_comparative_v4_openrouter.yaml"
    )
    adapter = _DryRunAdapter()
    result = _run(
        config,
        tasks=tasks,
        stage="serving",
        raw_path=tmp_path / "raw.json",
        cohort_rows_materialized=cohort_rows,
        model_adapter=adapter,
    )
    assert adapter.requests == 10
    assert result["usage"]["physical_requests"] == 10
    assert result["protocol"]["interface_version"].endswith("v4-1")
    assert result["gates"]["all_pass"]
