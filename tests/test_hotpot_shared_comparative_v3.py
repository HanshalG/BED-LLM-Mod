import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.hotpot_shared_comparative_v3 import (
    EXPOSED_CONFIRMATION_IDS,
    _run,
    exposed_serving_tasks,
    parse_rank_rows,
    rank_row_messages,
)


def test_parse_rank_rows_requires_numeric_root_order_and_rank_permutation() -> None:
    parsed = parse_rank_rows(
        "R1|T3|2\nR2|T1|1\nR3|T9|4\nR4|T2|3",
        candidate_count=9,
    )
    assert parsed["root_order"] == [1, 0, 3, 2]
    assert parsed["followup_indices"] == {0: 2, 1: 0, 2: 8, 3: 1}


@pytest.mark.parametrize(
    "text",
    [
        "R1|T3|2\nR2|T1|1\nR3|T9|4",
        "R1|T3|2\nR2|T1|2\nR3|T9|4\nR4|T2|3",
        "R2|T3|2\nR1|T1|1\nR3|T9|4\nR4|T2|3",
        "R1 T3 2\nR2|T1|1\nR3|T9|4\nR4|T2|3",
    ],
)
def test_parse_rank_rows_rejects_incomplete_or_noncanonical_output(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        parse_rank_rows(text, candidate_count=9)


def test_rank_row_prompt_preserves_complete_path_objective() -> None:
    messages = rank_row_messages(
        states=[[f"state {root}"] for root in range(4)],
        all_titles=["A", "B", "C", "D", "E"],
        root_context_indices=[0, 1, 2, 3],
    )
    payload = json.loads(messages[-1]["content"])
    assert [root["revealed_root_title"] for root in payload["root_paths"]] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert "complete two-article retrieval paths" in messages[0]["content"]
    assert "Do not rank continuation quality alone" in messages[0]["content"]


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
                outputs.append("ORDER|R1|R2|R3|R4")
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
def test_exposed_serving_row_and_full_real_file_dry_run(tmp_path) -> None:
    tasks, cohort_rows = exposed_serving_tasks(TRAIN_SHARDS)
    assert cohort_rows == 500
    assert [str(task["row"]["id"]) for task in tasks] == [
        EXPOSED_CONFIRMATION_IDS[0]
    ]
    config = load_config(
        "configs/config_hotpot_shared_comparative_v3_openrouter.yaml"
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
    assert result["protocol"]["interface_version"].endswith("v3-1")
    assert result["gates"]["all_pass"]
