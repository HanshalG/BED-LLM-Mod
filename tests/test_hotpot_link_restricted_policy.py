import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.hotpot_link_restricted_policy import (
    link_plan_messages,
    load_tasks,
    oracle_root_values,
    parse_link_plan,
    run_tasks,
)


def test_parse_link_plan_accepts_local_actions_and_stop() -> None:
    parsed = parse_link_plan(
        "R1|S|3\nR2|T1|1\nR3|T2|2\nR4|S|4",
        candidate_counts=[0, 1, 2, 0],
    )
    assert parsed["root_order"] == [1, 2, 0, 3]
    assert parsed["followup_indices"] == {
        0: None,
        1: 0,
        2: 1,
        3: None,
    }


def test_parse_link_plan_rejects_unavailable_action() -> None:
    with pytest.raises(ValueError):
        parse_link_plan(
            "R1|T1|3\nR2|T1|1\nR3|S|2\nR4|S|4",
            candidate_counts=[0, 1, 0, 0],
        )


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
            elif "Compare four complete retrieval paths" in system:
                lines = []
                for index, root in enumerate(payload["root_paths"], start=1):
                    actions = {
                        action["action_id"]
                        for action in root["available_next_actions"]
                    }
                    action = "T1" if "T1" in actions else "S"
                    lines.append(f"R{index}|{action}|{index}")
                outputs.append("\n".join(lines))
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
def test_real_file_link_restricted_serving_rehearsal(tmp_path) -> None:
    tasks, cohort_rows = load_tasks(
        TRAIN_SHARDS, split_name="mechanics", count=1
    )
    assert len(tasks) == 1
    assert oracle_root_values(tasks[0]).count(2) == 1
    messages = link_plan_messages(
        states=[[f"state {index}"] for index in range(4)],
        task=tasks[0],
    )
    payload = json.loads(messages[-1]["content"])
    assert all(
        root["available_next_actions"][0]["action_id"] == "S"
        for root in payload["root_paths"]
    )
    config = load_config(
        "configs/config_hotpot_link_restricted_policy_openrouter.yaml"
    )
    adapter = _DryRunAdapter()
    result = run_tasks(
        config,
        tasks=tasks,
        stage="serving",
        raw_path=tmp_path / "raw.json",
        cohort_rows_materialized=cohort_rows,
        model_adapter=adapter,
    )
    assert adapter.requests == 10
    assert result["gates"]["all_pass"]
    assert result["protocol"]["followup_actions_are_paragraph_links"]
