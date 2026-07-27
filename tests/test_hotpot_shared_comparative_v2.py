import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.hotpot_shared_comparative_v2 import (
    _pairwise_accuracy,
    parse_final_lines,
    parse_hypothesis_lines,
    parse_myopic_order,
    parse_plan,
    plan_rank_messages,
    run_tasks,
    serving_tasks,
)


def test_parse_hypothesis_lines_requires_ordered_distinct_values() -> None:
    text = "\n".join(f"H{index}|hypothesis {index}" for index in range(1, 9))
    assert len(parse_hypothesis_lines(text)) == 8
    try:
        parse_hypothesis_lines(text.replace("H2|", "H3|", 1))
    except ValueError as exc:
        assert "prefix order" in str(exc)
    else:
        raise AssertionError("out-of-order hypothesis prefix should fail")


def test_parse_myopic_order_requires_complete_root_permutation() -> None:
    assert parse_myopic_order("ORDER|R3|R1|R4|R2") == [2, 0, 3, 1]
    try:
        parse_myopic_order("ORDER|R3|R1|R4|R4")
    except ValueError as exc:
        assert "permutation" in str(exc)
    else:
        raise AssertionError("duplicate root should fail")


def test_parse_plan_requires_order_and_one_followup_per_root() -> None:
    parsed = parse_plan(
        "ORDER|R2|R1|R4|R3\nR1|T4\nR2|T2\nR3|T9\nR4|T1",
        candidate_count=9,
    )
    assert parsed["root_order"] == [1, 0, 3, 2]
    assert parsed["followup_indices"] == {0: 3, 1: 1, 2: 8, 3: 0}


def test_parse_final_lines() -> None:
    assert parse_final_lines("ANSWER|Ada Lovelace\nCONFIDENCE|87") == {
        "answer": "Ada Lovelace",
        "confidence": 87,
    }


def test_pairwise_accuracy_uses_complete_root_order() -> None:
    assert _pairwise_accuracy([0, 1, 2, 3], [3, 2, 1, 0]) == (6, 6)
    assert _pairwise_accuracy([3, 2, 1, 0], [3, 2, 1, 0]) == (0, 6)


def test_plan_prompt_exposes_root_title_and_complete_path_objective() -> None:
    messages = plan_rank_messages(
        states=[[f"state {root}"] for root in range(4)],
        all_titles=["A", "B", "C", "D", "E"],
        root_context_indices=[0, 1, 2, 3],
    )
    payload = json.loads(messages[-1]["content"])
    assert [root["revealed_root_title"] for root in payload["root_plans"]] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert "complete two-article paths" in messages[0]["content"]
    assert "Do not rank on continuation quality alone" in messages[0]["content"]


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
                root = payload["revealed_article"]["title"]
                outputs.append(
                    "\n".join(
                        f"H{index}|{root} refreshed hypothesis {index}"
                        for index in range(1, 9)
                    )
                )
            elif "Rank four candidate first articles" in system:
                outputs.append("ORDER|R1|R2|R3|R4")
            elif "Compare four two-step retrieval plans" in system:
                candidate_counts = [
                    len(root["candidate_titles"])
                    for root in payload["root_plans"]
                ]
                assert min(candidate_counts) >= 1
                outputs.append(
                    "ORDER|R2|R1|R3|R4\n"
                    "R1|T1\nR2|T1\nR3|T1\nR4|T1"
                )
            elif "Answer using only the supplied articles" in system:
                outputs.append("ANSWER|dry run answer\nCONFIDENCE|50")
            else:
                raise AssertionError(f"unexpected dry-run prompt: {system}")
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


@pytest.mark.skipif(
    not Path("external/hotpotqa/distractor_validation.parquet").exists(),
    reason="frozen HotpotQA validation file is not installed",
)
def test_real_file_serving_dry_run_has_exact_delayed_endpoint_shape(
    tmp_path,
) -> None:
    tasks = serving_tasks(
        Path("external/hotpotqa/distractor_validation.parquet")
    )
    config = load_config(
        "configs/config_hotpot_shared_comparative_v2_openrouter.yaml"
    )
    adapter = _DryRunAdapter()
    result = run_tasks(
        config,
        tasks=tasks,
        stage="serving",
        raw_path=tmp_path / "raw.json",
        cohort_rows_materialized=1,
        model_adapter=adapter,
    )
    assert adapter.requests == 10
    assert result["usage"]["physical_requests"] == 10
    assert result["protocol"][
        "endpoint_hidden_until_all_model_outputs_frozen"
    ]
    assert len(result["records"]) == 1
    assert result["gates"]["all_pass"]
