from __future__ import annotations

import json
from pathlib import Path

from helpers import load_config
from scripts.hotpot_future_uplift_confirmation import (
    _pairwise_accuracy,
    _sign_flip_p,
    future_first_root,
    policy_selections,
    run_serving,
    title_bm25_order,
)


def _row(aligned_max: int, aligned_index: int = 0) -> dict[str, list[int]]:
    aligned = [0] * 9
    initial = [0] * 9
    shuffled = [0] * 9
    aligned[aligned_index] = aligned_max
    initial[(aligned_index + 1) % 9] = max(0, aligned_max - 10)
    shuffled[(aligned_index + 2) % 9] = max(0, aligned_max - 5)
    return {
        "aligned_scores": aligned,
        "initial_scores": initial,
        "shuffled_scores": shuffled,
    }


def test_future_first_reproduces_the_open_hotpot_smoke_choice() -> None:
    rows = [_row(value) for value in (56, 100, 100, 99)]
    assert future_first_root([98, 42, 0, 1], rows) == 1


def test_future_first_breaks_future_ties_with_immediate_then_order() -> None:
    rows = [_row(100) for _ in range(4)]
    assert future_first_root([20, 30, 30, 10], rows) == 1


def test_policy_controls_use_shared_rows_but_distinct_objectives() -> None:
    rows = [_row(value, index) for index, value in enumerate((56, 100, 90, 80))]
    policies = policy_selections(
        immediate_scores=[98, 42, 0, 1],
        continuation_rows=rows,
        random_seed=7,
    )
    assert policies["future_uplift"]["root_index"] == 1
    assert policies["myopic"]["root_index"] == 0
    assert policies["old_total"]["root_index"] == 0
    assert policies["future_uplift"]["followup_candidate_index"] == 1


def test_pairwise_accuracy_ignores_value_and_score_ties() -> None:
    assert _pairwise_accuracy([3, 2, 1], [3, 1, 2]) == (2, 3)
    assert _pairwise_accuracy([3, 3, 1], [3, 2, 1]) == (2, 2)
    assert _pairwise_accuracy([3, 2, 1], [2, 2, 1]) == (2, 2)


def test_exact_one_sided_sign_flip_probability() -> None:
    assert _sign_flip_p([1, 1, 1, 1]) == 0.0625
    assert _sign_flip_p([1, 1, 1, 1, 1]) == 0.03125
    assert _sign_flip_p([1, -1]) == 0.75
    assert _sign_flip_p([0, 0]) == 1.0


def test_title_bm25_order_is_deterministic() -> None:
    titles = ["red planet", "blue ocean", "planetary science", "plain"]
    first = title_bm25_order("red planet science", titles)
    second = title_bm25_order("red planet science", titles)
    assert first == second
    assert first[0] == 0


class _FakeServingModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat_complete_messages_batched(
        self,
        messages,
        temperature,
        block_size,
        max_new_tokens,
    ):
        del temperature, block_size, max_new_tokens
        self.calls += len(messages)
        if len(messages) == 1 and self.calls == 1:
            return [
                json.dumps(
                    {
                        **{
                            f"hypothesis_{index}": f"initial hypothesis {index}"
                            for index in range(1, 9)
                        },
                        "root_1_score": 98,
                        "root_2_score": 42,
                        "root_3_score": 0,
                        "root_4_score": 1,
                    }
                )
            ]
        if len(messages) == 4 and self.calls == 5:
            return [
                json.dumps(
                    {
                        f"hypothesis_{index}": (
                            f"branch {branch} hypothesis {index}"
                        )
                        for index in range(1, 9)
                    }
                )
                for branch in range(4)
            ]
        if len(messages) == 4 and self.calls == 9:
            responses = []
            for root in range(4):
                payload = {}
                for state_offset, state in enumerate(("a", "b", "c")):
                    for index in range(1, 10):
                        payload[
                            f"state_{state}_title_{index}_score"
                        ] = (root * 17 + state_offset * 11 + index * 7) % 101
                responses.append(json.dumps(payload))
            return responses
        if len(messages) == 1 and self.calls == 10:
            return ['{"answer":"Army and Navy","confidence":90}']
        raise AssertionError("unexpected fake serving stage")

    def usage_snapshot(self):
        return {
            "adapter_requests": self.calls,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.01,
            "http_attempts": self.calls,
            "retry_count": 0,
            "forced_exits": 0,
        }


def test_serving_stage_accepts_exact_fake_transport(tmp_path: Path) -> None:
    config = load_config(
        "configs/config_hotpot_causal_belief_smoke_openrouter.yaml"
    )
    payload = run_serving(
        config,
        validation_path=Path("/tmp/hotpotqa-distractor-validation.parquet"),
        raw_path=tmp_path / "raw.json",
        model_adapter=_FakeServingModel(),
    )
    assert payload["status"] == "passed"
    assert payload["gates"]["all_pass"] is True
    assert payload["usage"]["physical_requests"] == 10
