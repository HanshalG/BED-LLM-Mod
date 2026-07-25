from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.zendo_particle_multiset_confirmation import (
    EXPECTED_REQUESTS,
    TASKS,
    deranged_pathways,
    exact_sign_flip_pvalue,
    first_executable_positive_scene,
    pairwise_accuracy,
    root_only_messages,
    run_confirmation,
)
from scripts.zendo_path_dependent_belief_gate import validate_rule


def test_confirmation_request_count_is_frozen() -> None:
    assert len(TASKS) == 7
    assert EXPECTED_REQUESTS == 84


def test_first_executable_positive_scene_skips_oversized_case() -> None:
    cases = json.load(
        open(
            "external/doing-experiments-and-revising-rules/data/zendo_cases.json"
        )
    )
    index, scene = first_executable_positive_scene(cases[8])
    assert index == 1
    assert len(scene["blocks"]) <= 6


def test_derangement_preserves_roots_and_rotates_future_beliefs() -> None:
    pathways = [
        {
            "label": label,
            "root_scene": {"blocks": [label]},
            "branches": [{"source": label}],
        }
        for label in ("A", "B", "C", "D")
    ]
    deranged = deranged_pathways(pathways, shift=1)
    assert [row["root_scene"] for row in deranged] == [
        row["root_scene"] for row in pathways
    ]
    assert [row["branches"] for row in deranged] == [
        pathways[index]["branches"] for index in (1, 2, 3, 0)
    ]


def test_root_only_prompt_excludes_future_and_exact_eig() -> None:
    hypotheses = [
        {"rule_text": "rule", "rule": {"op": "any"}} for _ in range(4)
    ]
    roots = [{"blocks": []} for _ in range(4)]
    rows = [
        {
            "initial_probability_yes": 0.5,
            "initial_eig": 99.0,
        }
        for _ in range(4)
    ]
    prompt = root_only_messages(
        hypotheses=hypotheses,
        weights=[0.25] * 4,
        roots=roots,
        root_selection=rows,
    )[1]["content"]
    assert "99.0" not in prompt
    assert "continuation_scene" not in prompt
    assert "refreshed_belief" not in prompt


def test_pairwise_accuracy_and_exact_sign_flip() -> None:
    assert pairwise_accuracy([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert pairwise_accuracy([4, 3, 2, 1], [10, 20, 30, 40]) == 0.0
    assert pairwise_accuracy([1, 1, 1, 1], [10, 20, 30, 40]) == 0.5
    assert exact_sign_flip_pvalue([1.0] * 7) == pytest.approx(1 / 128)


def _hypothesis_response() -> str:
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


class _FakeAdapter:
    def __init__(self, response: str | None = None) -> None:
        self.requests = 0
        self.response = response or _hypothesis_response()

    def chat_complete_messages_batched(
        self,
        messages: list[list[dict[str, str]]],
        **_: object,
    ) -> list[str]:
        self.requests += len(messages)
        if len(messages) in {len(TASKS), len(TASKS) * 8}:
            return [self.response] * len(messages)
        return ['{"root_scores":[10,20,30,40]}'] * len(messages)

    def usage_snapshot(self) -> dict[str, object]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.01,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def test_full_confirmation_freezes_84_calls_before_truth(
    tmp_path: Path,
) -> None:
    adapter = _FakeAdapter()
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    payload = run_confirmation(
        config,
        source_dir=Path(
            "external/doing-experiments-and-revising-rules"
        ),
        raw_checkpoint_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )
    assert adapter.requests == EXPECTED_REQUESTS
    assert len(payload["tasks"]) == len(TASKS)
    assert payload["protocol"][
        "truth_hidden_until_all_populations_and_scores_frozen"
    ]
    assert payload["usage"]["physical_requests"] == EXPECTED_REQUESTS


def test_full_confirmation_filters_invalid_ast_samples_prospectively(
    tmp_path: Path,
) -> None:
    payload = json.loads(_hypothesis_response())
    payload["hypotheses"][1]["rule"] = {
        "op": "exists",
        "predicate": {
            "op": "attribute",
            "attribute": "color",
            "value": "large",
        },
    }
    adapter = _FakeAdapter(json.dumps(payload))
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    result = run_confirmation(
        config,
        source_dir=Path(
            "external/doing-experiments-and-revising-rules"
        ),
        raw_checkpoint_path=tmp_path / "filtered-raw.json",
        model_adapter=adapter,
        interface_version="test-filtered",
        filter_invalid_particles=True,
    )
    assert adapter.requests == EXPECTED_REQUESTS
    assert result["protocol"]["particle_validation"] == (
        "filter_invalid_ast_samples"
    )
    assert all(
        task["valid_particle_count_minimum"] == 11
        for task in result["tasks"]
    )
    assert all(
        task["invalid_particle_count"] == 9
        for task in result["tasks"]
    )
