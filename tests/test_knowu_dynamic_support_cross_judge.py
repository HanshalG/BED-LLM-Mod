from __future__ import annotations

from scripts import knowu_dynamic_support_cross_judge as cross
from scripts import knowu_dynamic_support_mechanics as mechanics


def _judgment(scores):
    indices = tuple(1 if score >= 70 else 0 for score in scores)
    return mechanics.TruthJudgment(
        best_indices=indices,
        best_scores=tuple(scores),
        reasons=("reason",) * 5,
    )


def test_analyze_judgments_confirms_frozen_target_pattern():
    records = []
    judgments = []
    for index in range(6):
        target = index == 2
        scores = (68, 78, 66, 82, 69) if target else (90,) * 5
        presence = tuple(score >= 70 for score in scores)
        records.append(
            {
                "world_id": f"T{1 + index // 3}W{1 + index % 3}",
                "task_id": "SyntheticTask",
                "original_scores": scores,
                "original_presence": presence,
            }
        )
        judgments.append(_judgment(scores))
    usage = {
        "physical_requests": 6,
        "http_attempts": 6,
        "retry_count": 0,
        "reasoning_tokens": 0,
        "forced_exits": 0,
        "adapter_cost_usd": 0.01,
    }

    result = cross.analyze_judgments(records, judgments, usage)

    assert result["status"] == "passed"
    assert result["summary"]["presence_agreement"] == 1.0
    assert result["summary"]["target_max_os_or_platform_gain"] == 14


def test_analyze_judgments_rejects_target_without_entry():
    records = []
    judgments = []
    for index in range(6):
        scores = (68, 68, 66, 69, 69) if index == 2 else (90,) * 5
        presence = tuple(score >= 70 for score in scores)
        records.append(
            {
                "world_id": f"T{1 + index // 3}W{1 + index % 3}",
                "task_id": "SyntheticTask",
                "original_scores": scores,
                "original_presence": presence,
            }
        )
        judgments.append(_judgment(scores))
    usage = {
        "physical_requests": 6,
        "http_attempts": 6,
        "retry_count": 0,
        "reasoning_tokens": 0,
        "forced_exits": 0,
        "adapter_cost_usd": 0.01,
    }

    result = cross.analyze_judgments(records, judgments, usage)

    assert result["status"] == "gate_failed"
    assert not result["gates"]["target_os_or_platform_root_recovers_truth"]


def test_deterministic_cross_judge_serving_has_exact_ten_requests(tmp_path):
    model = mechanics.DeterministicFixtureModel()

    result = cross.run_serving_gate(
        model, raw_path=tmp_path / "raw.json"
    )

    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
