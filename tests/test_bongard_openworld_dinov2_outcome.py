from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts import bongard_openworld_dinov2_outcome as outcome
from scripts import bongard_openworld_vlm_bed as bed


def test_frozen_dino_outcome_inputs_verify() -> None:
    observed = outcome.verify_frozen_inputs()
    assert observed["protocol"] == outcome.PROTOCOL_SHA256
    assert observed["plans"] == outcome.PLANS_SHA256


def test_failed_authorization_cannot_load_opened_labels(
    tmp_path: Path,
) -> None:
    calls = []

    def reject(**kwargs):
        calls.append("authorize")
        raise RuntimeError("invalid predecessor")

    def bomb(stage: str):
        calls.append("labels")
        raise AssertionError("opened labels loaded before authorization")

    with pytest.raises(RuntimeError, match="invalid predecessor"):
        outcome.run_outcome(
            stage="mechanics",
            result_path=tmp_path / "missing-result.json",
            output_path=tmp_path / "out.json",
            wrapper_result=tmp_path / "missing-wrapper.json",
            authorizer=reject,
            task_loader=bomb,
        )
    assert calls == ["authorize"]
    assert not (tmp_path / "out.json").exists()


def test_synthetic_opened_stage_pairs_luna_and_dino_exactly() -> None:
    sealed = bed.load_validation_partition_tasks(
        "mechanics", include_endpoint_labels=False
    )
    tasks = []
    trees = []
    for task_index, task in enumerate(sealed):
        labels = dict(task.initial_history)
        labels.update(
            {
                image_id: (image_index + task_index) % 2 == 0
                for image_index, image_id in enumerate(
                    (*task.candidate_ids, *task.endpoint_ids)
                )
            }
        )
        tasks.append(replace(task, actual_labels=labels))
        policies = {}
        for policy_index, policy in enumerate(
            ("myopic_width", "dynamic_depth2")
        ):
            policies[policy] = {
                "endpoint": {
                    "mean_brier": 0.20 + 0.01 * policy_index,
                    "mean_log_loss": 0.60 + 0.01 * policy_index,
                    "accuracy": 0.50,
                    "mean_truth_probability": 0.55 - 0.01 * policy_index,
                }
            }
        trees.append({"task_id": task.task_id, "policies": policies})
    result = outcome.build_outcome(
        stage="mechanics",
        stage_result={"status": "mechanics_pass", "trees": trees},
        tasks=tasks,
        plan_rows=outcome.load_plan_rows("mechanics"),
        authorization={"verified": True},
        frozen_inputs=outcome.verify_frozen_inputs(),
    )
    assert result["status"] == "classical_comparator_complete"
    assert result["all_gates_pass"] is True
    assert set(result["paired_luna_minus_dino"]) == {
        "luna_myopic_minus_dinov2_myopic",
        "luna_dynamic_minus_dinov2_depth2",
    }
    assert all(
        summary["bootstrap_draws"] == 20_000
        for comparison in result["paired_luna_minus_dino"].values()
        for summary in comparison.values()
    )
