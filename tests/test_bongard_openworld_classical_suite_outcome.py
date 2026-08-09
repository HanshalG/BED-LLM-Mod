from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import bongard_openworld_classical_suite_outcome as outcome
from scripts import bongard_openworld_vlm_bed as bed


def test_failed_authorization_cannot_load_opened_labels(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(outcome, "verify_frozen_inputs", lambda: {})
    monkeypatch.setattr(outcome.dino_outcome, "load_plan_rows", lambda stage: [])
    monkeypatch.setattr(outcome, "load_siglip_plan_rows", lambda stage: [])

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


def test_synthetic_opened_stage_reports_both_classical_encoders() -> None:
    sealed = bed.load_validation_partition_tasks(
        "mechanics", include_endpoint_labels=False
    )
    dino_rows = outcome.dino_outcome.load_plan_rows("mechanics")
    siglip_rows = []
    for row in dino_rows:
        converted = copy.deepcopy(row)
        converted["policies"] = {
            "siglip_myopic": copy.deepcopy(row["policies"]["dinov2_myopic"]),
            "siglip_depth2": copy.deepcopy(row["policies"]["dinov2_depth2"]),
        }
        converted["policies"]["siglip_myopic"]["policy"] = "siglip_myopic"
        converted["policies"]["siglip_depth2"]["policy"] = "siglip_depth2"
        siglip_rows.append(converted)

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
        dino_plan_rows=dino_rows,
        siglip_plan_rows=siglip_rows,
        authorization={"verified": True},
        frozen_inputs={
            "dino": outcome.dino_outcome.verify_frozen_inputs(),
            "siglip": {"synthetic": "binding"},
        },
    )
    assert result["status"] == "classical_suite_complete"
    assert result["all_gates_pass"] is True
    assert set(result["dino"]["pooled"]) == {
        "dinov2_myopic",
        "dinov2_depth2",
    }
    assert set(result["siglip"]["pooled"]) == {
        "siglip_myopic",
        "siglip_depth2",
    }
    assert set(result["paired_luna_minus_siglip"]) == {
        "luna_myopic_minus_siglip_myopic",
        "luna_dynamic_minus_siglip_depth2",
    }
    assert all(
        summary["bootstrap_draws"] == 20_000
        for comparison in result["paired_luna_minus_siglip"].values()
        for summary in comparison.values()
    )
