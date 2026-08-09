from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from scripts import bongard_openworld_siglip_classical_baseline as baseline
from scripts import bongard_openworld_vlm_bed as bed


class BombLabels(Mapping[str, bool]):
    def __getitem__(self, key: str) -> bool:
        raise AssertionError("a real label was accessed")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("real labels were enumerated")

    def __len__(self) -> int:
        raise AssertionError("real labels were counted")


def _task() -> tuple[bed.VisualTask, dict[str, np.ndarray]]:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    rng = np.random.default_rng(20260808)
    values = rng.normal(size=(14, 12))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    task = bed.VisualTask(
        task_id="task-siglip-test",
        image_ids=image_ids,
        initial_history=(
            (image_ids[0], True),
            (image_ids[1], True),
            (image_ids[2], False),
            (image_ids[3], False),
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={},
        actual_labels=BombLabels(),
    )
    return task, dict(zip(image_ids, values, strict=True))


def test_counterfactual_plans_are_complete_without_real_labels() -> None:
    task, embeddings = _task()
    for policy in baseline.POLICIES:
        plan = baseline.policy_plan(task, embeddings, policy, scale=2.0)
        assert plan["policy"] == policy
        assert plan["first_image_id"] in task.candidate_ids
        assert set(plan["first_scores"]) == set(task.candidate_ids)
        assert set(plan["branches"]) == {"negative", "positive"}
        for branch in plan["branches"].values():
            assert branch["second_image_id"] in task.candidate_ids
            assert branch["second_image_id"] != plan["first_image_id"]
            assert set(branch["outcomes"]) == {"negative", "positive"}


def test_shared_planner_hash_is_frozen() -> None:
    baseline.verify_shared_planner()


def test_signed_scale_grid_can_calibrate_inverted_similarity() -> None:
    tasks = {"mechanics": [], "development": [], "confirmation": []}
    task, embeddings = _task()
    tasks["mechanics"] = [task]
    plans = baseline.build_plans(
        tasks,
        {task.task_id: embeddings},
    )
    assert plans["calibration"]["selected_scale"] in baseline.SCALE_GRID
    assert min(baseline.SCALE_GRID) < 0.0
    assert 0.0 in baseline.SCALE_GRID


def test_frozen_plan_interpreter_uses_realized_path_and_endpoint_labels() -> None:
    task, embeddings = _task()
    labels = {
        **dict(task.initial_history),
        **{
            image_id: index % 2 == 0
            for index, image_id in enumerate(task.candidate_ids)
        },
        task.endpoint_ids[0]: True,
        task.endpoint_ids[1]: False,
    }
    opened = bed.VisualTask(
        task_id=task.task_id,
        image_ids=task.image_ids,
        initial_history=task.initial_history,
        candidate_ids=task.candidate_ids,
        endpoint_ids=task.endpoint_ids,
        image_bytes=task.image_bytes,
        actual_labels=labels,
    )
    plans = {
        policy: baseline.policy_plan(opened, embeddings, policy, scale=2.0)
        for policy in baseline.POLICIES
    }
    result = baseline.evaluate_opened_partition(
        [opened], [{"task_id": opened.task_id, "policies": plans}]
    )
    assert set(result["pooled"]) == set(baseline.POLICIES)
    assert all(
        0.0 <= result["pooled"][policy]["mean_brier"] <= 1.0
        for policy in baseline.POLICIES
    )
    assert len(
        result["paired_depth2_minus_myopic"]["mean_brier"]["values"]
    ) == 1
