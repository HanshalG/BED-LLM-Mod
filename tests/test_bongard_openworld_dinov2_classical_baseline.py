from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from scripts import bongard_openworld_dinov2_classical_baseline as baseline
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
        task_id="task-classical-test",
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
        assert plan["first_image_id"] in task.candidate_ids
        assert set(plan["first_scores"]) == set(task.candidate_ids)
        assert set(plan["branches"]) == {"negative", "positive"}
        for branch in plan["branches"].values():
            assert branch["second_image_id"] in task.candidate_ids
            assert branch["second_image_id"] != plan["first_image_id"]
            assert set(branch["outcomes"]) == {"negative", "positive"}
            for outcome in branch["outcomes"].values():
                assert set(outcome["endpoint_positive_probabilities"]) == set(
                    task.endpoint_ids
                )


def test_scale_selection_uses_only_supplied_initial_samples() -> None:
    samples = [(-0.4, False), (-0.2, False), (0.2, True), (0.4, True)]
    selected, losses = baseline.select_scale(samples)
    assert selected in baseline.SCALE_GRID
    assert set(losses) == {str(scale) for scale in baseline.SCALE_GRID}


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
    row = {"task_id": opened.task_id, "policies": plans}
    result = baseline.evaluate_opened_partition([opened], [row])
    assert set(result["pooled"]) == set(baseline.POLICIES)
    assert all(
        0.0 <= result["pooled"][policy]["mean_brier"] <= 1.0
        for policy in baseline.POLICIES
    )
    assert len(
        result["paired_depth2_minus_myopic"]["mean_brier"]["values"]
    ) == 1
