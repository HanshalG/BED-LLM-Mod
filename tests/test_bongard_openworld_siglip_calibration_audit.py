from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from scripts import bongard_openworld_siglip_calibration_audit as audit
from scripts import bongard_openworld_vlm_bed as bed


class BombLabels(Mapping[str, bool]):
    def __getitem__(self, key: str) -> bool:
        raise AssertionError("an unobserved label was accessed")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("unobserved labels were enumerated")

    def __len__(self) -> int:
        raise AssertionError("unobserved labels were counted")


def _task() -> tuple[bed.VisualTask, dict[str, np.ndarray]]:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    values = np.zeros((14, 4), dtype=float)
    values[0] = [1.0, 0.0, 0.0, 0.0]
    values[1] = [0.9, 0.1, 0.0, 0.0]
    values[2] = [0.0, 1.0, 0.0, 0.0]
    values[3] = [0.1, 0.9, 0.0, 0.0]
    values /= np.where(
        np.linalg.norm(values, axis=1, keepdims=True) == 0,
        1.0,
        np.linalg.norm(values, axis=1, keepdims=True),
    )
    task = bed.VisualTask(
        task_id="task-calibration-audit",
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


def test_task_samples_use_only_initial_history_labels() -> None:
    task, embeddings = _task()
    samples = audit.task_samples(task, embeddings)
    assert len(samples) == 4
    assert [label for _, label in samples] == [True, True, False, False]


def test_partition_metrics_report_task_paired_uncertainty() -> None:
    task, embeddings = _task()
    result = audit.partition_metrics(
        [task], {task.task_id: embeddings}, seed=20260809
    )
    assert result["tasks"] == 1
    assert result["samples"] == 4
    assert result["paired_frozen_minus_zero_log_loss"]["bootstrap_draws"] == 20_000
    assert result["paired_frozen_minus_zero_brier"]["bootstrap_draws"] == 20_000


def test_frozen_inputs_verify() -> None:
    observed = audit.verify_frozen_inputs()
    assert observed["protocol"] == audit.PROTOCOL_SHA256
    assert observed["plans"] == audit.FINAL_PLANS_SHA256
