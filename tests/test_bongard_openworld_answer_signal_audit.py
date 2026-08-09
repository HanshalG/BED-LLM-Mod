from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_answer_signal_audit as audit
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_vlm_bed as bed


def _task(task_id: str) -> bed.VisualTask:
    image_ids = tuple(f"{task_id}-I{index:02d}" for index in range(14))
    return bed.VisualTask(
        task_id=task_id,
        image_ids=image_ids,
        initial_history=tuple(
            (image_id, index % 2 == 0)
            for index, image_id in enumerate(image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={image_id: b"unused" for image_id in image_ids},
        actual_labels={},
    )


def _belief(
    task: bed.VisualTask,
    *,
    history: tuple[tuple[str, bool], ...],
    probability: float,
    rule_prefix: str,
) -> bed.SemanticBelief:
    hypotheses = tuple(
        bed.SemanticHypothesis(
            hypothesis_id=f"H{index:02d}",
            rule=f"{rule_prefix} rule {index}",
            history_weight=1.0,
            positive_probabilities=(probability,) * len(task.image_ids),
        )
        for index in range(1, 11)
    )
    return bed.SemanticBelief(
        image_ids=task.image_ids,
        history=history,
        hypotheses=hypotheses,
        history_weights=(0.1,) * 10,
    )


def _belief_maps(tasks: list[bed.VisualTask]):
    dynamic = {}
    blind = {}
    for task in tasks:
        dynamic[task.task_id] = {}
        blind[task.task_id] = {}
        for candidate_id in task.candidate_ids:
            for label, dynamic_probability, blind_probability in (
                (False, 0.2, 0.2),
                (True, 0.3, 0.21),
            ):
                dynamic[task.task_id][(candidate_id, label)] = _belief(
                    task,
                    history=tuple(sorted((*task.initial_history, (candidate_id, label)))),
                    probability=dynamic_probability,
                    rule_prefix=f"dynamic-{label}",
                )
                blind[task.task_id][(candidate_id, label)] = _belief(
                    task,
                    history=task.initial_history,
                    probability=blind_probability,
                    rule_prefix=f"blind-{label}",
                )
    return dynamic, blind


def _response(probability: int, prefix: str) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {
                    "hypothesis_id": f"H{index:02d}",
                    "rule": f"{prefix} visual rule {index}",
                    "history_weight": 10,
                    "positive_probabilities": [probability] * 14,
                }
                for index in range(1, 11)
            ]
        }
    )


def _mechanics_fixture(root: Path, tasks: list[bed.VisualTask]) -> Path:
    result_path = root / "RESULT.json"
    raw_path = root / "private/RAW_RESPONSES.json"
    cases = mechanics.first_stage_cases(tasks)
    responses = []
    for case in cases:
        if case.kind == "root":
            responses.append(_response(50, "root"))
        elif case.kind == "branch":
            responses.append(
                _response(30 if case.simulated_label else 20, "dynamic")
            )
        else:
            responses.append(
                _response(21 if case.simulated_label else 20, "blind")
            )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "first_stage_case_ids": [case.case_id for case in cases],
                "first_stage_responses": responses,
            }
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "mechanics_pass",
                "raw_responses_sha256": audit.sha256_file(raw_path),
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_answer_signal_passes_only_above_same_prompt_noise() -> None:
    tasks = [_task(f"T{index}") for index in range(4)]
    dynamic, blind = _belief_maps(tasks)

    metrics = audit.answer_signal_metrics(
        tasks=tasks,
        dynamic_by_task=dynamic,
        blind_by_task=blind,
    )

    assert metrics["pair_count"] == 32
    assert metrics["pooled_dynamic_prediction_mae"] == pytest.approx(0.1)
    assert metrics["pooled_history_blind_prediction_mae"] == pytest.approx(0.01)
    assert metrics["pooled_prediction_mae_advantage"] == pytest.approx(0.09)
    assert metrics["gates"]["all_pass"] is True


def test_one_task_without_positive_advantage_fails() -> None:
    tasks = [_task(f"T{index}") for index in range(4)]
    dynamic, blind = _belief_maps(tasks)
    failed = tasks[-1]
    for candidate_id in failed.candidate_ids:
        blind[failed.task_id][(candidate_id, True)] = _belief(
            failed,
            history=failed.initial_history,
            probability=0.4,
            rule_prefix="noisy-blind",
        )

    metrics = audit.answer_signal_metrics(
        tasks=tasks,
        dynamic_by_task=dynamic,
        blind_by_task=blind,
    )

    assert metrics["gates"][
        "answer_conditioned_mae_advantage_is_positive_in_every_task"
    ] is False
    assert metrics["gates"]["all_pass"] is False


def test_report_replays_exact_mechanics_raw_responses(tmp_path: Path) -> None:
    tasks = [_task(f"T{index}") for index in range(4)]
    mechanics_result = _mechanics_fixture(tmp_path / "mechanics", tasks)
    output = tmp_path / "ANSWER_SIGNAL.json"

    report = audit.run_report(
        mechanics_result=mechanics_result,
        output_path=output,
        tasks=tasks,
    )
    verification = audit.verify_report(
        report_path=output,
        mechanics_result=mechanics_result,
        tasks=tasks,
    )

    assert report["status"] == "answer_signal_valid"
    assert report["model_calls_made"] == 0
    assert report["cost_usd"] == 0.0
    assert report["authorizes_development"] is True
    assert verification["report_sha256"] == audit.sha256_file(output)

    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["metrics"]["pooled_prediction_mae_advantage"] = 99.0
    output.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="not an exact passing report"):
        audit.verify_report(
            report_path=output,
            mechanics_result=mechanics_result,
            tasks=tasks,
        )


def test_default_report_uses_only_frozen_label_free_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = audit.load_label_free_mechanics_tasks()
    mechanics_result = _mechanics_fixture(tmp_path / "mechanics", tasks)

    def forbidden_general_loader():
        raise AssertionError("general task loader materializes sealed labels")

    monkeypatch.setattr(bed, "load_mechanics_tasks", forbidden_general_loader)
    report = audit.build_report(mechanics_result=mechanics_result)

    assert report["task_manifest"] == {
        "sha256": audit.PUBLIC_TASK_MANIFEST_SHA256,
        "sealed_candidate_and_endpoint_labels": True,
        "hidden_source_values_loaded": False,
    }
    assert all(not task.actual_labels for task in tasks)
    assert all(not task.image_bytes for task in tasks)
    assert all(not task.hidden_values for task in tasks)


@pytest.mark.parametrize(
    "changed",
    (
        {"actual_labels": {"image-00": True}},
        {"hidden_values": ("sealed concept",)},
    ),
)
def test_report_rejects_materialized_sealed_task_values(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    tasks = audit.load_label_free_mechanics_tasks()
    mechanics_result = _mechanics_fixture(tmp_path / "mechanics", tasks)
    tasks[0] = replace(tasks[0], **changed)

    with pytest.raises(ValueError, match="forbids"):
        audit.build_report(mechanics_result=mechanics_result, tasks=tasks)
