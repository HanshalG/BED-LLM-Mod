from copy import deepcopy
import json
import math

import pytest

from scripts import bongard_openworld_random_strategy_control as audit
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_vlm_bed as bed


def _tree(index: int) -> dict:
    task_id = f"task-{index}"
    candidates = ["d", "b", "a", "c"]
    random_first, random_second = audit._expected_random_choices(
        task_id=task_id, candidates=candidates
    )
    dynamic_first = next(
        candidate for candidate in sorted(candidates) if candidate != random_first
    )
    dynamic_second = next(
        candidate
        for candidate in sorted(candidates)
        if candidate not in {random_first, dynamic_first}
    )
    return {
        "task_id": task_id,
        "root_scores": {
            "dynamic_depth2": {
                candidate: float(position)
                for position, candidate in enumerate(candidates)
            }
        },
        "policies": {
            "dynamic_depth2": {
                "first_image_id": dynamic_first,
                "second_image_id": dynamic_second,
                "final_history_key": f"dynamic-{index}",
                "endpoint": {
                    "mean_brier": 0.10 + index * 0.001,
                    "mean_log_loss": 0.25 + index * 0.002,
                },
            },
            "random": {
                "first_image_id": random_first,
                "second_image_id": random_second,
                "first_score": None,
                "first_score_margin": None,
                "second_scores": None,
                "second_score_margin": None,
                "final_history_key": f"random-{index}",
                "endpoint": {
                    "mean_brier": 0.24 + index * 0.001,
                    "mean_log_loss": 0.48 + index * 0.002,
                },
            },
        },
    }


def _stage_result() -> dict:
    return {
        "status": "mechanics_pass",
        "claim_tier": None,
        "trees": [_tree(index) for index in range(4)],
    }


def test_build_report_replays_random_draws_and_paired_effects() -> None:
    report = audit.build_report(
        stage="mechanics",
        stage_result=_stage_result(),
        authorization={"verified": True, "stage": "mechanics"},
    )

    assert report["status"] == "random_strategy_control_audit_complete"
    assert report["random_policy_seed"] == 2_026_081_022
    assert report["random_policy_draws_without_replacement"] is True
    assert report["random_draws_replayed_exactly"] is True
    assert report["task_count"] == 4
    assert report["first_query_changes"] == 4
    assert report["final_history_changes"] == 4
    assert report["model_calls"] == 0
    assert report["cost_usd"] == 0.0
    assert report["authorizes_paid_calls"] is False
    assert report["changes_claim_tier"] is False
    for metric in audit.METRICS:
        summary = report["comparisons"][metric]
        assert summary["n"] == 4
        assert summary["mean_difference"] < 0
        assert summary["bootstrap_draws"] == 20_000
        assert summary["bootstrap_probability_improvement"] == 1.0
        assert summary["wins"] == 4
        assert summary["ties"] == 0
        assert summary["losses"] == 0
        assert math.isclose(
            summary["standard_error"],
            summary["sample_sd"] / math.sqrt(summary["n"]),
        )


def test_all_frozen_stage_tasks_replay_the_random_policy_without_labels() -> None:
    expected_counts = {"mechanics": 4, "development": 64, "confirmation": 96}

    for stage, expected_count in expected_counts.items():
        tasks = bed.load_validation_partition_tasks(
            stage, include_endpoint_labels=False
        )
        assert len(tasks) == expected_count
        for task in tasks:
            assert list(task.candidate_ids) == sorted(task.candidate_ids)
            assert audit._expected_random_choices(
                task_id=task.task_id, candidates=task.candidate_ids
            ) == mechanics._random_choices(task)


@pytest.mark.parametrize(
    "mutation",
    ("first", "duplicate", "score", "endpoint", "dynamic_choice", "task_id"),
)
def test_build_report_fails_closed_on_random_contract_tamper(mutation: str) -> None:
    result = deepcopy(_stage_result())
    tree = result["trees"][0]
    random_policy = tree["policies"]["random"]
    if mutation == "first":
        random_policy["first_image_id"] = "z"
    elif mutation == "duplicate":
        random_policy["second_image_id"] = random_policy["first_image_id"]
    elif mutation == "score":
        random_policy["first_score"] = 0.1
    elif mutation == "endpoint":
        random_policy["endpoint"]["mean_brier"] = float("nan")
    elif mutation == "dynamic_choice":
        tree["policies"]["dynamic_depth2"]["second_image_id"] = "z"
    else:
        tree["task_id"] = result["trees"][1]["task_id"]

    with pytest.raises(ValueError):
        audit.build_report(
            stage="mechanics",
            stage_result=result,
            authorization={"verified": True},
        )


def test_run_report_authorizes_before_loading_and_writes_once(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(_stage_result()), encoding="utf-8")
    output_path = tmp_path / "AUDIT.json"
    events = []

    def authorizer(**kwargs):
        assert kwargs["stage"] == "mechanics"
        assert not events
        events.append("authorized")
        return {"verified": True, "stage": "mechanics"}

    def loader(path):
        assert events == ["authorized"]
        events.append("loaded")
        return json.loads(path.read_text(encoding="utf-8"))

    report = audit.run_report(
        stage="mechanics",
        result_path=result_path,
        output_path=output_path,
        authorizer=authorizer,
        result_loader=loader,
    )

    assert events == ["authorized", "loaded"]
    assert json.loads(output_path.read_text()) == report
    assert report["stage_result_sha256"] == audit._sha256(result_path)
    with pytest.raises(FileExistsError):
        audit.run_report(
            stage="mechanics",
            result_path=result_path,
            output_path=output_path,
            authorizer=authorizer,
            result_loader=loader,
        )


def test_run_report_never_loads_an_unauthorized_result(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(_stage_result()), encoding="utf-8")
    loaded = False

    def loader(path):
        nonlocal loaded
        loaded = True
        return json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="authorization failed"):
        audit.run_report(
            stage="mechanics",
            result_path=result_path,
            output_path=tmp_path / "AUDIT.json",
            authorizer=lambda **kwargs: {"verified": False},
            result_loader=loader,
        )
    assert loaded is False
