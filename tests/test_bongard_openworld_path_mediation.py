from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_path_mediation as mediation
from scripts import bongard_openworld_vlm_bed as bed
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics


def _task(task_id: str) -> bed.VisualTask:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    return bed.VisualTask(
        task_id=task_id,
        image_ids=image_ids,
        initial_history=tuple(
            (image_id, index % 2 == 0)
            for index, image_id in enumerate(image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={image_id: b"image" for image_id in image_ids},
        actual_labels={image_id: index % 2 == 0 for index, image_id in enumerate(image_ids)},
    )


def _belief(
    task: bed.VisualTask,
    *,
    informative_candidate: str,
    rule_prefix: str,
    includes_first_label: bool,
) -> bed.SemanticBelief:
    hypotheses = []
    for index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        latent = 0.1 if index < 5 else 0.9
        probabilities = []
        for image_id in task.image_ids:
            if image_id in task.endpoint_ids or image_id == informative_candidate:
                probabilities.append(latent)
            else:
                probabilities.append(0.5)
        hypotheses.append(
            bed.SemanticHypothesis(
                hypothesis_id=hypothesis_id,
                rule=f"{rule_prefix} semantic rule {index}",
                history_weight=1.0,
                positive_probabilities=tuple(probabilities),
            )
        )
    history = task.initial_history
    if includes_first_label:
        history = tuple(sorted((*history, (task.candidate_ids[0], True))))
    return bed.SemanticBelief(
        image_ids=task.image_ids,
        history=history,
        hypotheses=tuple(hypotheses),
        history_weights=(0.1,) * 10,
    )


def _fixture(task_id: str):
    task = _task(task_id)
    first = task.candidate_ids[0]
    dynamic_belief = _belief(
        task,
        informative_candidate=task.candidate_ids[1],
        rule_prefix="dynamic",
        includes_first_label=True,
    )
    blind_belief = _belief(
        task,
        informative_candidate=task.candidate_ids[2],
        rule_prefix="blind",
        includes_first_label=False,
    )
    blind_weights = bed.updated_weights_for_label(blind_belief, first, True)
    remaining = tuple(image_id for image_id in task.candidate_ids if image_id != first)
    dynamic_scores = bed.candidate_endpoint_eigs(
        dynamic_belief, remaining, task.endpoint_ids
    )
    blind_scores = bed.candidate_endpoint_eigs(
        blind_belief, remaining, task.endpoint_ids, weights=blind_weights
    )
    dynamic_second = bed.select_best(dynamic_scores)
    blind_second = bed.select_best(blind_scores)
    assert dynamic_second != blind_second
    tree = {
        "task_id": task_id,
        "policies": {
            "dynamic_depth2": {
                "first_image_id": first,
                "first_label": "positive",
                "second_image_id": dynamic_second,
                "second_scores": dynamic_scores,
                "second_score_margin": mechanics.selection_margin(
                    dynamic_scores, dynamic_second
                ),
                "endpoint": {"mean_brier": 0.10, "mean_log_loss": 0.22},
            },
            "history_blind_update_matched_first": {
                "first_image_id": first,
                "first_label": "positive",
                "second_image_id": blind_second,
                "second_scores": blind_scores,
                "second_score_margin": mechanics.selection_margin(
                    blind_scores, blind_second
                ),
                "endpoint": {"mean_brier": 0.20, "mean_log_loss": 0.35},
            },
        },
    }
    key = (first, True)
    context = (task, {key: dynamic_belief}, {key: blind_belief})
    return task, tree, context


def test_task_mediation_replays_belief_action_endpoint_chain() -> None:
    task, tree, context = _fixture("task-a")
    row = mediation.task_mediation_row(
        task=task,
        tree=tree,
        dynamic_branches=context[1],
        history_blind_branches=context[2],
    )
    assert row["second_action_changed"] is True
    assert row["robust_second_action_changed"] is True
    assert row["both_supports_robustly_prefer_own_action"] is True
    assert row["dynamic_support_rule_jaccard_vs_history_blind"] == 0.0
    assert row["endpoint_predictive_probability_mae"] == pytest.approx(0.0)
    assert row["candidate_predictive_probability_mae"] > 0.0
    assert row["second_query_score_spearman"] < 1.0
    assert row["dynamic_minus_history_blind_endpoint_brier"] == pytest.approx(-0.1)
    assert row["dynamic_minus_history_blind_endpoint_log_loss"] == pytest.approx(-0.13)


def test_task_mediation_rejects_changed_score_map() -> None:
    task, tree, context = _fixture("task-a")
    tampered = json.loads(json.dumps(tree))
    second = tampered["policies"]["dynamic_depth2"]["second_image_id"]
    tampered["policies"]["dynamic_depth2"]["second_scores"][second] += 0.01
    with pytest.raises(ValueError, match="score maps changed"):
        mediation.task_mediation_row(
            task=task,
            tree=tampered,
            dynamic_branches=context[1],
            history_blind_branches=context[2],
        )


def test_build_report_summarizes_changed_paths_without_changing_tier() -> None:
    fixtures = [_fixture(f"task-{index}") for index in range(4)]
    report = mediation.build_report(
        stage="mechanics",
        stage_result={
            "status": "mechanics_pass",
            "claim_tier": None,
            "trees": [tree for _, tree, _ in fixtures],
        },
        contexts={task.task_id: context for task, _, context in fixtures},
        authorization={"verified": True},
    )
    assert report["status"] == "path_mediation_complete"
    assert report["task_count"] == 4
    assert report["summary"]["second_action_changed"] == 4
    assert report["summary"]["both_supports_robustly_prefer_own_action"] == 4
    assert report["endpoint_effects"]["changed_actions"][
        "dynamic_minus_history_blind_brier"
    ]["mean_difference"] == pytest.approx(-0.1)
    assert report["changes_claim_tier"] is False
    assert report["authorizes_paid_calls"] is False


def test_run_authorizes_before_loading_raw_contexts(tmp_path: Path) -> None:
    fixtures = [_fixture(f"task-{index}") for index in range(4)]
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "mechanics_pass",
                "claim_tier": None,
                "trees": [tree for _, tree, _ in fixtures],
            }
        ),
        encoding="utf-8",
    )
    state = {"authorized": False}

    def authorizer(**kwargs):
        del kwargs
        state["authorized"] = True
        return {"verified": True}

    def context_loader(**kwargs):
        del kwargs
        assert state["authorized"] is True
        return {task.task_id: context for task, _, context in fixtures}

    output = tmp_path / "MEDIATION.json"
    report = mediation.run_report(
        stage="mechanics",
        result_path=result_path,
        output_path=output,
        wrapper_result=tmp_path / "WRAPPER.json",
        authorizer=authorizer,
        context_loader=context_loader,
    )
    assert output.is_file()
    assert report["status"] == "path_mediation_complete"
    assert report["model_calls"] == 0
    assert report["cost_usd"] == 0.0
