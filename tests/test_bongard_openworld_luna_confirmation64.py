from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_vlm_bed as bed


def _image_bytes(index: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (36, 28), (50 + index, 100, 150)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _task(task_id: str, index: int) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {
        image_id: image_index % 2 == 0
        for image_index, image_id in enumerate(image_ids)
    }
    return bed.VisualTask(
        task_id=task_id,
        image_ids=image_ids,
        initial_history=tuple(
            sorted((image_id, labels[image_id]) for image_id in image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={
            image_id: _image_bytes(image_index)
            for image_index, image_id in enumerate(image_ids)
        },
        actual_labels=labels,
        hidden_values=(f"secret concept {index}", f"images/{index}/source.jpg"),
    )


def _tasks() -> list[bed.VisualTask]:
    manifest = confirmation.verify_protocol_manifest()
    ids = [
        task_id
        for block_id in confirmation.BLOCK_ORDER
        for task_id in manifest["task_ids_by_block"][block_id]
    ]
    return [_task(task_id, index) for index, task_id in enumerate(ids)]


def _request(messages):
    return json.loads(
        next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
    )


def _response(messages, *, seed: int) -> str:
    request = _request(messages)
    image_ids = request["image_order"]
    observed = {
        item["image_id"]: item["label"] == "positive"
        for item in request["observed_labels"]
    }
    extras = sorted(set(observed) - set(image_ids[:4]))
    first_extra = extras[0] if extras else None
    hypotheses = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        centered = (hypothesis_index - 4.5) / 4.5
        probabilities = []
        for image_index, image_id in enumerate(image_ids):
            if image_id in observed:
                value = 90 - hypothesis_index if observed[image_id] else 10 + hypothesis_index
            else:
                amplitude = 8 + (seed % 7) * 3 + (image_index % 4) * 2
                if first_extra == "image-05":
                    amplitude = 35 if image_id != first_extra else 8
                elif first_extra is not None:
                    amplitude = 8
                base = 65 if image_index % 2 == 0 else 35
                value = min(95, max(5, round(base + amplitude * centered)))
            probabilities.append(value)
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": f"{request['task_id']} {first_extra or 'root'} rule {hypothesis_index}",
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": hypotheses})


class FixtureAdapter:
    def __init__(self):
        self.requests = 0

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        del kwargs
        self.requests += len(batch_messages)
        return [
            _response(messages, seed=seed)
            for messages, seed in zip(batch_messages, seeds, strict=True)
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": self.requests * 100,
            "adapter_completion_tokens": self.requests * 200,
            "adapter_cost_usd": self.requests * 0.001,
        }


def test_manifest_and_task_slices_are_exact() -> None:
    manifest = confirmation.verify_protocol_manifest()
    tasks = confirmation.load_planning_tasks()
    assert manifest["task_count"] == 96
    assert len(tasks) == 96
    assert [
        len(confirmation.confirmation_tasks_for_block(tasks, block_id))
        for block_id in confirmation.BLOCK_ORDER
    ] == [24, 24, 24, 24]


def test_confirmation_block_runs_and_replays_with_fixture(
    tmp_path, monkeypatch
) -> None:
    tasks = _tasks()
    monkeypatch.setattr(
        confirmation,
        "verify_development_authorization",
        lambda: {
            "verified": True,
            "claim_report_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        confirmation.development,
        "verify_mechanics_result",
        lambda path: {"verified": True, "result_sha256": "b" * 64},
    )
    output = tmp_path / "block-a"
    result = confirmation.run_block(
        output_dir=output,
        run_id="fixture",
        block_id="a",
        mechanics_result=tmp_path / "mechanics.json",
        all_confirmation_tasks=tasks,
        adapter=FixtureAdapter(),
    )
    assert result["status"] == "block_mechanics_pass"
    assert result["gates"][
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes"
    ]
    assert result["gates"][
        "terminal_beliefs_retain_both_queried_labels_better_than_constant_half"
    ]
    assert result["gates"][
        "history_blind_update_matched_first_exactly_matches_dynamic_first"
    ]
    assert result["terminal_label_obedience"]["negative_mean_brier"] < 0.25
    assert result["terminal_label_obedience"]["positive_mean_brier"] < 0.25
    assert result["protocol"]["block_size"] == 24
    assert result["usage"]["adapter_requests"] <= 1_056
    replay = confirmation.replay_block(
        result_path=output / "RESULT.json", all_confirmation_tasks=tasks
    )
    assert replay["verified"] is True
    assert replay["block_id"] == "a"
    monkeypatch.setattr(
        confirmation.mechanics,
        "history_blind_update_matched_first_is_exact",
        lambda tree: False,
    )
    tampered_gates = confirmation._block_gates(
        task_count=24,
        artifacts=replay["artifacts"],
        usage=result["usage"],
        prompt_errors=[],
        mechanics_verification={"verified": True},
    )
    assert not tampered_gates[
        "history_blind_update_matched_first_exactly_matches_dynamic_first"
    ]
    assert not tampered_gates["all_pass"]


def _scored_tree(task_id: str, index: int) -> dict:
    endpoint_values = {
        "myopic_width": (0.20, 0.40),
        "compute_matched_myopic_ensemble": (0.15, 0.33),
        "fixed_depth2": (0.12, 0.30),
        "fixed_score_dynamic_update": (0.14, 0.32),
        "dynamic_depth2": (0.10, 0.25),
        "shuffled_dynamic_depth2": (0.13, 0.31),
        "history_blind_depth2": (0.18, 0.36),
        "history_blind_update_matched_first": (0.16, 0.34),
        "random": (0.24, 0.48),
    }
    policies = {}
    for policy in mechanics.POLICIES:
        brier, log_loss = endpoint_values[policy]
        if policy in {
            "dynamic_depth2",
            "history_blind_update_matched_first",
        }:
            first = "dynamic-first"
        elif policy in {"fixed_depth2", "fixed_score_dynamic_update"}:
            first = "fixed_depth2-first"
        else:
            first = f"{policy}-first"
        policies[policy] = {
            "first_image_id": first,
            "first_score_margin": 0.1,
            "second_image_id": (
                "blind-second"
                if policy == "history_blind_update_matched_first"
                else f"{policy}-second"
            ),
            "second_score_margin": 0.1,
            "final_history_key": f"{policy}-{index}",
            "endpoint": {
                "mean_brier": brier,
                "mean_log_loss": log_loss,
                "accuracy": 0.8,
                "mean_truth_probability": 0.8,
            },
        }
    return {
        "task_id": task_id,
        "policies": policies,
        "root_scores": {
            "dynamic_depth2": {
                "dynamic-first": 1.0,
                "myopic_width-first": 0.5,
                "fixed_depth2-first": 0.4,
            }
        },
        "ranking_fidelity": {
            "dynamic_depth2": 0.8,
            "compute_matched_myopic_ensemble": 0.45,
            "myopic_width": 0.5,
            "fixed_depth2": 0.6,
            "shuffled_dynamic_depth2": 0.3,
            "history_blind_depth2": 0.4,
        },
        "root_candidate_brier": 0.10,
    }


def test_combined_analysis_uses_strict_confirmation_gates(
    tmp_path, monkeypatch
) -> None:
    tasks = _tasks()
    replays = []
    for block_id in confirmation.BLOCK_ORDER:
        block_tasks = confirmation.confirmation_tasks_for_block(tasks, block_id)
        replays.append(
            {
                "verified": True,
                "block_id": block_id,
                "protocol_manifest_sha256": confirmation.PROTOCOL_MANIFEST_SHA256,
                "result_sha256": block_id * 64,
                "raw_responses_sha256": block_id.upper() * 64,
                "tasks": block_tasks,
                "artifacts": {"trees": []},
            }
        )
    by_name = {f"{block}.json": replay for block, replay in zip(confirmation.BLOCK_ORDER, replays)}
    monkeypatch.setattr(
        confirmation,
        "replay_block",
        lambda result_path, **kwargs: by_name[result_path.name],
    )
    trees = [_scored_tree(task.task_id, index) for index, task in enumerate(tasks)]
    monkeypatch.setattr(
        confirmation,
        "_build_scored_trees",
        lambda **kwargs: trees,
    )
    result = confirmation.analyze_combined(
        block_results=[tmp_path / f"{block}.json" for block in confirmation.BLOCK_ORDER],
        output_path=tmp_path / "combined.json",
        all_confirmation_tasks=tasks,
    )
    assert result["status"] == "confirmation_pass"
    assert result["gates"]["all_pass"] is True
    assert result["comparisons_vs_myopic"]["dynamic_depth2"]["mean_brier"]["ci95"][1] < 0
    assert result["dynamic_vs_fixed_depth2"]["mean_brier"]["ci95"][1] < 0
    assert result["dynamic_vs_fixed_score_dynamic_update"]["mean_brier"][
        "ci95"
    ][1] < 0
    assert result["dynamic_vs_history_blind_update_matched_first"][
        "mean_brier"
    ]["ci95"][1] < 0
    assert (
        result[
            "dynamic_vs_history_blind_update_matched_first_changed_final_histories"
        ]
        == 96
    )
    assert (
        result[
            "dynamic_vs_history_blind_update_matched_first_robust_second_action_changes"
        ]
        == 96
    )
    assert result["gates"][
        "dynamic_brier_vs_fixed_depth2_paired_tree_bootstrap_95pct_upper_below_zero"
    ]
    assert result["gates"][
        "dynamic_brier_vs_fixed_score_dynamic_update_paired_tree_bootstrap_95pct_upper_below_zero"
    ]
    assert result["gates"][
        "dynamic_brier_vs_history_blind_update_matched_first_paired_tree_bootstrap_95pct_upper_below_zero"
    ]
    assert result["gates"][
        "dynamic_brier_relative_improvement_vs_history_blind_update_matched_first_at_least_3_percent"
    ]
    assert result["gates"][
        "dynamic_log_loss_is_not_worse_than_history_blind_update_matched_first"
    ]
    assert result["sealed_test_authorized"] is False


def test_confirmation_rejects_gain_caused_only_by_unmatched_fixed_update(
    tmp_path, monkeypatch
) -> None:
    tasks = _tasks()
    replays = []
    for block_id in confirmation.BLOCK_ORDER:
        replays.append(
            {
                "verified": True,
                "block_id": block_id,
                "protocol_manifest_sha256": confirmation.PROTOCOL_MANIFEST_SHA256,
                "result_sha256": block_id * 64,
                "raw_responses_sha256": block_id.upper() * 64,
                "tasks": confirmation.confirmation_tasks_for_block(
                    tasks, block_id
                ),
                "artifacts": {"trees": []},
            }
        )
    by_name = {
        f"{block}.json": replay
        for block, replay in zip(
            confirmation.BLOCK_ORDER, replays, strict=True
        )
    }
    monkeypatch.setattr(
        confirmation,
        "replay_block",
        lambda result_path, **kwargs: by_name[result_path.name],
    )
    trees = [_scored_tree(task.task_id, index) for index, task in enumerate(tasks)]
    for tree in trees:
        tree["policies"]["fixed_score_dynamic_update"]["endpoint"][
            "mean_brier"
        ] = 0.08
        tree["policies"]["fixed_score_dynamic_update"]["endpoint"][
            "mean_log_loss"
        ] = 0.20
    monkeypatch.setattr(
        confirmation, "_build_scored_trees", lambda **kwargs: trees
    )

    result = confirmation.analyze_combined(
        block_results=[
            tmp_path / f"{block}.json" for block in confirmation.BLOCK_ORDER
        ],
        output_path=tmp_path / "combined-unmatched-only.json",
        all_confirmation_tasks=tasks,
    )

    assert result["status"] == "confirmation_null"
    assert all(
        result["gates"][name]
        for name in (
            "dynamic_brier_relative_improvement_vs_fixed_depth2_at_least_3_percent",
            "dynamic_brier_vs_fixed_depth2_paired_tree_bootstrap_95pct_upper_below_zero",
            "dynamic_log_loss_is_not_worse_than_fixed_depth2",
        )
    )
    assert result["gates"][
        "dynamic_brier_relative_improvement_vs_fixed_score_dynamic_update_at_least_3_percent"
    ] is False
    assert result["gates"][
        "dynamic_brier_vs_fixed_score_dynamic_update_paired_tree_bootstrap_95pct_upper_below_zero"
    ] is False
    assert result["gates"]["all_pass"] is False


def test_all_confirmation_blocks_and_combined_result_replay(
    tmp_path, monkeypatch
) -> None:
    tasks = _tasks()
    monkeypatch.setattr(
        confirmation,
        "verify_development_authorization",
        lambda: {
            "verified": True,
            "claim_report_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        confirmation.development,
        "verify_mechanics_result",
        lambda path: {"verified": True, "result_sha256": "b" * 64},
    )
    block_results = []
    for block_id in confirmation.BLOCK_ORDER:
        output = tmp_path / f"block-{block_id}"
        confirmation.run_block(
            output_dir=output,
            run_id=f"fixture-{block_id}",
            block_id=block_id,
            mechanics_result=tmp_path / "mechanics.json",
            all_confirmation_tasks=tasks,
            adapter=FixtureAdapter(),
        )
        block_results.append(output / "RESULT.json")

    combined = tmp_path / "COMBINED_RESULT.json"
    confirmation.analyze_combined(
        block_results=block_results,
        output_path=combined,
        all_confirmation_tasks=tasks,
    )
    verification = confirmation.verify_combined_result(
        result_path=combined,
        block_results=block_results,
        all_confirmation_tasks=tasks,
    )
    assert verification["verified"] is True
    assert verification["status"] in {"confirmation_pass", "confirmation_null"}
    assert verification["sealed_test_authorized"] is False
