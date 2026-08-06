from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as tree
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed


def _image_bytes(index: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (36, 28), (50 + index, 100, 150)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _task(index: int) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {
        image_id: image_index % 2 == 0
        for image_index, image_id in enumerate(image_ids)
    }
    return bed.VisualTask(
        task_id=f"task-{index:012d}",
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
        hidden_values=(
            f"secret concept {index}",
            f"images/{index:04d}/pos__0__source.jpg",
        ),
    )


def _request(messages):
    return json.loads(
        next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
    )


def _response(messages) -> str:
    request = _request(messages)
    image_ids = request["image_order"]
    observed = {
        item["image_id"]: item["label"] == "positive"
        for item in request["observed_labels"]
    }
    initial_ids = set(image_ids[:4])
    extras = sorted(set(observed) - initial_ids)
    first_extra = extras[0] if extras else None
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        probabilities = []
        centered = (hypothesis_index - 4.5) / 4.5
        for image_index, image_id in enumerate(image_ids):
            if image_id in observed:
                value = 90 - hypothesis_index if observed[image_id] else 10 + hypothesis_index
            else:
                amplitude = 12 + (image_index % 4) * 3
                if first_extra == "image-05":
                    amplitude = 35 if image_id != first_extra else 8
                elif first_extra is not None:
                    amplitude = 8
                value = round(50 + amplitude * centered)
                value = min(95, max(5, value))
            probabilities.append(value)
        branch_tag = first_extra or "root"
        label_tag = (
            "none" if first_extra is None else str(int(observed[first_extra]))
        )
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": (
                    f"{request['task_id']} {branch_tag} {label_tag} "
                    f"semantic visual rule {hypothesis_index + 1}"
                ),
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


class FixtureAdapter:
    def __init__(self):
        self.requests = 0

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        self.requests += len(batch_messages)
        return [_response(messages) for messages in batch_messages]

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


def _serving_result(tmp_path: Path, tasks) -> Path:
    cases = serving.build_smoke_cases(tasks)

    class ServingAdapter:
        def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
            del kwargs
            return [_response(messages) for messages in batch_messages]

        def usage_snapshot(self):
            return {
                "adapter_requests": 10,
                "http_attempts": 10,
                "retry_count": 0,
                "provider_error_retries": 0,
                "adapter_reasoning_tokens": 0,
                "forced_exits": 0,
                "adapter_prompt_tokens": 1000,
                "adapter_completion_tokens": 2000,
                "adapter_cost_usd": 0.01,
            }

    output = tmp_path / "serving"
    result = serving.run_smoke(
        output_dir=output,
        run_id="serving-fixture",
        tasks=tasks,
        adapter=ServingAdapter(),
    )
    assert result["status"] == "passed"
    assert len(cases) == 10
    return output / "RESULT.json"


def test_full_fixture_tree_is_shared_executable_and_endpoint_scored(
    tmp_path: Path,
) -> None:
    tasks = [_task(index) for index in range(4)]
    serving_result = _serving_result(tmp_path, tasks)
    adapter = FixtureAdapter()
    result = tree.run_mechanics(
        output_dir=tmp_path / "tree",
        run_id="tree-fixture",
        serving_result=serving_result,
        tasks=tasks,
        adapter=adapter,
    )
    assert result["status"] == "mechanics_pass"
    assert result["gates"]["all_pass"]
    assert result["protocol"]["first_stage_requests"] == 68
    assert result["usage"]["adapter_requests"] == (
        68 + result["protocol"]["distinct_final_history_requests"]
    )
    assert result["protocol"]["development_accessed"] is False
    assert set(result["pooled_policy_metrics"]) == set(tree.POLICIES)
    assert all(
        set(task_result["policies"]) == set(tree.POLICIES)
        for task_result in result["trees"]
    )
    assert all(
        len(task_result["all_first_action_paths"]) == 8
        for task_result in result["trees"]
    )
    assert set(result["mean_ranking_fidelity"]) == {
        "myopic_width",
        "fixed_depth2",
        "dynamic_depth2",
        "shuffled_dynamic_depth2",
    }
    raw = json.loads(
        (tmp_path / "tree/private/RAW_RESPONSES.json").read_text()
    )
    assert raw["endpoint_labels_accessed_after_all_query_selection"] is True
    assert raw["development_accessed"] is False
    serving_verification = aug10.validate_serving_artifact(
        serving_result, tasks=tasks
    )
    mechanics_verification = aug10.validate_mechanics_artifact(
        tmp_path / "tree/RESULT.json",
        serving_result=serving_result,
        tasks=tasks,
    )
    assert serving_verification["verified"]
    assert mechanics_verification["verified"]


def test_shuffled_mapping_is_rotation_without_fixed_points() -> None:
    task = _task(0)
    belief = bed.parse_belief_response(
        _response(bed.build_belief_messages(task, task.initial_history)),
        image_ids=task.image_ids,
        history=task.initial_history,
    )
    branches = {
        (candidate, label): belief
        for candidate in task.candidate_ids
        for label in (False, True)
    }
    shuffled, mapping = tree.rotate_branch_map(task, branches)
    assert set(shuffled) == set(branches)
    assert all(source != target for source, target in mapping.items())
    assert len(set(mapping.values())) == len(mapping)


def test_final_cases_deduplicate_shared_policy_histories() -> None:
    task = _task(0)
    history = tuple(
        sorted((*task.initial_history, ("image-04", True), ("image-05", False)))
    )
    plans = {
        task.task_id: {
            "policies": {
                policy: {
                    "final_history": history,
                    "final_history_key": tree.history_key(history),
                }
                for policy in tree.POLICIES
            }
        }
    }
    cases = tree.final_cases([task], plans)
    assert len(cases) == 1


def test_reconcile_full_tree_preserves_prior_smoke_spend() -> None:
    ledger = {
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 0.03,
        "bongard_luna_vlm_serving_smoke": {
            "status": "passed",
            "actual_cost_usd": 0.03,
        },
    }
    result = tree.reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=0.22,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 100.2,
            "balance_usd": 29.8,
        },
        status="mechanics_pass",
    )
    assert math.isclose(result["recorded_actual_spend_usd"], 0.25)
    assert math.isclose(
        result["bongard_luna_vlm_mechanics_tree"]["actual_cost_usd"],
        0.22,
    )
    assert result["bongard_luna_vlm_serving_smoke"]["status"] == "passed"
