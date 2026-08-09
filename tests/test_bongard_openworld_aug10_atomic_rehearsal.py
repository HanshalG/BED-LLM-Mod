from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_aug10_final_handoff as final_handoff
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_random_strategy_control as random_control
from scripts import bongard_openworld_vlm_bed as bed


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(messages) -> dict:
    return json.loads(
        next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
    )


def _real_task_response(
    messages, *, tasks_by_id: dict[str, bed.VisualTask], seed: int | None = None
) -> str:
    request = _request(messages)
    task = tasks_by_id[request["task_id"]]
    image_ids = request["image_order"]
    observed = {
        item["image_id"]: item["label"] == "positive"
        for item in request["observed_labels"]
    }
    initial_ids = set(dict(task.initial_history))
    extras = sorted(set(observed) - initial_ids)
    first_extra = extras[0] if extras else None
    branch_label = "none" if first_extra is None else str(int(observed[first_extra]))
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        centered = (hypothesis_index - 4.5) / 4.5
        probabilities = []
        for image_index, image_id in enumerate(image_ids):
            if image_id in observed:
                value = (
                    90 - hypothesis_index
                    if observed[image_id]
                    else 10 + hypothesis_index
                )
            else:
                seed_scale = 0 if seed is None else seed % 7
                amplitude = 8 + seed_scale * 3 + (image_index % 4) * 2
                if first_extra is None and seed is not None:
                    preferred = 4 + (seed % 8)
                    amplitude = 40 if image_index == preferred else 6
                if first_extra is not None:
                    branch_key = sum(map(ord, first_extra))
                    amplitude = 12 + (
                        (
                            branch_key
                            + int(observed[first_extra]) * 17
                            + image_index
                        )
                        % 5
                    ) * 5
                base = 65 if task.actual_labels[image_id] else 35
                value = round(base + amplitude * centered)
                value = min(95, max(5, value))
            probabilities.append(value)
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": (
                    f"{request['task_id']} {first_extra or 'root'} "
                    f"{branch_label} zero-call semantic rule "
                    f"{hypothesis_index + 1}"
                ),
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


class _RealTaskFixtureAdapter:
    def __init__(self, tasks: list[bed.VisualTask]) -> None:
        self.tasks_by_id = {task.task_id: task for task in tasks}
        self.requests = 0

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        self.requests += len(batch_messages)
        return [
            _real_task_response(messages, tasks_by_id=self.tasks_by_id)
            for messages in batch_messages
        ]

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        del kwargs
        assert len(batch_messages) == len(seeds)
        self.requests += len(batch_messages)
        return [
            _real_task_response(
                messages, tasks_by_id=self.tasks_by_id, seed=seed
            )
            for messages, seed in zip(batch_messages, seeds, strict=True)
        ]

    def usage_snapshot(self) -> dict:
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


def _real_authorized_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    tasks = sorted(bed.load_mechanics_tasks(), key=lambda task: task.task_id)
    serving_dir = tmp_path / "real-serving"
    serving_result = serving.run_smoke(
        output_dir=serving_dir,
        run_id="zero-call-atomic-serving",
        tasks=tasks,
        adapter=_RealTaskFixtureAdapter(tasks),
    )
    assert serving_result["status"] == "passed"
    serving_path = serving_dir / "RESULT.json"

    mechanics_dir = tmp_path / "real-mechanics"
    mechanics_result = mechanics.run_mechanics(
        output_dir=mechanics_dir,
        run_id="zero-call-atomic-mechanics",
        serving_result=serving_path,
        tasks=tasks,
        adapter=_RealTaskFixtureAdapter(tasks),
    )
    assert mechanics_result["status"] == "mechanics_pass"
    mechanics_path = mechanics_dir / "RESULT.json"

    serving_verification = aug10.validate_serving_artifact(
        serving_path, tasks=tasks
    )
    mechanics_verification = aug10.validate_mechanics_artifact(
        mechanics_path, serving_result=serving_path, tasks=tasks
    )
    wrapper_path = tmp_path / "real-wrapper/RESULT.json"
    _write(
        wrapper_path,
        {
            "schema_version": 1,
            "interface_version": aug10.INTERFACE_VERSION,
            "status": "complete",
            "date": aug10.EXPECTED_DATE,
            "authorizes_development": True,
            "components": {
                "serving": aug10._component_record(
                    serving_path, serving_verification
                ),
                "mechanics": aug10._component_record(
                    mechanics_path, mechanics_verification
                ),
            },
        },
    )
    return wrapper_path, mechanics_path


def test_atomic_finalizer_runs_and_replays_every_real_downstream_component(
    tmp_path: Path,
) -> None:
    wrapper, mechanics_result = _real_authorized_wrapper(tmp_path)
    output_dir = tmp_path / "atomic-final-handoff"
    postprocess_dir = tmp_path / "atomic-postprocess"
    paid_calls = []

    def existing_paid_wrapper(**_) -> dict:
        paid_calls.append("validated_existing_wrapper")
        return json.loads(wrapper.read_text(encoding="utf-8"))

    result = final_handoff.run_final_handoff(
        output_dir=output_dir,
        wrapper_dir=wrapper.parent,
        serving_dir=tmp_path / "unused-serving",
        mechanics_dir=mechanics_result.parent,
        postprocess_dir=postprocess_dir,
        execute_runner=existing_paid_wrapper,
    )

    assert paid_calls == ["validated_existing_wrapper"]
    assert result["status"] == "handoff_complete"
    assert result["primary_disposition"] == "mechanics_pass"
    assert result["downstream_analyses_opened"] is True
    assert result["random_strategy_audit_opened"] is True
    assert result["model_calls_added_by_handoff"] == 0
    assert result["cost_usd_added_by_handoff"] == 0.0
    assert result["this_record_authorizes_development"] is False
    assert set(result["components"]) == {
        "paid_terminal",
        "postprocess",
        "random_strategy_control",
    }
    random_result = json.loads(
        (output_dir / "RANDOM_STRATEGY_CONTROL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert random_result["interface_version"] == random_control.INTERFACE_VERSION
    assert random_result["status"] == "random_strategy_control_audit_complete"
    assert random_result["task_count"] == 4
    assert random_result["random_draws_replayed_exactly"] is True
    assert random_result["stage_result_sha256"] == _sha256(mechanics_result)
    assert random_result["model_calls"] == 0
    assert random_result["cost_usd"] == 0.0

    replay = final_handoff.run_final_handoff(
        output_dir=output_dir,
        wrapper_dir=wrapper.parent,
        serving_dir=tmp_path / "unused-serving",
        mechanics_dir=mechanics_result.parent,
        postprocess_dir=postprocess_dir,
        execute_runner=lambda **_: pytest.fail("paid wrapper repeated"),
    )
    assert replay == result
    assert paid_calls == ["validated_existing_wrapper"]
