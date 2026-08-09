from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bongard_openworld_aug10_postprocess as postprocess
from scripts import bongard_openworld_classical_suite_outcome as classical_suite
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_path_mediation as path_mediation
from scripts import bongard_openworld_vlm_bed as bed


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapper(tmp_path: Path) -> tuple[Path, Path]:
    mechanics = tmp_path / "mechanics/RESULT.json"
    _write(mechanics, {"status": "mechanics_pass"})
    wrapper = tmp_path / "wrapper/RESULT.json"
    _write(
        wrapper,
        {
            "schema_version": 1,
            "interface_version": aug10.INTERFACE_VERSION,
            "status": "complete",
            "date": aug10.EXPECTED_DATE,
            "authorizes_development": True,
            "components": {
                "serving": {"synthetic": True},
                "mechanics": {
                    "artifact": str(mechanics),
                    "artifact_sha256": _sha256(mechanics),
                    "status": "mechanics_pass",
                    "verified": True,
                },
            },
        },
    )
    return wrapper, mechanics


def _bindings() -> dict:
    return {
        "protocol": {"path": "protocol", "sha256": "protocol-sha"},
        "implementations": {"synthetic": {"sha256": "implementation-sha"}},
    }


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
                        (branch_key + int(observed[first_extra]) * 17 + image_index)
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
                    f"{('none' if first_extra is None else int(observed[first_extra]))} "
                    f"zero-call semantic rule {hypothesis_index + 1}"
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
        run_id="zero-call-real-serving",
        tasks=tasks,
        adapter=_RealTaskFixtureAdapter(tasks),
    )
    assert serving_result["status"] == "passed"
    serving_path = serving_dir / "RESULT.json"

    mechanics_dir = tmp_path / "real-mechanics"
    mechanics_result = mechanics.run_mechanics(
        output_dir=mechanics_dir,
        run_id="zero-call-real-mechanics",
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


def _pass_disposition() -> dict:
    return {
        "schema_version": 1,
        "interface_version": "synthetic-disposition",
        "status": "valid_disposition",
        "primary_category": "mechanics_pass",
        "existing_wrapper_authorizes_development": True,
        "this_record_authorizes_paid_calls": False,
        "this_record_authorizes_rerun": False,
    }


def _write_component(
    *,
    output_path: Path,
    result_path: Path,
    interface_version: str,
    status: str,
) -> dict:
    result = {
        "schema_version": 1,
        "interface_version": interface_version,
        "status": status,
        "stage": "mechanics",
        "stage_result_path": str(result_path),
        "stage_result_sha256": _sha256(result_path),
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
    }
    if interface_version == classical_suite.INTERFACE_VERSION:
        result["all_gates_pass"] = True
    else:
        result["changes_claim_tier"] = False
    _write(output_path, result)
    return result


def test_wrapper_pass_runs_suite_then_mediation_and_binds_zero_call_result(
    tmp_path: Path,
) -> None:
    wrapper, mechanics = _wrapper(tmp_path)
    calls = []

    def suite_runner(**kwargs):
        calls.append("classical_suite")
        assert kwargs["result_path"] == mechanics
        assert kwargs["wrapper_result"] == wrapper
        return _write_component(
            output_path=kwargs["output_path"],
            result_path=kwargs["result_path"],
            interface_version=classical_suite.INTERFACE_VERSION,
            status="classical_suite_complete",
        )

    def mediation_runner(**kwargs):
        calls.append("path_mediation")
        assert calls == ["classical_suite", "path_mediation"]
        return _write_component(
            output_path=kwargs["output_path"],
            result_path=kwargs["result_path"],
            interface_version=path_mediation.INTERFACE_VERSION,
            status="path_mediation_complete",
        )

    result = postprocess.run_postprocess(
        artifact_path=wrapper,
        output_dir=tmp_path / "postprocess",
        implementation_verifier=_bindings,
        classifier=lambda path: _pass_disposition(),
        suite_runner=suite_runner,
        mediation_runner=mediation_runner,
    )

    assert calls == ["classical_suite", "path_mediation"]
    assert result["status"] == "postprocess_complete"
    assert result["downstream_analyses_opened"] is True
    assert set(result["components"]) == {
        "disposition",
        "classical_suite",
        "path_mediation",
    }
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    assert result["existing_wrapper_authorizes_development"] is True
    assert result["this_record_authorizes_development"] is False


def test_real_authorized_mechanics_pass_runs_full_zero_call_handoff(
    tmp_path: Path,
) -> None:
    wrapper, mechanics_result = _real_authorized_wrapper(tmp_path)

    result = postprocess.run_postprocess(
        artifact_path=wrapper,
        output_dir=tmp_path / "real-postprocess",
    )

    assert result["status"] == "postprocess_complete"
    assert result["primary_disposition"] == "mechanics_pass"
    assert result["downstream_analyses_opened"] is True
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    disposition_result = json.loads(
        (tmp_path / "real-postprocess/MECHANICS_DISPOSITION.json").read_text()
    )
    assert disposition_result["input"]["hash_bound_by_wrapper"] is True
    assert disposition_result["existing_wrapper_authorizes_development"] is True
    for name, interface_version, status in (
        (
            "CLASSICAL_SUITE_RESULT.json",
            classical_suite.INTERFACE_VERSION,
            "classical_suite_complete",
        ),
        (
            "PATH_MEDIATION_RESULT.json",
            path_mediation.INTERFACE_VERSION,
            "path_mediation_complete",
        ),
    ):
        component = json.loads(
            (tmp_path / "real-postprocess" / name).read_text()
        )
        assert component["interface_version"] == interface_version
        assert component["status"] == status
        assert component["stage_result_sha256"] == _sha256(mechanics_result)
        assert component["model_calls"] == 0
        assert component["cost_usd"] == 0.0


def test_null_disposition_never_opens_endpoint_analyses(tmp_path: Path) -> None:
    artifact = tmp_path / "component/FAILURE.json"
    _write(artifact, {"status": "failed_closed"})
    disposition = {
        "schema_version": 1,
        "interface_version": "synthetic-disposition",
        "status": "valid_disposition",
        "primary_category": "transport_or_schema_inconclusive",
        "existing_wrapper_authorizes_development": False,
    }

    def bomb(**kwargs):
        del kwargs
        raise AssertionError("endpoint analysis opened after a null disposition")

    result = postprocess.run_postprocess(
        artifact_path=artifact,
        output_dir=tmp_path / "postprocess",
        implementation_verifier=_bindings,
        classifier=lambda path: disposition,
        suite_runner=bomb,
        mediation_runner=bomb,
    )

    assert result["status"] == "terminal_disposition_only"
    assert result["downstream_analyses_opened"] is False
    assert set(result["components"]) == {"disposition"}
    assert not (tmp_path / "postprocess/CLASSICAL_SUITE_RESULT.json").exists()
    assert not (tmp_path / "postprocess/PATH_MEDIATION_RESULT.json").exists()


def test_real_component_failure_classifier_stays_disposition_only(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "mechanics/FAILURE.json"
    _write(
        artifact,
        {
            "schema_version": 1,
            "interface_version": mechanics.INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": "TimeoutError",
            "error": "synthetic incomplete provider response",
        },
    )

    def bomb(**kwargs):
        del kwargs
        raise AssertionError("endpoint analysis opened after component failure")

    result = postprocess.run_postprocess(
        artifact_path=artifact,
        output_dir=tmp_path / "postprocess",
        implementation_verifier=_bindings,
        suite_runner=bomb,
        mediation_runner=bomb,
    )

    assert result["status"] == "terminal_disposition_only"
    assert result["primary_disposition"] == "transport_or_schema_inconclusive"
    assert result["existing_wrapper_authorizes_development"] is False
    assert result["downstream_analyses_opened"] is False


def test_downstream_failure_is_banked_once_and_never_runs_later_stage(
    tmp_path: Path,
) -> None:
    wrapper, _ = _wrapper(tmp_path)
    calls = []

    def reject_suite(**kwargs):
        del kwargs
        calls.append("classical_suite")
        raise RuntimeError("synthetic suite failure")

    def bomb_mediation(**kwargs):
        del kwargs
        raise AssertionError("mediation ran after suite failure")

    kwargs = {
        "artifact_path": wrapper,
        "output_dir": tmp_path / "postprocess",
        "implementation_verifier": _bindings,
        "classifier": lambda path: _pass_disposition(),
        "suite_runner": reject_suite,
        "mediation_runner": bomb_mediation,
    }
    failure = postprocess.run_postprocess(**kwargs)
    repeated = postprocess.run_postprocess(**kwargs)

    assert calls == ["classical_suite"]
    assert repeated == failure
    assert failure["status"] == "failed_closed"
    assert failure["failed_stage"] == "classical_suite"
    assert failure["error_type"] == "RuntimeError"
    assert failure["authorizes_paid_calls"] is False
    assert failure["authorizes_rerun"] is False
    assert failure["existing_wrapper_authorizes_development"] is True
    assert failure["this_record_authorizes_development"] is False
    assert not (tmp_path / "postprocess/RESULT.json").exists()
    assert (tmp_path / "postprocess/FAILURE.json").is_file()


def test_bound_protocol_and_implementations_match_current_tree() -> None:
    observed = postprocess.verify_bound_implementations()

    assert observed["protocol"]["sha256"] == postprocess.PROTOCOL_SHA256
    assert set(observed["implementations"]) == set(
        postprocess.BOUND_IMPLEMENTATIONS
    )
