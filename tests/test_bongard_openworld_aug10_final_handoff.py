from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_aug10_final_handoff as handoff


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_dir": tmp_path / "handoff",
        "wrapper_dir": tmp_path / "wrapper",
        "serving_dir": tmp_path / "serving",
        "mechanics_dir": tmp_path / "mechanics",
        "postprocess_dir": tmp_path / "postprocess",
    }


def _bindings() -> dict:
    return {"verified": True}


def test_bound_production_components_are_exact() -> None:
    result = handoff.verify_bindings()
    assert result["protocol"]["sha256"] == handoff.PROTOCOL_SHA256
    assert result["ordering_amendment"]["sha256"] == (
        handoff.ORDERING_AMENDMENT_SHA256
    )
    assert result["implementations"] == {
        name: {"path": relative, "sha256": expected}
        for name, (relative, expected) in handoff.BOUND_IMPLEMENTATIONS.items()
    }


def test_preflight_is_read_only_and_delegates_to_paid_preflight(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def paid_preflight(**kwargs):
        calls.append(kwargs)
        return {"status": "ready_without_paid_calls", "model_calls_made": 0}

    result = handoff.preflight_final_handoff(
        output_dir=paths["output_dir"],
        postprocess_dir=paths["postprocess_dir"],
        aug10_preflight=paid_preflight,
        binding_verifier=_bindings,
        output_dir_for_paid="paid-wrapper",
    )

    assert calls == [{"output_dir_for_paid": "paid-wrapper"}]
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not paths["output_dir"].exists()
    assert not paths["postprocess_dir"].exists()


def test_complete_handoff_orders_paid_postprocess_then_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    mechanics_result = paths["mechanics_dir"] / "RESULT.json"
    _write(mechanics_result, {"status": "mechanics_pass"})
    calls = []

    def execute_runner(**_):
        calls.append("paid")
        result = {"status": "complete"}
        _write(paths["wrapper_dir"] / "RESULT.json", result)
        return result

    def postprocess_runner(*, output_dir: Path, **_):
        calls.append("postprocess")
        result = {
            "status": "postprocess_complete",
            "primary_disposition": "mechanics_pass",
            "downstream_analyses_opened": True,
            "existing_wrapper_authorizes_development": True,
        }
        _write(output_dir / "RESULT.json", result)
        return result

    def answer_runner(*, output_path: Path, **_):
        calls.append("answer")
        result = {"status": "answer_signal_valid"}
        _write(output_path, result)
        return result

    def random_runner(*, output_path: Path, **_):
        calls.append("random")
        result = {"status": "random_strategy_control_audit_complete"}
        _write(output_path, result)
        return result

    monkeypatch.setattr(
        handoff.postprocess,
        "_mechanics_from_wrapper",
        lambda _: mechanics_result,
    )
    result = handoff.run_final_handoff(
        **paths,
        execute_runner=execute_runner,
        postprocess_runner=postprocess_runner,
        answer_signal_runner=answer_runner,
        random_runner=random_runner,
        binding_verifier=_bindings,
    )

    assert calls == ["paid", "answer", "postprocess", "random"]
    assert result["status"] == "handoff_complete"
    assert result["random_strategy_audit_opened"] is True
    assert result["model_calls_added_by_handoff"] == 0
    assert result["authorizes_paid_calls"] is False
    assert result["this_record_authorizes_development"] is False
    assert set(result["components"]) == {
        "paid_terminal",
        "postprocess",
        "answer_signal_audit",
        "random_strategy_control",
    }


def test_answer_signal_null_blocks_random_and_development(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    mechanics_result = paths["mechanics_dir"] / "RESULT.json"
    _write(mechanics_result, {"status": "mechanics_pass"})

    def execute_runner(**_):
        result = {"status": "complete", "authorizes_development": True}
        _write(paths["wrapper_dir"] / "RESULT.json", result)
        return result

    def answer_runner(*, output_path: Path, **_):
        result = {"status": "gated_null"}
        _write(output_path, result)
        return result

    monkeypatch.setattr(
        handoff.postprocess, "_mechanics_from_wrapper", lambda _: mechanics_result
    )
    result = handoff.run_final_handoff(
        **paths,
        execute_runner=execute_runner,
        postprocess_runner=lambda **_: pytest.fail(
            "postprocess opened after answer null"
        ),
        answer_signal_runner=answer_runner,
        random_runner=lambda **_: pytest.fail("random opened after answer null"),
        binding_verifier=_bindings,
    )

    assert result["status"] == "failed_closed"
    assert result["primary_disposition"] == (
        "answer_signal_not_above_regeneration_noise"
    )
    assert result["answer_signal_authorizes_development"] is False
    assert result["postprocess_opened"] is False
    assert result["downstream_analyses_opened"] is False
    assert result["random_strategy_audit_opened"] is False
    assert result["existing_wrapper_authorizes_development"] is True
    assert set(result["components"]) == {
        "paid_terminal",
        "answer_signal_audit",
    }
    assert not paths["postprocess_dir"].exists()

    monkeypatch.setattr(
        handoff.answer_signal, "build_report", lambda **_: {"status": "gated_null"}
    )
    monkeypatch.setattr(
        handoff.postprocess,
        "run_postprocess",
        lambda **_: pytest.fail("postprocess replayed after answer null"),
    )
    replayed = handoff.run_final_handoff(
        **paths,
        execute_runner=lambda **_: pytest.fail("paid stage repeated"),
        binding_verifier=_bindings,
    )
    assert replayed == result


def test_answer_signal_error_propagates_before_postprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    mechanics_result = paths["mechanics_dir"] / "RESULT.json"
    _write(mechanics_result, {"status": "mechanics_pass"})

    def execute_runner(**_):
        result = {"status": "complete", "authorizes_development": True}
        _write(paths["wrapper_dir"] / "RESULT.json", result)
        return result

    monkeypatch.setattr(
        handoff.postprocess, "_mechanics_from_wrapper", lambda _: mechanics_result
    )
    with pytest.raises(ValueError, match="malformed answer audit"):
        handoff.run_final_handoff(
            **paths,
            execute_runner=execute_runner,
            answer_signal_runner=lambda **_: (_ for _ in ()).throw(
                ValueError("malformed answer audit")
            ),
            postprocess_runner=lambda **_: pytest.fail(
                "postprocess opened after malformed answer audit"
            ),
            binding_verifier=_bindings,
        )

    assert not paths["postprocess_dir"].exists()
    assert not paths["output_dir"].exists()


def test_mechanics_null_never_opens_random_control(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    def execute_runner(**_):
        result = {"status": "complete", "authorizes_development": False}
        _write(paths["wrapper_dir"] / "RESULT.json", result)
        return result

    def postprocess_runner(*, output_dir: Path, **_):
        result = {
            "status": "terminal_disposition_only",
            "primary_disposition": "nonmyopic_opportunity_absent",
            "downstream_analyses_opened": False,
            "existing_wrapper_authorizes_development": False,
        }
        _write(output_dir / "RESULT.json", result)
        return result

    result = handoff.run_final_handoff(
        **paths,
        execute_runner=execute_runner,
        postprocess_runner=postprocess_runner,
        random_runner=lambda **_: pytest.fail("random control opened after null"),
        binding_verifier=_bindings,
    )

    assert result["status"] == "handoff_complete"
    assert result["random_strategy_audit_opened"] is False
    assert set(result["components"]) == {"paid_terminal", "postprocess"}


@pytest.mark.parametrize("failed_stage", ["serving", "mechanics"])
def test_banked_paid_failure_is_dispositioned_without_reissue(
    tmp_path: Path, failed_stage: str
) -> None:
    paths = _paths(tmp_path)
    failure_dir = paths[f"{failed_stage}_dir"]
    calls = []

    def execute_runner(**_):
        calls.append("paid")
        _write(
            failure_dir / "FAILURE.json",
            {"status": "failed_closed", "interface_version": failed_stage},
        )
        raise RuntimeError("bounded paid component failed")

    def postprocess_runner(*, artifact_path: Path, output_dir: Path):
        calls.append("postprocess")
        assert artifact_path == failure_dir / "FAILURE.json"
        result = {
            "status": "terminal_disposition_only",
            "primary_disposition": "transport_or_schema_inconclusive",
            "downstream_analyses_opened": False,
            "existing_wrapper_authorizes_development": False,
        }
        _write(output_dir / "RESULT.json", result)
        return result

    result = handoff.run_final_handoff(
        **paths,
        execute_runner=execute_runner,
        postprocess_runner=postprocess_runner,
        random_runner=lambda **_: pytest.fail("random control opened after failure"),
        binding_verifier=_bindings,
    )

    assert calls == ["paid", "postprocess"]
    assert result["status"] == "handoff_complete"
    assert result["paid_execution_error_type"] == "RuntimeError"
    assert result["random_strategy_audit_opened"] is False


def test_postprocess_failure_after_answer_pass_is_banked_and_never_opens_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    mechanics_result = paths["mechanics_dir"] / "RESULT.json"
    _write(mechanics_result, {"status": "mechanics_pass"})

    def execute_runner(**_):
        result = {"status": "complete"}
        _write(paths["wrapper_dir"] / "RESULT.json", result)
        return result

    def postprocess_runner(*, output_dir: Path, **_):
        result = {
            "status": "failed_closed",
            "primary_disposition": "mechanics_pass",
            "existing_wrapper_authorizes_development": True,
        }
        _write(output_dir / "FAILURE.json", result)
        return result

    def answer_runner(*, output_path: Path, **_):
        result = {"status": "answer_signal_valid"}
        _write(output_path, result)
        return result

    monkeypatch.setattr(
        handoff.postprocess,
        "_mechanics_from_wrapper",
        lambda _: mechanics_result,
    )

    result = handoff.run_final_handoff(
        **paths,
        execute_runner=execute_runner,
        postprocess_runner=postprocess_runner,
        answer_signal_runner=answer_runner,
        random_runner=lambda **_: pytest.fail("random opened after postprocess failure"),
        binding_verifier=_bindings,
    )

    assert result["status"] == "failed_closed"
    assert (paths["output_dir"] / "FAILURE.json").is_file()
    assert result["authorizes_rerun"] is False
    assert result["answer_signal_status"] == "answer_signal_valid"


def test_unbanked_paid_exception_does_not_create_handoff_artifact(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(RuntimeError, match="unexpected paid failure"):
        handoff.run_final_handoff(
            **paths,
            execute_runner=lambda **_: (_ for _ in ()).throw(
                RuntimeError("unexpected paid failure")
            ),
            postprocess_runner=lambda **_: pytest.fail("postprocess opened"),
            binding_verifier=_bindings,
        )

    assert not paths["output_dir"].exists()


def test_existing_final_replays_postprocess_and_random(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paid = paths["wrapper_dir"] / "RESULT.json"
    processed_path = paths["postprocess_dir"] / "RESULT.json"
    random_path = paths["output_dir"] / "RANDOM_STRATEGY_CONTROL_RESULT.json"
    answer_path = paths["output_dir"] / "ANSWER_SIGNAL_AUDIT_RESULT.json"
    mechanics = paths["mechanics_dir"] / "RESULT.json"
    processed = {"status": "postprocess_complete"}
    random_report = {"status": "random_strategy_control_audit_complete"}
    answer_report = {"status": "answer_signal_valid"}
    _write(paid, {"status": "complete"})
    _write(processed_path, processed)
    _write(random_path, random_report)
    _write(answer_path, answer_report)
    _write(mechanics, {"status": "mechanics_pass"})
    record = {
        "schema_version": handoff.SCHEMA_VERSION,
        "interface_version": handoff.INTERFACE_VERSION,
        "status": "handoff_complete",
        "bindings": _bindings(),
        "components": {
            "paid_terminal": handoff._component(paid, {"status": "complete"}),
            "postprocess": handoff._component(processed_path, processed),
            "answer_signal_audit": handoff._component(
                answer_path, answer_report
            ),
            "random_strategy_control": handoff._component(
                random_path, random_report
            ),
        },
        "model_calls_added_by_handoff": 0,
        "cost_usd_added_by_handoff": 0.0,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "this_record_authorizes_development": False,
        "answer_signal_audit_opened": True,
        "answer_signal_status": "answer_signal_valid",
        "answer_signal_authorizes_development": True,
        "postprocess_opened": True,
    }
    _write(paths["output_dir"] / "RESULT.json", record)
    replay_calls = []

    def replay_postprocess(**_):
        replay_calls.append("postprocess")
        return processed

    monkeypatch.setattr(handoff.postprocess, "run_postprocess", replay_postprocess)
    monkeypatch.setattr(
        handoff.postprocess, "_mechanics_from_wrapper", lambda _: mechanics
    )
    monkeypatch.setattr(
        handoff.answer_signal, "build_report", lambda **_: answer_report
    )
    monkeypatch.setattr(
        handoff, "_expected_random_report", lambda **_: random_report
    )
    result = handoff.run_final_handoff(
        **paths,
        execute_runner=lambda **_: pytest.fail("paid stage repeated"),
        binding_verifier=_bindings,
    )

    assert result == record
    assert replay_calls == ["postprocess"]


def test_existing_final_rejects_changed_random_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paid = paths["wrapper_dir"] / "RESULT.json"
    processed_path = paths["postprocess_dir"] / "RESULT.json"
    random_path = paths["output_dir"] / "RANDOM_STRATEGY_CONTROL_RESULT.json"
    _write(paid, {"status": "complete"})
    _write(processed_path, {"status": "postprocess_complete"})
    _write(random_path, {"status": "original"})
    record = {
        "schema_version": handoff.SCHEMA_VERSION,
        "interface_version": handoff.INTERFACE_VERSION,
        "status": "handoff_complete",
        "bindings": _bindings(),
        "components": {
            "paid_terminal": handoff._component(paid, {}),
            "postprocess": handoff._component(processed_path, {}),
            "random_strategy_control": handoff._component(random_path, {}),
        },
        "model_calls_added_by_handoff": 0,
        "cost_usd_added_by_handoff": 0.0,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "this_record_authorizes_development": False,
    }
    _write(paths["output_dir"] / "RESULT.json", record)
    _write(random_path, {"status": "changed"})
    monkeypatch.setattr(
        handoff.postprocess,
        "run_postprocess",
        lambda **_: {"status": "postprocess_complete"},
    )

    with pytest.raises(ValueError, match="component changed"):
        handoff.run_final_handoff(**paths, binding_verifier=_bindings)
