from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bongard_openworld_aug10_postprocess as postprocess
from scripts import bongard_openworld_classical_suite_outcome as classical_suite
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_path_mediation as path_mediation


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
