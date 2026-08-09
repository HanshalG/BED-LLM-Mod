#!/usr/bin/env python3
"""Run the zero-call downstream analyses for a terminal Bongard Aug 10 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_classical_suite_outcome as classical_suite
from scripts import bongard_openworld_compute_matched_control as compute_control
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_mechanics_disposition as disposition
from scripts import bongard_openworld_path_mediation as path_mediation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-aug10-postprocess-2"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_POSTPROCESS_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "f7d19ef3a8a48478aa30d4b63541e0c680f74641de70ed3120b25f203d888f5e"
)
COMPUTE_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_POSTPROCESS_COMPUTE_AMENDMENT_20260809.md"
)
COMPUTE_AMENDMENT_SHA256 = (
    "0e0a443b033b4ea5dcbe426dbf1820de9baa029ff2d78794d87e329f52cda66f"
)
BUDGET_BOUNDARY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_ACCOUNT_WIDE_BUDGET_BOUNDARY_CORRECTION_20260809.md"
)
BUDGET_BOUNDARY_AMENDMENT_SHA256 = (
    "3ea4f1805f5dbda20d8ac8cf31215fc92ad5bedccee40834573fa3762fec94de"
)
BOUND_IMPLEMENTATIONS = {
    "aug10_wrapper": (
        "scripts/bongard_openworld_luna_aug10_execute.py",
        "27b78be19f2e68e14335abecc5bfde0dcd197fbe6e6b895ff2581dafdc3cddbb",
    ),
    "mechanics_disposition": (
        "scripts/bongard_openworld_mechanics_disposition.py",
        "870c6fe5ba1a9dd09250193bc36e4bb108708a10dc07bc225dc075d596f4d8bb",
    ),
    "classical_suite": (
        "scripts/bongard_openworld_classical_suite_outcome.py",
        "8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032",
    ),
    "path_mediation": (
        "scripts/bongard_openworld_path_mediation.py",
        "1aef1c9eb90757bd31fec4beb077ddf79965e1a42b2715b4f7a6788e57e8b912",
    ),
    "compute_matched_control": (
        "scripts/bongard_openworld_compute_matched_control.py",
        "929eda107f8cb60caf4cd7363f07856e135adaa89f16e946edb710c1a93dfbba",
    ),
}
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_aug10_postprocess/"
    "bongard-openworld-aug10-postprocess-20260810"
)
DEFAULT_ARTIFACT = aug10.OUTPUT_DIR / "RESULT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_bound_implementations() -> dict[str, Any]:
    protocol_hash = _sha256(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("August 10 postprocess protocol changed")
    amendment_hash = _sha256(COMPUTE_AMENDMENT)
    if amendment_hash != COMPUTE_AMENDMENT_SHA256:
        raise ValueError("August 10 postprocess compute amendment changed")
    budget_amendment_hash = _sha256(BUDGET_BOUNDARY_AMENDMENT)
    if budget_amendment_hash != BUDGET_BOUNDARY_AMENDMENT_SHA256:
        raise ValueError("August 10 account-wide budget amendment changed")
    observed = {
        name: {"path": relative, "sha256": _sha256(REPO_ROOT / relative)}
        for name, (relative, _) in BOUND_IMPLEMENTATIONS.items()
    }
    expected = {
        name: {"path": relative, "sha256": expected_hash}
        for name, (relative, expected_hash) in BOUND_IMPLEMENTATIONS.items()
    }
    if observed != expected:
        raise ValueError(
            "bound August 10 postprocess implementation changed: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_hash},
        "compute_amendment": {
            "path": str(COMPUTE_AMENDMENT),
            "sha256": amendment_hash,
        },
        "budget_boundary_amendment": {
            "path": str(BUDGET_BOUNDARY_AMENDMENT),
            "sha256": budget_amendment_hash,
        },
        "implementations": observed,
    }


def _resolve_artifact_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("wrapper component artifact path is missing")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _mechanics_from_wrapper(wrapper_path: Path) -> Path:
    wrapper = _load(wrapper_path)
    if (
        wrapper.get("schema_version") != SCHEMA_VERSION
        or wrapper.get("interface_version") != aug10.INTERFACE_VERSION
        or wrapper.get("status") != "complete"
        or wrapper.get("date") != aug10.EXPECTED_DATE
        or wrapper.get("authorizes_development") is not True
    ):
        raise ValueError("downstream analyses require the complete authorized wrapper")
    components = wrapper.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "serving",
        "mechanics",
    }:
        raise ValueError("authorized wrapper component set changed")
    record = components["mechanics"]
    if not isinstance(record, Mapping):
        raise ValueError("authorized wrapper mechanics component is missing")
    path = _resolve_artifact_path(record.get("artifact"))
    if (
        record.get("verified") is not True
        or record.get("status") != "mechanics_pass"
        or not path.is_file()
        or _sha256(path) != record.get("artifact_sha256")
    ):
        raise ValueError("authorized wrapper mechanics binding changed")
    return path


def _validate_zero_call_component(
    *,
    path: Path,
    expected_interface: str,
    expected_status: str,
    stage_result: Path,
) -> dict[str, Any]:
    result = _load(path)
    recorded_stage_path = Path(str(result.get("stage_result_path", "")))
    if not recorded_stage_path.is_absolute():
        recorded_stage_path = REPO_ROOT / recorded_stage_path
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("interface_version") != expected_interface
        or result.get("status") != expected_status
        or result.get("stage") != "mechanics"
        or result.get("model_calls") != 0
        or not isinstance(result.get("cost_usd"), (int, float))
        or isinstance(result.get("cost_usd"), bool)
        or not math.isclose(float(result["cost_usd"]), 0.0, abs_tol=1e-15)
        or result.get("authorizes_paid_calls") is not False
        or recorded_stage_path.resolve() != stage_result.resolve()
        or result.get("stage_result_sha256") != _sha256(stage_result)
    ):
        raise ValueError(f"zero-call postprocess component is invalid: {path}")
    if expected_interface == classical_suite.INTERFACE_VERSION and (
        result.get("all_gates_pass") is not True
    ):
        raise ValueError("classical suite did not complete all gates")
    if expected_interface == path_mediation.INTERFACE_VERSION and (
        result.get("changes_claim_tier") is not False
    ):
        raise ValueError("path mediation changed the claim tier")
    return result


def _validate_compute_component_exact(
    *,
    path: Path,
    expected_interface: str,
    expected_status: str,
    stage_result: Path,
) -> dict[str, Any]:
    result = _validate_zero_call_component(
        path=path,
        expected_interface=expected_interface,
        expected_status=expected_status,
        stage_result=stage_result,
    )
    authorization = result.get("stage_authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("verified") is not True
    ):
        raise ValueError("compute-matched checkpoint lacks stage authorization")
    replay = compute_control.build_report(
        stage="mechanics",
        stage_result=_load(stage_result),
        authorization=authorization,
    )
    replay["protocol"] = {
        "path": str(compute_control.PROTOCOL),
        "sha256": compute_control.PROTOCOL_SHA256,
    }
    replay["stage_result_path"] = str(stage_result)
    replay["stage_result_sha256"] = _sha256(stage_result)
    replay["block_result_sha256"] = []
    if _canonical(result) != _canonical(replay):
        raise ValueError("compute-matched checkpoint does not exactly replay")
    return result


def _component_record(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "interface_version": result["interface_version"],
        "status": result["status"],
    }


def _load_or_run_component(
    *,
    path: Path,
    runner: Callable[..., dict[str, Any]],
    runner_kwargs: Mapping[str, Any],
    expected_interface: str,
    expected_status: str,
    stage_result: Path,
    validator: Callable[..., dict[str, Any]] = _validate_zero_call_component,
) -> dict[str, Any]:
    if not path.exists():
        runner(**runner_kwargs)
    if not path.is_file():
        raise ValueError(f"postprocess component omitted its output: {path}")
    return validator(
        path=path,
        expected_interface=expected_interface,
        expected_status=expected_status,
        stage_result=stage_result,
    )


def _validate_terminal_record(
    *,
    record: Mapping[str, Any],
    artifact_path: Path,
    bindings: Mapping[str, Any],
) -> None:
    input_record = record.get("input_artifact")
    cost = record.get("cost_usd")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("interface_version") != INTERFACE_VERSION
        or record.get("status")
        not in {
            "postprocess_complete",
            "terminal_disposition_only",
            "failed_closed",
        }
        or record.get("model_calls") != 0
        or not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or float(cost) != 0.0
        or record.get("authorizes_paid_calls") is not False
        or record.get("authorizes_rerun") is not False
        or record.get("this_record_authorizes_development") is not False
        or record.get("bindings") != bindings
        or not isinstance(input_record, Mapping)
        or Path(str(input_record.get("path", ""))).resolve()
        != artifact_path.resolve()
        or input_record.get("sha256") != _sha256(artifact_path)
    ):
        raise ValueError("banked August 10 postprocess record changed")
    status = record["status"]
    if status == "postprocess_complete":
        artifact_records = record.get("components")
        expected_names = {
            "disposition",
            "classical_suite",
            "path_mediation",
            "compute_matched_control",
        }
    elif status == "terminal_disposition_only":
        artifact_records = record.get("components")
        expected_names = {"disposition"}
    else:
        artifact_records = record.get("completed_artifacts")
        expected_names = {
            "disposition": set(),
            "classical_suite": {"disposition"},
            "path_mediation": {"disposition", "classical_suite"},
            "compute_matched_control": {
                "disposition",
                "classical_suite",
                "path_mediation",
            },
        }.get(record.get("failed_stage"))
    if not isinstance(artifact_records, Mapping):
        raise ValueError("banked postprocess component bindings are missing")
    if expected_names is None or set(artifact_records) != expected_names:
        raise ValueError("banked postprocess component set changed")
    for name, component in artifact_records.items():
        if not isinstance(component, Mapping):
            raise ValueError(f"banked component record is malformed: {name}")
        component_path = Path(str(component.get("path", "")))
        if not component_path.is_absolute():
            component_path = REPO_ROOT / component_path
        if (
            not component_path.is_file()
            or component.get("sha256") != _sha256(component_path)
        ):
            raise ValueError(f"banked postprocess component changed: {name}")


def _bank_failure(
    *,
    path: Path,
    artifact_path: Path,
    bindings: Mapping[str, Any],
    failed_stage: str,
    error: Exception,
    completed_artifacts: Mapping[str, Any],
    existing_wrapper_authorizes_development: bool,
) -> dict[str, Any]:
    failure = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "failed_closed",
        "date": aug10.EXPECTED_DATE,
        "input_artifact": {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
        },
        "bindings": dict(bindings),
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "completed_artifacts": dict(completed_artifacts),
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "existing_wrapper_authorizes_development": (
            existing_wrapper_authorizes_development
        ),
        "this_record_authorizes_development": False,
    }
    _write_once(path, failure)
    return failure


def run_postprocess(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    output_dir: Path = OUTPUT_DIR,
    implementation_verifier: Callable[[], dict[str, Any]] = (
        verify_bound_implementations
    ),
    classifier: Callable[[Path], dict[str, Any]] = disposition.classify_artifact,
    suite_runner: Callable[..., dict[str, Any]] = classical_suite.run_outcome,
    mediation_runner: Callable[..., dict[str, Any]] = path_mediation.run_report,
    compute_runner: Callable[..., dict[str, Any]] = compute_control.run_report,
    compute_validator: Callable[..., dict[str, Any]] = (
        _validate_compute_component_exact
    ),
) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    bindings = implementation_verifier()
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    disposition_path = output_dir / "MECHANICS_DISPOSITION.json"
    classical_path = output_dir / "CLASSICAL_SUITE_RESULT.json"
    mediation_path = output_dir / "PATH_MEDIATION_RESULT.json"
    compute_path = output_dir / "COMPUTE_MATCHED_CONTROL_RESULT.json"
    terminal = [path for path in (result_path, failure_path) if path.exists()]
    if len(terminal) > 1:
        raise RuntimeError("ambiguous August 10 postprocess terminal artifacts")
    if terminal:
        record = _load(terminal[0])
        _validate_terminal_record(
            record=record, artifact_path=artifact_path, bindings=bindings
        )
        return record

    failed_stage = "disposition"
    existing_authorization = False
    components: dict[str, Any] = {}
    try:
        fresh_disposition = classifier(artifact_path)
        if disposition_path.exists():
            if _load(disposition_path) != fresh_disposition:
                raise ValueError("banked mechanics disposition changed")
        else:
            _write_once(disposition_path, fresh_disposition)
        if fresh_disposition.get("status") != "valid_disposition":
            raise ValueError("terminal artifact has no valid mechanics disposition")
        existing_authorization = (
            fresh_disposition.get("existing_wrapper_authorizes_development") is True
        )
        downstream = (
            fresh_disposition.get("primary_category") == "mechanics_pass"
            and existing_authorization
        )
        components["disposition"] = _component_record(
            disposition_path, fresh_disposition
        )
        if downstream:
            mechanics_result = _mechanics_from_wrapper(artifact_path)
            failed_stage = "classical_suite"
            suite = _load_or_run_component(
                path=classical_path,
                runner=suite_runner,
                runner_kwargs={
                    "stage": "mechanics",
                    "result_path": mechanics_result,
                    "output_path": classical_path,
                    "wrapper_result": artifact_path,
                },
                expected_interface=classical_suite.INTERFACE_VERSION,
                expected_status="classical_suite_complete",
                stage_result=mechanics_result,
            )
            components["classical_suite"] = _component_record(
                classical_path, suite
            )
            failed_stage = "path_mediation"
            mediation = _load_or_run_component(
                path=mediation_path,
                runner=mediation_runner,
                runner_kwargs={
                    "stage": "mechanics",
                    "result_path": mechanics_result,
                    "output_path": mediation_path,
                    "wrapper_result": artifact_path,
                },
                expected_interface=path_mediation.INTERFACE_VERSION,
                expected_status="path_mediation_complete",
                stage_result=mechanics_result,
            )
            components["path_mediation"] = _component_record(
                mediation_path, mediation
            )
            failed_stage = "compute_matched_control"
            compute = _load_or_run_component(
                path=compute_path,
                runner=compute_runner,
                runner_kwargs={
                    "stage": "mechanics",
                    "result_path": mechanics_result,
                    "output_path": compute_path,
                    "wrapper_result": artifact_path,
                },
                expected_interface=compute_control.INTERFACE_VERSION,
                expected_status="compute_matched_control_audit_complete",
                stage_result=mechanics_result,
                validator=compute_validator,
            )
            components["compute_matched_control"] = _component_record(
                compute_path, compute
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": (
                "postprocess_complete" if downstream else "terminal_disposition_only"
            ),
            "date": aug10.EXPECTED_DATE,
            "input_artifact": {
                "path": str(artifact_path),
                "sha256": _sha256(artifact_path),
            },
            "bindings": bindings,
            "primary_disposition": fresh_disposition["primary_category"],
            "downstream_analyses_opened": downstream,
            "components": components,
            "model_calls": 0,
            "cost_usd": 0.0,
            "authorizes_paid_calls": False,
            "authorizes_rerun": False,
            "existing_wrapper_authorizes_development": existing_authorization,
            "this_record_authorizes_development": False,
        }
        _write_once(result_path, result)
        return result
    except Exception as exc:
        return _bank_failure(
            path=failure_path,
            artifact_path=artifact_path,
            bindings=bindings,
            failed_stage=failed_stage,
            error=exc,
            completed_artifacts=components,
            existing_wrapper_authorizes_development=existing_authorization,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_postprocess(
        artifact_path=args.artifact_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "failed_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
