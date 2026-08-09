#!/usr/bin/env python3
"""Run and bind the complete frozen Bongard August 10 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_aug10_postprocess as postprocess
from scripts import bongard_openworld_answer_signal_audit as answer_signal
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_random_strategy_control as random_control


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-aug10-final-handoff-2"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_FINAL_HANDOFF_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "76858ea571f478572f8067fa85084e0ea54798017b452d30caca33d9a0cc7e5a"
)
BOUND_IMPLEMENTATIONS = {
    "answer_signal_audit": (
        "scripts/bongard_openworld_answer_signal_audit.py",
        "e8b90ef239ad80fa0c2b0404c9ab1f5093f53ddedb9db338d2822e1577042ecd",
    ),
    "aug10_wrapper": (
        "scripts/bongard_openworld_luna_aug10_execute.py",
        "adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5",
    ),
    "postprocess_v2": (
        "scripts/bongard_openworld_aug10_postprocess.py",
        "883707739185f91fc7d60fe12661896e3a62c690b406fc993e5ccbdeffd69ce0",
    ),
    "random_strategy_control": (
        "scripts/bongard_openworld_random_strategy_control.py",
        "f99b68adb9b0db9d066ac2aa36a11351330ff476e6df430361431d07191f7441",
    ),
}
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_aug10_final_handoff/"
    "bongard-openworld-aug10-final-handoff-20260810"
)


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


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _component(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "status": value.get("status"),
    }


def verify_bindings() -> dict[str, Any]:
    protocol_hash = _sha256(PROTOCOL)
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("August 10 final-handoff protocol changed")
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
            "bound August 10 handoff implementation changed: "
            f"expected {expected}, observed {observed}"
        )
    postprocess.verify_bound_implementations()
    return {
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_hash},
        "implementations": observed,
    }


def preflight_final_handoff(
    *,
    output_dir: Path = OUTPUT_DIR,
    postprocess_dir: Path = postprocess.OUTPUT_DIR,
    aug10_preflight: Callable[..., dict[str, Any]] = aug10.preflight_aug10_sequence,
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
    **aug10_kwargs: Any,
) -> dict[str, Any]:
    bindings = binding_verifier()
    for label, path in (
        ("final handoff", output_dir),
        ("postprocess", postprocess_dir),
    ):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError(f"{label} path is not pristine: {path}")
    paid_preflight = aug10_preflight(**aug10_kwargs)
    if paid_preflight.get("status") != "ready_without_paid_calls":
        raise RuntimeError("August 10 paid preflight is not ready")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "execution_date": aug10.EXPECTED_DATE,
        "bindings": bindings,
        "paid_preflight": paid_preflight,
        "model_calls_made": 0,
        "files_written": 0,
    }


def _terminal_paid_artifact(
    *, wrapper_dir: Path, serving_dir: Path, mechanics_dir: Path
) -> Path | None:
    wrapper_result = wrapper_dir / "RESULT.json"
    if wrapper_result.is_file():
        return wrapper_result
    for path in (
        mechanics_dir / "FAILURE.json",
        serving_dir / "FAILURE.json",
    ):
        if path.is_file():
            return path
    return None


def _expected_random_report(
    *, mechanics_result: Path, wrapper_result: Path
) -> dict[str, Any]:
    authorization = random_control.stage_outcome.authorize_stage(
        stage="mechanics",
        result_path=mechanics_result,
        block_results=(),
        wrapper_result=wrapper_result,
    )
    report = random_control.build_report(
        stage="mechanics",
        stage_result=_load(mechanics_result),
        authorization=authorization,
    )
    report["protocol"] = {
        "path": str(random_control.PROTOCOL),
        "sha256": random_control.PROTOCOL_SHA256,
    }
    report["stage_result_path"] = str(mechanics_result)
    report["stage_result_sha256"] = _sha256(mechanics_result)
    report["block_result_sha256"] = []
    return report


def _load_or_run_answer_signal(
    *,
    output_path: Path,
    mechanics_result: Path,
    runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if output_path.exists():
        observed = _load(output_path)
        expected = answer_signal.build_report(mechanics_result=mechanics_result)
        if observed != expected:
            raise ValueError("banked August 10 answer-signal audit changed")
        return observed
    return runner(
        mechanics_result=mechanics_result,
        output_path=output_path,
    )


def _load_or_run_random(
    *,
    output_path: Path,
    mechanics_result: Path,
    wrapper_result: Path,
    runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if output_path.exists():
        observed = _load(output_path)
        expected = _expected_random_report(
            mechanics_result=mechanics_result,
            wrapper_result=wrapper_result,
        )
        if observed != expected:
            raise ValueError("banked August 10 random audit changed")
        return observed
    return runner(
        stage="mechanics",
        result_path=mechanics_result,
        output_path=output_path,
        wrapper_result=wrapper_result,
    )


def _validate_final(
    *, record: Mapping[str, Any], path: Path, bindings: Mapping[str, Any]
) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("interface_version") != INTERFACE_VERSION
        or record.get("status") not in {"handoff_complete", "failed_closed"}
        or record.get("bindings") != bindings
        or record.get("model_calls_added_by_handoff") != 0
        or record.get("cost_usd_added_by_handoff") != 0.0
        or record.get("authorizes_paid_calls") is not False
        or record.get("authorizes_rerun") is not False
        or record.get("this_record_authorizes_development") is not False
    ):
        raise ValueError("banked August 10 final handoff changed")
    components = record.get("components")
    if (
        not isinstance(components, Mapping)
        or "paid_terminal" not in components
        or "postprocess" not in components
    ):
        raise ValueError("banked final handoff components are incomplete")
    for component in components.values():
        if not isinstance(component, Mapping):
            raise ValueError("banked final handoff component is malformed")
        component_path = Path(str(component.get("path", "")))
        if (
            not component_path.is_file()
            or component.get("sha256") != _sha256(component_path)
        ):
            raise ValueError("banked final handoff component changed")
    if path.name == "RESULT.json" and record.get("status") != "handoff_complete":
        raise ValueError("final handoff result status changed")
    if path.name == "FAILURE.json" and record.get("status") != "failed_closed":
        raise ValueError("final handoff failure status changed")
    paid_path = Path(str(components["paid_terminal"]["path"]))
    postprocess_path = Path(str(components["postprocess"]["path"]))
    replayed_postprocess = postprocess.run_postprocess(
        artifact_path=paid_path,
        output_dir=postprocess_path.parent,
    )
    if replayed_postprocess != _load(postprocess_path):
        raise ValueError("banked final handoff postprocess did not replay")
    answer_component = components.get("answer_signal_audit")
    should_have_answer = False
    if paid_path.name == "RESULT.json":
        try:
            postprocess._mechanics_from_wrapper(paid_path)
        except ValueError:
            pass
        else:
            should_have_answer = True
    if (answer_component is not None) is not should_have_answer:
        raise ValueError("banked final handoff answer-signal scope changed")
    answer_passed = False
    if answer_component is not None:
        if paid_path.name != "RESULT.json":
            raise ValueError("answer-signal audit requires the terminal wrapper")
        mechanics_result = postprocess._mechanics_from_wrapper(paid_path)
        expected_answer = answer_signal.build_report(
            mechanics_result=mechanics_result
        )
        observed_answer = _load(Path(str(answer_component["path"])))
        if expected_answer != observed_answer:
            raise ValueError("banked final handoff answer-signal audit did not replay")
        answer_passed = observed_answer.get("status") == "answer_signal_valid"
    if (
        record.get("answer_signal_audit_opened") is not should_have_answer
        or record.get("answer_signal_status")
        != (
            observed_answer.get("status")
            if answer_component is not None
            else None
        )
        or record.get("answer_signal_authorizes_development") is not answer_passed
        or (
            should_have_answer
            and not answer_passed
            and record.get("status") != "failed_closed"
        )
    ):
        raise ValueError("banked final handoff answer-signal disposition changed")
    random_component = components.get("random_strategy_control")
    should_have_random = (
        replayed_postprocess.get("status") == "postprocess_complete"
        and answer_passed
    )
    if (random_component is not None) is not should_have_random:
        raise ValueError("banked final handoff random-control scope changed")
    if random_component is not None:
        if paid_path.name != "RESULT.json":
            raise ValueError("random-control handoff requires the terminal wrapper")
        mechanics_result = postprocess._mechanics_from_wrapper(paid_path)
        expected_random = _expected_random_report(
            mechanics_result=mechanics_result,
            wrapper_result=paid_path,
        )
        if expected_random != _load(Path(str(random_component["path"]))):
            raise ValueError("banked final handoff random audit did not replay")


def run_final_handoff(
    *,
    output_dir: Path = OUTPUT_DIR,
    wrapper_dir: Path = aug10.OUTPUT_DIR,
    serving_dir: Path = aug10.SERVING_DIR,
    mechanics_dir: Path = aug10.MECHANICS_DIR,
    postprocess_dir: Path = postprocess.OUTPUT_DIR,
    execute_runner: Callable[..., dict[str, Any]] = aug10.execute_aug10_sequence,
    postprocess_runner: Callable[..., dict[str, Any]] = postprocess.run_postprocess,
    answer_signal_runner: Callable[..., dict[str, Any]] = answer_signal.run_report,
    random_runner: Callable[..., dict[str, Any]] = random_control.run_report,
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
    execute_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = binding_verifier()
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    terminal = [path for path in (result_path, failure_path) if path.exists()]
    if len(terminal) > 1:
        raise RuntimeError("ambiguous August 10 final-handoff artifacts")
    if terminal:
        record = _load(terminal[0])
        _validate_final(record=record, path=terminal[0], bindings=bindings)
        return record

    execute_error: Exception | None = None
    try:
        execute_runner(**dict(execute_kwargs or {}))
    except Exception as exc:
        execute_error = exc
    paid_artifact = _terminal_paid_artifact(
        wrapper_dir=wrapper_dir,
        serving_dir=serving_dir,
        mechanics_dir=mechanics_dir,
    )
    if paid_artifact is None:
        if execute_error is not None:
            raise execute_error
        raise RuntimeError("August 10 paid wrapper omitted a terminal artifact")

    answer_report = None
    answer_path = None
    if paid_artifact.name == "RESULT.json":
        try:
            mechanics_result = postprocess._mechanics_from_wrapper(paid_artifact)
        except ValueError:
            pass
        else:
            answer_path = (
                mechanics_result.parent / "ANSWER_SIGNAL_AUDIT_RESULT.json"
            )
            answer_report = _load_or_run_answer_signal(
                output_path=answer_path,
                mechanics_result=mechanics_result,
                runner=answer_signal_runner,
            )

    processed = postprocess_runner(
        artifact_path=paid_artifact,
        output_dir=postprocess_dir,
    )
    postprocess_artifact = (
        postprocess_dir / "FAILURE.json"
        if processed.get("status") == "failed_closed"
        else postprocess_dir / "RESULT.json"
    )
    components = {
        "paid_terminal": _component(paid_artifact, _load(paid_artifact)),
        "postprocess": _component(postprocess_artifact, processed),
    }
    if answer_report is not None and answer_path is not None:
        components["answer_signal_audit"] = _component(
            answer_path, answer_report
        )
    random_report = None
    if processed.get("status") == "postprocess_complete":
        if paid_artifact.resolve() != (wrapper_dir / "RESULT.json").resolve():
            raise ValueError("endpoint analyses require the terminal wrapper result")
        mechanics_result = postprocess._mechanics_from_wrapper(paid_artifact)
    if (
        processed.get("status") == "postprocess_complete"
        and answer_report is not None
        and answer_report.get("status") == "answer_signal_valid"
    ):
        random_path = output_dir / "RANDOM_STRATEGY_CONTROL_RESULT.json"
        random_report = _load_or_run_random(
            output_path=random_path,
            mechanics_result=mechanics_result,
            wrapper_result=paid_artifact,
            runner=random_runner,
        )
        components["random_strategy_control"] = _component(
            random_path, random_report
        )

    answer_failed = (
        answer_report is not None
        and answer_report.get("status") != "answer_signal_valid"
    )
    failed = processed.get("status") == "failed_closed" or answer_failed
    record = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "failed_closed" if failed else "handoff_complete",
        "date": aug10.EXPECTED_DATE,
        "bindings": bindings,
        "components": components,
        "primary_disposition": (
            "answer_signal_not_above_regeneration_noise"
            if answer_failed
            else processed.get("primary_disposition")
        ),
        "downstream_analyses_opened": processed.get(
            "downstream_analyses_opened", False
        ),
        "random_strategy_audit_opened": random_report is not None,
        "answer_signal_audit_opened": answer_report is not None,
        "answer_signal_status": (
            answer_report.get("status") if answer_report is not None else None
        ),
        "answer_signal_authorizes_development": (
            answer_report is not None
            and answer_report.get("status") == "answer_signal_valid"
        ),
        "paid_execution_error_type": (
            type(execute_error).__name__ if execute_error is not None else None
        ),
        "existing_wrapper_authorizes_development": processed.get(
            "existing_wrapper_authorizes_development", False
        ),
        "model_calls_added_by_handoff": 0,
        "cost_usd_added_by_handoff": 0.0,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "this_record_authorizes_development": False,
    }
    _write_once(failure_path if failed else result_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight_final_handoff() if args.preflight else run_final_handoff()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
