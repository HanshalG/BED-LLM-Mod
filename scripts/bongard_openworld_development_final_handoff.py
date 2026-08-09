#!/usr/bin/env python3
"""Finalize every zero-call artifact for Bongard Development64."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_classical_suite_outcome as classical_suite
from scripts import bongard_openworld_compute_matched_control as compute_control
from scripts import bongard_openworld_development_daily_handoff as paired_daily
from scripts import bongard_openworld_luna_development32_daily_execute as main_daily
from scripts import bongard_openworld_luna_development_claim_finalize as claim_finalize
from scripts import bongard_openworld_luna_paper_fragment as luna_fragment
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_paper_with_classical_suite as paper
from scripts import bongard_openworld_path_mediation as path_mediation
from scripts import bongard_openworld_random_strategy_control as random_control


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-development-final-handoff-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_DEVELOPMENT_FINAL_HANDOFF_PROTOCOL_20260809.md"
)
PROTOCOL_SHA256 = (
    "474dc8efbf175abf59da8e3f6f12889405e2de135a6d92dcf024d1534eb2fbac"
)
BOUND_IMPLEMENTATIONS = {
    "paired_daily_handoff": (
        "scripts/bongard_openworld_development_daily_handoff.py",
        "0c94b16aaf12a4f509b051b49ecd256260b2b254c9c5b62b014baac7a6265027",
    ),
    "claim_finalizer": (
        "scripts/bongard_openworld_luna_development_claim_finalize.py",
        "89fd4f64587908fbeaaffe6aa0c4db28824ca83cc10e5ca020d03ef89d2a24f2",
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
    "random_strategy_control": (
        "scripts/bongard_openworld_random_strategy_control.py",
        "f99b68adb9b0db9d066ac2aa36a11351330ff476e6df430361431d07191f7441",
    ),
    "paper_wrapper_v6": (
        "scripts/bongard_openworld_paper_with_classical_suite.py",
        "57fa47391f85cf66ec194a93998f2bf50a68da674709a84f58115f1b7bd5b151",
    ),
}
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_development_final_handoff/"
    "development64"
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


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
        raise ValueError("development final-handoff protocol changed")
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
            "bound development final-handoff implementation changed: "
            f"expected {expected}, observed {observed}"
        )
    paper.verify_bound_implementations()
    return {
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_hash},
        "implementations": observed,
    }


def _default_blocks() -> list[Path]:
    return [
        main_daily.BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in development.BLOCK_ORDER
    ]


def _default_paired_paths() -> dict[str, Path]:
    return {
        block_id: paired_daily.OUTPUT_DIRS[block_id] / "RESULT.json"
        for block_id in development.BLOCK_ORDER
    }


def _validate_paired_days(
    *,
    paired_paths: Mapping[str, Path],
    validator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    records = {}
    for block_id in development.BLOCK_ORDER:
        path = paired_paths[block_id]
        if not path.is_file():
            raise FileNotFoundError(f"paired development block {block_id} is missing")
        record = validator(block_id=block_id)
        if (
            record.get("status") != "paired_daily_complete"
            or record.get("block_id") != block_id
        ):
            raise ValueError(f"paired development block {block_id} is incomplete")
        if _canonical(record) != _canonical(_load(path)):
            raise ValueError(f"paired development block {block_id} changed")
        records[block_id] = _component(path, record)
    return records


def preflight_final_handoff(
    *,
    output_dir: Path = OUTPUT_DIR,
    paired_paths: Mapping[str, Path] | None = None,
    combined_result: Path = main_daily.COMBINED_RESULT,
    paired_validator: Callable[..., dict[str, Any]] = paired_daily.run_daily_handoff,
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise RuntimeError(f"development final-handoff path is not pristine: {output_dir}")
    bindings = binding_verifier()
    paths = dict(paired_paths or _default_paired_paths())
    if set(paths) != set(development.BLOCK_ORDER):
        raise ValueError("development final preflight requires four paired blocks")
    completed = {}
    waiting_for = None
    for block_id in development.BLOCK_ORDER:
        result_path = paths[block_id]
        failure_path = result_path.with_name("FAILURE.json")
        if failure_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": f"blocked_by_block_{block_id}_failure",
                "bindings": bindings,
                "model_calls_made": 0,
                "files_written": 0,
            }
        if not result_path.is_file():
            waiting_for = block_id
            break
        record = paired_validator(block_id=block_id)
        if _canonical(record) != _canonical(_load(result_path)):
            raise ValueError(f"paired block {block_id} did not replay")
        completed[block_id] = _component(result_path, record)
    status = (
        "ready_without_model_calls"
        if waiting_for is None and combined_result.is_file()
        else f"waiting_for_block_{waiting_for or 'd_combined'}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "bindings": bindings,
        "completed_paired_blocks": completed,
        "combined_result": str(combined_result),
        "model_calls_made": 0,
        "files_written": 0,
    }


def _load_or_replay_json(
    *,
    path: Path,
    runner: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    if not path.exists():
        return runner(path)
    saved = _load(path)
    with tempfile.TemporaryDirectory(prefix="bongard-development-replay-") as tmp:
        replay_path = Path(tmp) / path.name
        replay = runner(replay_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError(f"saved development component did not replay: {path.name}")
    return saved


def _paper_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output,
        output.with_suffix(".json"),
        output.with_name(luna_fragment.HEADLINE_FILENAME),
    )


def _load_or_replay_paper(
    *, output: Path, runner: Callable[[Path], dict[str, Any]]
) -> dict[str, Any]:
    paths = _paper_paths(output)
    present = [path.is_file() for path in paths]
    if not any(present):
        return runner(output)
    if not all(present):
        raise ValueError("development paper artifact is partial")
    saved = [path.read_bytes() for path in paths]
    saved_result = {
        "status": "written_with_mandatory_classical_compute_random_and_mediation_suites",
        "stage": "development",
        "claim_tier": _load(paths[1])["claim_tier"],
        "tex_path": str(output),
        "tex_sha256": _sha256(paths[0]),
        "headline_path": str(paths[2]),
        "headline_sha256": _sha256(paths[2]),
        "metadata_path": str(paths[1]),
        "metadata_sha256": _sha256(paths[1]),
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    with tempfile.TemporaryDirectory(prefix="bongard-paper-replay-") as tmp:
        replay_output = Path(tmp) / output.name
        replay = runner(replay_output)
        replay_bytes = [path.read_bytes() for path in _paper_paths(replay_output)]
    if saved != replay_bytes:
        raise ValueError("saved development paper did not replay byte-identically")
    for key in (
        "status",
        "stage",
        "claim_tier",
        "tex_sha256",
        "headline_sha256",
        "metadata_sha256",
        "model_calls",
        "cost_usd",
    ):
        if saved_result[key] != replay[key]:
            raise ValueError("saved development paper replay summary changed")
    return saved_result


def _validate_terminal_record(
    *, record: Mapping[str, Any], path: Path, bindings: Mapping[str, Any]
) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("interface_version") != INTERFACE_VERSION
        or record.get("status") not in {"development_handoff_complete", "failed_closed"}
        or record.get("bindings") != bindings
        or record.get("model_calls") != 0
        or float(record.get("cost_usd", -1.0)) != 0.0
        or record.get("authorizes_paid_calls") is not False
        or record.get("authorizes_rerun") is not False
        or record.get("this_record_authorizes_confirmation") is not False
    ):
        raise ValueError("banked development final handoff changed")
    components = record.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("banked development final components are missing")
    for component in components.values():
        component_path = Path(str(component.get("path", "")))
        if not component_path.is_file() or component.get("sha256") != _sha256(
            component_path
        ):
            raise ValueError("banked development final component changed")
    expected_status = (
        "development_handoff_complete" if path.name == "RESULT.json" else "failed_closed"
    )
    if record.get("status") != expected_status:
        raise ValueError("development final terminal filename disagrees with status")
    if expected_status == "failed_closed" and record.get("failed_stage") not in {
        "paired_daily_handoffs",
        "claim_finalization",
        "classical_suite",
        "path_mediation",
        "compute_matched_control",
        "random_strategy_control",
        "paper",
    }:
        raise ValueError("banked development failure stage is invalid")


def _return_replayed_failure(
    *,
    stage: str,
    existing_failure: Mapping[str, Any] | None,
    components: Mapping[str, Any],
) -> dict[str, Any] | None:
    if existing_failure is None or existing_failure.get("failed_stage") != stage:
        return None
    if _canonical(existing_failure.get("components")) != _canonical(components):
        raise ValueError("banked development failure prefix did not replay")
    return dict(existing_failure)


def run_final_handoff(
    *,
    output_dir: Path = OUTPUT_DIR,
    combined_result: Path = main_daily.COMBINED_RESULT,
    block_results: Sequence[Path] | None = None,
    paired_paths: Mapping[str, Path] | None = None,
    claim_report_path: Path = claim_finalize.CLAIM_REPORT,
    paired_validator: Callable[..., dict[str, Any]] = paired_daily.run_daily_handoff,
    claim_runner: Callable[..., dict[str, Any]] = claim_finalize.finalize_development_claim,
    classical_runner: Callable[..., dict[str, Any]] = classical_suite.run_outcome,
    mediation_runner: Callable[..., dict[str, Any]] = path_mediation.run_report,
    compute_runner: Callable[..., dict[str, Any]] = compute_control.run_report,
    random_runner: Callable[..., dict[str, Any]] = random_control.run_report,
    paper_runner: Callable[..., dict[str, Any]] = paper.write_combined_fragment,
    binding_verifier: Callable[[], dict[str, Any]] = verify_bindings,
) -> dict[str, Any]:
    bindings = binding_verifier()
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    terminal = [path for path in (result_path, failure_path) if path.exists()]
    if len(terminal) > 1:
        raise RuntimeError("ambiguous development final-handoff artifacts")
    existing_result = None
    existing_failure = None
    if terminal:
        record = _load(terminal[0])
        _validate_terminal_record(record=record, path=terminal[0], bindings=bindings)
        if terminal[0] == failure_path:
            existing_failure = record
        else:
            existing_result = record

    blocks = list(block_results or _default_blocks())
    if len(blocks) != len(development.BLOCK_ORDER):
        raise ValueError("development final handoff requires four block results")
    paired = dict(paired_paths or _default_paired_paths())
    components: dict[str, Any] = {}
    failed_stage = "paired_daily_handoffs"
    try:
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        paired_records = _validate_paired_days(
            paired_paths=paired, validator=paired_validator
        )
        components.update(
            {f"paired_{block_id}": value for block_id, value in paired_records.items()}
        )
        if not combined_result.is_file():
            raise FileNotFoundError("combined development result is missing")

        failed_stage = "claim_finalization"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        claim_summary_path = output_dir / "CLAIM_FINALIZATION.json"

        def run_claim(path: Path) -> dict[str, Any]:
            value = claim_runner(
                combined_result=combined_result,
                claim_report_path=claim_report_path,
                block_results=dict(zip(development.BLOCK_ORDER, blocks, strict=True)),
            )
            _write_once(path, value)
            return value

        claim = _load_or_replay_json(path=claim_summary_path, runner=run_claim)
        components["claim_finalization"] = _component(claim_summary_path, claim)
        components["claim_report"] = _component(
            claim_report_path, _load(claim_report_path)
        )

        shared = {
            "stage": "development",
            "result_path": combined_result,
            "block_results": blocks,
        }
        failed_stage = "classical_suite"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        classical_path = output_dir / "CLASSICAL_SUITE_RESULT.json"
        classical = _load_or_replay_json(
            path=classical_path,
            runner=lambda path: classical_runner(output_path=path, **shared),
        )
        components["classical_suite"] = _component(classical_path, classical)

        failed_stage = "path_mediation"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        mediation_path = output_dir / "PATH_MEDIATION_RESULT.json"
        mediation = _load_or_replay_json(
            path=mediation_path,
            runner=lambda path: mediation_runner(output_path=path, **shared),
        )
        components["path_mediation"] = _component(mediation_path, mediation)

        failed_stage = "compute_matched_control"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        compute_path = output_dir / "COMPUTE_MATCHED_CONTROL_RESULT.json"
        compute = _load_or_replay_json(
            path=compute_path,
            runner=lambda path: compute_runner(output_path=path, **shared),
        )
        components["compute_matched_control"] = _component(compute_path, compute)

        failed_stage = "random_strategy_control"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        random_path = output_dir / "RANDOM_STRATEGY_CONTROL_RESULT.json"
        random_report = _load_or_replay_json(
            path=random_path,
            runner=lambda path: random_runner(output_path=path, **shared),
        )
        components["random_strategy_control"] = _component(
            random_path, random_report
        )

        failed_stage = "paper"
        replayed_failure = _return_replayed_failure(
            stage=failed_stage,
            existing_failure=existing_failure,
            components=components,
        )
        if replayed_failure is not None:
            return replayed_failure
        paper_output = output_dir / "bongard_openworld_result.tex"

        def render(output: Path) -> dict[str, Any]:
            return paper_runner(
                stage="development",
                output=output,
                classical_suite_path=classical_path,
                compute_audit_path=compute_path,
                random_audit_path=random_path,
                mediation_path=mediation_path,
                claim_report_path=claim_report_path,
                combined_result=combined_result,
                block_results=blocks,
            )

        paper_result = _load_or_replay_paper(output=paper_output, runner=render)
        for name, path in zip(
            ("paper_tex", "paper_metadata", "paper_headline"),
            _paper_paths(paper_output),
            strict=True,
        ):
            components[name] = _component(path, paper_result)
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "development_handoff_complete",
            "bindings": bindings,
            "components": components,
            "claim_tier": claim["claim_tier"],
            "existing_claim_authorizes_confirmation": (
                claim.get("confirmation_authorization") is not None
            ),
            "all_dispositions_rendered_identically": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "authorizes_paid_calls": False,
            "authorizes_rerun": False,
            "this_record_authorizes_confirmation": False,
        }
        if existing_failure is not None:
            raise ValueError("banked development failure stage was not reached")
        if existing_result is not None:
            if _canonical(existing_result) != _canonical(result):
                raise ValueError("banked development final result did not replay")
            return existing_result
        _write_once(result_path, result)
        return result
    except Exception as exc:
        if existing_result is not None or existing_failure is not None:
            raise
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "bindings": bindings,
            "failed_stage": failed_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "components": components,
            "model_calls": 0,
            "cost_usd": 0.0,
            "authorizes_paid_calls": False,
            "authorizes_rerun": False,
            "this_record_authorizes_confirmation": False,
        }
        _write_once(failure_path, failure)
        return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight_final_handoff() if args.preflight else run_final_handoff()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
