#!/usr/bin/env python3
"""Classify a banked Bongard serving/mechanics outcome without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-mechanics-disposition-1"

TRANSPORT_GATES = (
    "transport_usage_counts_are_nonnegative_integers",
    "exact_expected_accepted_requests",
    "http_attempts_equal_accepted_plus_retries",
    "total_retries_within_preregistered_bound",
    "provider_error_retries_are_bounded_subset",
)

SERVING_GATE_CATEGORIES = {
    "transport_or_schema_inconclusive": (
        *TRANSPORT_GATES,
        "zero_reasoning_tokens",
        "zero_forced_exits",
        "all_10_strict_schemas_parse",
        "all_responses_have_exact_unique_hypothesis_support",
        "cost_at_most_0_25",
    ),
    "integrity_or_leakage_failure": (
        "all_history_conditioned_weights_have_finite_positive_mass_and_entropy",
        "all_multimodal_prompts_hide_bound_source_truth",
    ),
    "predictive_belief_invalid": (
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes",
    ),
    "path_dependence_absent": (
        "all_four_simulated_branch_pairs_change_unobserved_beliefs",
    ),
    "nonmyopic_opportunity_absent": (
        "both_roots_have_nonzero_nonidentical_candidate_eig",
    ),
    "endpoint_saturated": (),
}

MECHANICS_GATE_CATEGORIES = {
    "transport_or_schema_inconclusive": (
        *TRANSPORT_GATES,
        "zero_reasoning_tokens",
        "zero_forced_exits",
        "all_root_branch_and_final_responses_parse",
        "cost_at_most_1_75",
    ),
    "integrity_or_leakage_failure": (
        "serving_result_independently_replays",
        "exact_paired_seed_and_prompt_difference_accounting",
        "terminal_histories_use_task_level_common_random_numbers",
        "all_scores_are_finite_and_executable",
        "all_policies_use_endpoint_predictive_information_gain",
        "fixed_score_dynamic_update_exactly_matches_fixed_first_and_dynamic_second",
        "history_blind_update_matched_first_exactly_matches_dynamic_first",
        "shuffled_control_exactly_permutes_complete_continuation_values",
        "all_distinct_all_action_final_supports_generated_once_and_mapped",
        "all_eight_realized_first_action_continuations_are_scored",
        "all_ranking_fidelity_diagnostics_are_finite",
        "all_endpoint_metrics_are_finite",
        "all_prompts_hide_bound_source_truth",
    ),
    "predictive_belief_invalid": (
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes",
        "terminal_beliefs_retain_both_queried_labels_better_than_constant_half",
        "root_candidate_brier_beats_constant_half",
    ),
    "path_dependence_absent": (
        "at_least_24_of_32_branch_pairs_change_unobserved_beliefs",
    ),
    "nonmyopic_opportunity_absent": (
        "dynamic_depth2_changes_at_least_one_myopic_first_action",
        "dynamic_action_change_clears_numerical_tie_margin",
        "dynamic_and_history_blind_change_a_nontied_first_action",
        "dynamic_and_matched_history_blind_update_change_at_least_one_second_action",
        "at_least_two_controls_have_a_distinct_final_history",
    ),
    "endpoint_saturated": ("myopic_endpoint_is_not_saturated",),
}

CATEGORY_PRIORITY = tuple(MECHANICS_GATE_CATEGORIES)

CATEGORY_ACTIONS = {
    "transport_or_schema_inconclusive": (
        "Bank the run as infrastructure/schema inconclusive. Diagnose transport or "
        "the response interface offline; do not infer a semantic null or rerun this run."
    ),
    "integrity_or_leakage_failure": (
        "Bank the run as scientifically uninterpretable. Repair the deterministic "
        "instrument in a separately preregistered future design before any new model run."
    ),
    "predictive_belief_invalid": (
        "The LLM predictive belief interface failed its truth-anchored validity gate. "
        "A future experiment must change the belief model/interface, not tune the planner."
    ),
    "path_dependence_absent": (
        "The generated belief transition did not materially depend on simulated history. "
        "Do not open development; use a prospectively new source or belief interface."
    ),
    "nonmyopic_opportunity_absent": (
        "The mechanics tree exposed no robust non-myopic action opportunity. More tasks "
        "under this frozen mechanism cannot establish the first-link claim."
    ),
    "endpoint_saturated": (
        "The myopic endpoint is saturated. Define a prospectively harder endpoint/task "
        "split before another experiment; do not reinterpret this run."
    ),
}


class DispositionValidationError(ValueError):
    """Raised when an input artifact cannot support a scientific disposition."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispositionValidationError(f"artifact is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DispositionValidationError("artifact root must be a JSON object")
    return value


def _expected_gates(categories: Mapping[str, Sequence[str]]) -> set[str]:
    names = {name for values in categories.values() for name in values}
    if sum(len(values) for values in categories.values()) != len(names):
        raise RuntimeError("disposition gate categories overlap")
    return names | {"all_pass"}


SERVING_GATES = _expected_gates(SERVING_GATE_CATEGORIES)
MECHANICS_GATES = _expected_gates(MECHANICS_GATE_CATEGORIES)


def _validate_gate_record(
    result: Mapping[str, Any],
    *,
    artifact_type: str,
    interface_version: str,
    pass_status: str,
    categories: Mapping[str, Sequence[str]],
) -> dict[str, bool]:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise DispositionValidationError(f"{artifact_type} schema version changed")
    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise DispositionValidationError(f"{artifact_type} protocol is missing")
    if protocol.get("interface_version") != interface_version:
        raise DispositionValidationError(f"{artifact_type} interface version changed")
    status = result.get("status")
    if status not in {pass_status, "gated_null"}:
        raise DispositionValidationError(f"{artifact_type} status is invalid")
    gates = result.get("gates")
    if not isinstance(gates, dict):
        raise DispositionValidationError(f"{artifact_type} gates are missing")
    expected = _expected_gates(categories)
    if set(gates) != expected:
        missing = sorted(expected - set(gates))
        extra = sorted(set(gates) - expected)
        raise DispositionValidationError(
            f"{artifact_type} gate schema changed; missing={missing}, extra={extra}"
        )
    if any(type(value) is not bool for value in gates.values()):
        raise DispositionValidationError(f"{artifact_type} gates must be booleans")
    derived_all_pass = all(value for name, value in gates.items() if name != "all_pass")
    if gates["all_pass"] is not derived_all_pass:
        raise DispositionValidationError(f"{artifact_type} all_pass is inconsistent")
    if (status == pass_status) is not derived_all_pass:
        raise DispositionValidationError(f"{artifact_type} status contradicts its gates")
    return dict(gates)


def _validate_component(
    record: Mapping[str, Any],
    *,
    expected_statuses: set[str],
) -> tuple[Path, dict[str, Any]]:
    expected_fields = {
        "artifact",
        "artifact_sha256",
        "raw_responses_sha256",
        "status",
        "cost_usd",
        "verified",
    }
    if set(record) != expected_fields:
        raise DispositionValidationError("wrapper component schema changed")
    if record.get("verified") is not True:
        raise DispositionValidationError("wrapper component was not independently verified")
    if record.get("status") not in expected_statuses:
        raise DispositionValidationError("wrapper component status is invalid")
    cost = record.get("cost_usd")
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or cost < 0
    ):
        raise DispositionValidationError("wrapper component cost is invalid")
    artifact_text = record.get("artifact")
    if not isinstance(artifact_text, str) or not artifact_text:
        raise DispositionValidationError("wrapper component artifact path is invalid")
    path = Path(artifact_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file() or _sha256(path) != record.get("artifact_sha256"):
        raise DispositionValidationError("wrapper component artifact hash changed")
    result = _load(path)
    raw_hash = result.get("raw_responses_sha256")
    if raw_hash != record.get("raw_responses_sha256"):
        raise DispositionValidationError("wrapper component raw-response binding changed")
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if not raw_path.is_file() or _sha256(raw_path) != raw_hash:
        raise DispositionValidationError("wrapper component raw responses changed")
    return path, result


def _validate_raw_binding(path: Path, result: Mapping[str, Any]) -> None:
    raw_hash = result.get("raw_responses_sha256")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise DispositionValidationError("result raw-response hash is missing")
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if not raw_path.is_file() or _sha256(raw_path) != raw_hash:
        raise DispositionValidationError("result raw responses changed")


def _classify_gates(
    *,
    artifact_type: str,
    source_path: Path,
    source_sha256: str,
    gates: Mapping[str, bool],
    categories: Mapping[str, Sequence[str]],
    existing_wrapper_authorizes_development: bool = False,
    hash_bound_by_wrapper: bool = False,
) -> dict[str, Any]:
    failed = sorted(name for name, passed in gates.items() if name != "all_pass" and not passed)
    if not failed:
        category = (
            "mechanics_pass"
            if artifact_type == "mechanics_result"
            else "serving_pass_mechanics_unobserved"
        )
        if category == "mechanics_pass" and existing_wrapper_authorizes_development:
            action = (
                "Use only the already frozen, dated development preflight and its existing "
                "authorization chain. This disposition creates no authorization."
            )
        elif category == "mechanics_pass":
            action = (
                "Bank the standalone mechanics pass, then require the frozen wrapper's "
                "hash-bound verification before any dated development preflight."
            )
        else:
            action = "Continue only through the existing frozen wrapper to the mechanics gate."
        matched: list[str] = []
        category_failed: dict[str, list[str]] = {}
    else:
        category_failed = {
            name: sorted(set(failed).intersection(gate_names))
            for name, gate_names in categories.items()
            if set(failed).intersection(gate_names)
        }
        matched = [name for name in CATEGORY_PRIORITY if name in category_failed]
        if not matched:
            raise DispositionValidationError("a failed gate has no frozen disposition")
        category = matched[0]
        action = CATEGORY_ACTIONS[category]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "valid_disposition",
        "input": {
            "artifact_type": artifact_type,
            "path": str(source_path),
            "sha256": source_sha256,
            "hash_bound_by_wrapper": hash_bound_by_wrapper,
        },
        "primary_category": category,
        "matched_failure_categories": matched,
        "failed_gates": failed,
        "failed_gates_by_category": category_failed,
        "next_action": action,
        "existing_wrapper_authorizes_development": (
            existing_wrapper_authorizes_development
        ),
        "this_record_authorizes_development": False,
        "this_record_authorizes_paid_calls": False,
        "this_record_authorizes_rerun": False,
    }


def _classify_serving_result(
    result: Mapping[str, Any],
    *,
    path: Path,
    source_sha256: str,
    hash_bound_by_wrapper: bool = False,
) -> dict[str, Any]:
    _validate_raw_binding(path, result)
    gates = _validate_gate_record(
        result,
        artifact_type="serving",
        interface_version=serving.INTERFACE_VERSION,
        pass_status="passed",
        categories=SERVING_GATE_CATEGORIES,
    )
    return _classify_gates(
        artifact_type="serving_result",
        source_path=path,
        source_sha256=source_sha256,
        gates=gates,
        categories=SERVING_GATE_CATEGORIES,
        hash_bound_by_wrapper=hash_bound_by_wrapper,
    )


def _classify_mechanics_result(
    result: Mapping[str, Any],
    *,
    path: Path,
    source_sha256: str,
    existing_wrapper_authorizes_development: bool = False,
    hash_bound_by_wrapper: bool = False,
) -> dict[str, Any]:
    _validate_raw_binding(path, result)
    gates = _validate_gate_record(
        result,
        artifact_type="mechanics",
        interface_version=mechanics.INTERFACE_VERSION,
        pass_status="mechanics_pass",
        categories=MECHANICS_GATE_CATEGORIES,
    )
    return _classify_gates(
        artifact_type="mechanics_result",
        source_path=path,
        source_sha256=source_sha256,
        gates=gates,
        categories=MECHANICS_GATE_CATEGORIES,
        existing_wrapper_authorizes_development=existing_wrapper_authorizes_development,
        hash_bound_by_wrapper=hash_bound_by_wrapper,
    )


def _classify_wrapper(result: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise DispositionValidationError("wrapper schema version changed")
    if result.get("interface_version") != aug10.INTERFACE_VERSION:
        raise DispositionValidationError("wrapper interface version changed")
    if result.get("date") != aug10.EXPECTED_DATE:
        raise DispositionValidationError("wrapper execution date changed")
    status = result.get("status")
    if status not in {"complete", "stopped_after_serving_gated_null"}:
        raise DispositionValidationError("wrapper is not a terminal result")
    components = result.get("components")
    if not isinstance(components, dict):
        raise DispositionValidationError("wrapper components are missing")
    expected_components = (
        {"serving", "mechanics"} if status == "complete" else {"serving"}
    )
    if set(components) != expected_components:
        raise DispositionValidationError("wrapper component set is inconsistent")
    serving_path, serving_result = _validate_component(
        components["serving"], expected_statuses={"passed", "gated_null"}
    )
    serving_gates = _validate_gate_record(
        serving_result,
        artifact_type="serving",
        interface_version=serving.INTERFACE_VERSION,
        pass_status="passed",
        categories=SERVING_GATE_CATEGORIES,
    )
    if components["serving"]["status"] != serving_result["status"]:
        raise DispositionValidationError("wrapper serving status binding changed")
    if status == "stopped_after_serving_gated_null":
        if serving_gates["all_pass"] or result.get("authorizes_development") is not False:
            raise DispositionValidationError("stopped wrapper contradicts serving outcome")
        return _classify_gates(
            artifact_type="serving_result",
            source_path=serving_path,
            source_sha256=components["serving"]["artifact_sha256"],
            gates=serving_gates,
            categories=SERVING_GATE_CATEGORIES,
            hash_bound_by_wrapper=True,
        )
    if not serving_gates["all_pass"]:
        raise DispositionValidationError("complete wrapper contains a failed serving gate")
    mechanics_path, mechanics_result = _validate_component(
        components["mechanics"], expected_statuses={"mechanics_pass", "gated_null"}
    )
    if components["mechanics"]["status"] != mechanics_result["status"]:
        raise DispositionValidationError("wrapper mechanics status binding changed")
    authorizes = result.get("authorizes_development")
    if type(authorizes) is not bool:
        raise DispositionValidationError("wrapper authorization is not boolean")
    expected_authorization = mechanics_result.get("status") == "mechanics_pass"
    if authorizes is not expected_authorization:
        raise DispositionValidationError("wrapper authorization contradicts mechanics status")
    return _classify_mechanics_result(
        mechanics_result,
        path=mechanics_path,
        source_sha256=components["mechanics"]["artifact_sha256"],
        existing_wrapper_authorizes_development=authorizes,
        hash_bound_by_wrapper=True,
    )


def _classify_failure(result: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    interface = result.get("interface_version")
    if interface not in {serving.INTERFACE_VERSION, mechanics.INTERFACE_VERSION}:
        raise DispositionValidationError("failed-closed component interface changed")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != "failed_closed":
        raise DispositionValidationError("failed-closed component header is invalid")
    if not isinstance(result.get("error_type"), str) or not isinstance(result.get("error"), str):
        raise DispositionValidationError("failed-closed component error is missing")
    artifact_type = (
        "serving_failure"
        if interface == serving.INTERFACE_VERSION
        else "mechanics_failure"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "valid_disposition",
        "input": {
            "artifact_type": artifact_type,
            "path": str(path),
            "sha256": _sha256(path),
            "hash_bound_by_wrapper": False,
        },
        "primary_category": "transport_or_schema_inconclusive",
        "matched_failure_categories": ["transport_or_schema_inconclusive"],
        "failed_gates": [],
        "failed_gates_by_category": {},
        "next_action": CATEGORY_ACTIONS["transport_or_schema_inconclusive"],
        "existing_wrapper_authorizes_development": False,
        "this_record_authorizes_development": False,
        "this_record_authorizes_paid_calls": False,
        "this_record_authorizes_rerun": False,
    }


def classify_artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        result = _load(path)
        if result.get("status") == "failed_closed":
            return _classify_failure(result, path=path)
        if result.get("interface_version") == aug10.INTERFACE_VERSION:
            return _classify_wrapper(result, path=path)
        protocol = result.get("protocol")
        interface = protocol.get("interface_version") if isinstance(protocol, dict) else None
        if interface == serving.INTERFACE_VERSION:
            return _classify_serving_result(
                result, path=path, source_sha256=_sha256(path)
            )
        if interface == mechanics.INTERFACE_VERSION:
            return _classify_mechanics_result(
                result, path=path, source_sha256=_sha256(path)
            )
        raise DispositionValidationError("artifact interface is unknown")
    except DispositionValidationError as exc:
        source_hash = _sha256(path) if path.is_file() else None
        return {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "invalid_record",
            "input": {"path": str(path), "sha256": source_hash},
            "primary_category": "invalid_record",
            "validation_error": str(exc),
            "existing_wrapper_authorizes_development": False,
            "this_record_authorizes_development": False,
            "this_record_authorizes_paid_calls": False,
            "this_record_authorizes_rerun": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = classify_artifact(args.artifact)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "valid_disposition" else 2


if __name__ == "__main__":
    raise SystemExit(main())
