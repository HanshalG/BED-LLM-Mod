from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_mechanics_disposition as disposition


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(*, kind: str, failed: tuple[str, ...] = ()) -> dict[str, object]:
    if kind == "mechanics":
        names = disposition.MECHANICS_GATES
        interface = mechanics.INTERFACE_VERSION
        pass_status = "mechanics_pass"
    else:
        names = disposition.SERVING_GATES
        interface = serving.INTERFACE_VERSION
        pass_status = "passed"
    gates = {name: name not in failed for name in names if name != "all_pass"}
    gates["all_pass"] = not failed
    return {
        "schema_version": 1,
        "status": pass_status if not failed else "gated_null",
        "protocol": {"interface_version": interface},
        "gates": gates,
    }


def _artifact(tmp_path: Path, *, kind: str, failed: tuple[str, ...] = ()) -> Path:
    path = tmp_path / kind / "RESULT.json"
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    _write(raw_path, {"responses": []})
    result = _result(kind=kind, failed=failed)
    result["raw_responses_sha256"] = _sha256(raw_path)
    _write(path, result)
    return path


def _component(path: Path, *, status: str) -> dict[str, object]:
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
        "raw_responses_sha256": _sha256(raw_path),
        "status": status,
        "cost_usd": 0.5,
        "verified": True,
    }


def _literal_gate_names(function: object) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(
            isinstance(parent, ast.Dict) and node in parent.keys
            for parent in ast.walk(tree)
            if isinstance(parent, ast.Dict)
        )
    }
    return names | set(disposition.TRANSPORT_GATES) | {"all_pass"}


def test_frozen_gate_sets_match_authoritative_gate_constructors() -> None:
    assert disposition.SERVING_GATES == _literal_gate_names(serving.serving_gates)
    assert disposition.MECHANICS_GATES == _literal_gate_names(mechanics.mechanics_gates)


def test_mechanics_pass_is_advisory_only(tmp_path: Path) -> None:
    result = disposition.classify_artifact(_artifact(tmp_path, kind="mechanics"))

    assert result["status"] == "valid_disposition"
    assert result["primary_category"] == "mechanics_pass"
    assert result["failed_gates"] == []
    assert result["existing_wrapper_authorizes_development"] is False
    assert result["this_record_authorizes_development"] is False
    assert result["this_record_authorizes_paid_calls"] is False
    assert result["this_record_authorizes_rerun"] is False


def test_multi_failure_uses_frozen_priority_and_reports_every_category(
    tmp_path: Path,
) -> None:
    path = _artifact(
        tmp_path,
        kind="mechanics",
        failed=(
            "zero_forced_exits",
            "all_scores_are_finite_and_executable",
            "root_candidate_brier_beats_constant_half",
            "at_least_24_of_32_branch_pairs_change_unobserved_beliefs",
            "dynamic_depth2_changes_at_least_one_myopic_first_action",
            "myopic_endpoint_is_not_saturated",
        ),
    )
    result = disposition.classify_artifact(path)

    assert result["primary_category"] == "transport_or_schema_inconclusive"
    assert result["matched_failure_categories"] == list(disposition.CATEGORY_PRIORITY)
    assert set(result["failed_gates_by_category"]) == set(disposition.CATEGORY_PRIORITY)
    assert len(result["failed_gates"]) == 6


def test_each_scientific_failure_category_is_reachable(tmp_path: Path) -> None:
    cases = {
        "integrity_or_leakage_failure": "all_scores_are_finite_and_executable",
        "predictive_belief_invalid": "root_candidate_brier_beats_constant_half",
        "path_dependence_absent": (
            "at_least_24_of_32_branch_pairs_change_unobserved_beliefs"
        ),
        "nonmyopic_opportunity_absent": (
            "dynamic_depth2_changes_at_least_one_myopic_first_action"
        ),
        "endpoint_saturated": "myopic_endpoint_is_not_saturated",
    }
    for category, gate in cases.items():
        path = _artifact(tmp_path / category, kind="mechanics", failed=(gate,))
        result = disposition.classify_artifact(path)
        assert result["primary_category"] == category


def test_gate_schema_and_consistency_mutations_fail_closed(tmp_path: Path) -> None:
    clean = _result(kind="mechanics")
    raw_path = tmp_path / "private/RAW_RESPONSES.json"
    _write(raw_path, {"responses": []})
    clean["raw_responses_sha256"] = _sha256(raw_path)
    mutations = []
    missing = copy.deepcopy(clean)
    del missing["gates"]["root_candidate_brier_beats_constant_half"]
    mutations.append(missing)
    extra = copy.deepcopy(clean)
    extra["gates"]["new_unregistered_gate"] = True
    mutations.append(extra)
    non_boolean = copy.deepcopy(clean)
    non_boolean["gates"]["zero_forced_exits"] = 1
    mutations.append(non_boolean)
    inconsistent = copy.deepcopy(clean)
    inconsistent["gates"]["all_pass"] = False
    mutations.append(inconsistent)
    stale = copy.deepcopy(clean)
    stale["protocol"]["interface_version"] = "stale-interface"
    mutations.append(stale)
    missing_raw = copy.deepcopy(clean)
    del missing_raw["raw_responses_sha256"]
    mutations.append(missing_raw)

    for index, value in enumerate(mutations):
        path = tmp_path / f"mutation-{index}.json"
        _write(path, value)
        result = disposition.classify_artifact(path)
        assert result["status"] == "invalid_record"
        assert result["primary_category"] == "invalid_record"
        assert result["this_record_authorizes_paid_calls"] is False
        assert result["this_record_authorizes_rerun"] is False


def test_wrapper_hash_binding_carries_existing_authorization_only(
    tmp_path: Path,
) -> None:
    serving_path = _artifact(tmp_path, kind="serving")
    mechanics_path = _artifact(tmp_path, kind="mechanics")
    wrapper = {
        "schema_version": 1,
        "interface_version": aug10.INTERFACE_VERSION,
        "status": "complete",
        "date": aug10.EXPECTED_DATE,
        "authorizes_development": True,
        "components": {
            "serving": _component(serving_path, status="passed"),
            "mechanics": _component(mechanics_path, status="mechanics_pass"),
        },
    }
    wrapper_path = tmp_path / "wrapper" / "RESULT.json"
    _write(wrapper_path, wrapper)

    result = disposition.classify_artifact(wrapper_path)

    assert result["primary_category"] == "mechanics_pass"
    assert result["input"]["hash_bound_by_wrapper"] is True
    assert result["existing_wrapper_authorizes_development"] is True
    assert result["this_record_authorizes_development"] is False
    assert result["this_record_authorizes_paid_calls"] is False

    mechanics_path.write_text("{}", encoding="utf-8")
    tampered = disposition.classify_artifact(wrapper_path)
    assert tampered["status"] == "invalid_record"
    assert tampered["existing_wrapper_authorizes_development"] is False


def test_failed_closed_component_is_inconclusive_and_never_a_rerun_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "FAILURE.json"
    _write(
        path,
        {
            "schema_version": 1,
            "interface_version": mechanics.INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": "TimeoutError",
            "error": "provider response was incomplete",
            "ledger_reconciliation_error": None,
        },
    )

    result = disposition.classify_artifact(path)

    assert result["primary_category"] == "transport_or_schema_inconclusive"
    assert result["this_record_authorizes_paid_calls"] is False
    assert result["this_record_authorizes_rerun"] is False
