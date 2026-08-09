from copy import deepcopy
import json

import pytest

from scripts import bongard_openworld_classical_horizon_opportunity as opportunity


def _tree(task_id: str, index: int) -> dict:
    root = {"a": 0.10, "b": 0.20, "c": 0.15}
    dynamic_future = {"a": 0.30, "b": 0.10, "c": 0.20}
    mapping = {"a": "b", "b": "c", "c": "a"}
    shuffled_future = {
        candidate: dynamic_future[source] for candidate, source in mapping.items()
    }
    dynamic = {
        candidate: root[candidate] + dynamic_future[candidate]
        for candidate in root
    }
    shuffled = {
        candidate: root[candidate] + shuffled_future[candidate]
        for candidate in root
    }
    scores = {
        "dynamic_depth2": dynamic,
        "compute_matched_myopic_ensemble": {
            "a": 0.24,
            "b": 0.31,
            "c": 0.22,
        },
        "shuffled_dynamic_depth2": shuffled,
        "history_blind_depth2": {"a": 0.25, "b": 0.30, "c": 0.20},
        "myopic_width": root,
    }
    endpoints = {
        "dynamic_depth2": (0.20 + index * 0.0001, 0.50 + index * 0.0002),
        "compute_matched_myopic_ensemble": (
            0.245 + index * 0.0001,
            0.61 + index * 0.0002,
        ),
        "shuffled_dynamic_depth2": (0.24, 0.60),
        "history_blind_depth2": (0.23, 0.57),
        "myopic_width": (0.25, 0.63),
    }
    policies = {
        policy: {
            "first_image_id": max(scores[policy], key=scores[policy].get),
            "final_history_key": f"{policy}-{task_id}",
            "endpoint": {
                "mean_brier": values[0],
                "mean_log_loss": values[1],
            },
        }
        for policy, values in endpoints.items()
    }
    return {
        "task_id": task_id,
        "root_scores": scores,
        "continuation_values": {
            "dynamic_expected_continuation_utility": dynamic_future,
            "shuffled_expected_continuation_utility": shuffled_future,
        },
        "shuffled_branch_mapping": mapping,
        "policies": policies,
        "ranking_fidelity": {
            "dynamic_depth2": 0.5,
            "compute_matched_myopic_ensemble": 0.3,
        },
    }


def _manifest(task_ids: list[str]) -> dict:
    rows = []
    for index, task_id in enumerate(task_ids):
        changed = index < opportunity.DISAGREEMENT_COUNTS["development"]
        rows.append(
            {
                "task_id": task_id,
                "stratum": opportunity.STRATA[0] if changed else opportunity.STRATA[1],
                "encoders": {
                    "dinov2": {
                        "depth2_first_image_id": "a" if changed else "b",
                        "myopic_first_image_id": "b",
                        "changed_first_query": changed,
                    },
                    "siglip": {
                        "depth2_first_image_id": "b",
                        "myopic_first_image_id": "b",
                        "changed_first_query": False,
                    },
                },
            }
        )
    return {
        "definition": "dinov2_depth2_or_siglip_depth2_changes_own_myopic_first_query",
        "partitions": {"development": {"rows": rows}},
    }


def _stage_result(task_ids: list[str]) -> dict:
    return {
        "status": "development_signal",
        "claim_tier": "fixture",
        "trees": [_tree(task_id, index) for index, task_id in enumerate(task_ids)],
    }


def test_frozen_manifest_is_endpoint_blind_and_replays() -> None:
    manifest = opportunity.load_frozen_manifest()

    assert manifest == opportunity.build_manifest()
    assert manifest["candidate_labels_accessed"] is False
    assert manifest["endpoint_labels_accessed"] is False
    assert manifest["luna_responses_accessed"] is False
    assert manifest["model_calls"] == 0
    assert manifest["authorizes_paid_calls"] is False
    for partition, expected in opportunity.PARTITION_COUNTS.items():
        summary = manifest["partitions"][partition]
        assert summary["task_count"] == expected
        assert summary["classical_horizon_disagreement_count"] == (
            opportunity.DISAGREEMENT_COUNTS[partition]
        )


def test_build_report_stratifies_exact_compute_matched_effects() -> None:
    task_ids = [f"task-{index:02d}" for index in range(64)]
    report = opportunity.build_report(
        stage="development",
        stage_result=_stage_result(task_ids),
        authorization={"verified": True, "stage": "development"},
        manifest=_manifest(task_ids),
    )

    assert report["status"] == "classical_horizon_opportunity_stratum_complete"
    assert report["compute_contract_exact"] is True
    assert report["authorizes_paid_calls"] is False
    assert report["changes_primary_gates"] is False
    assert report["changes_claim_tier"] is False
    disagreement = report["strata"][opportunity.STRATA[0]]
    agreement = report["strata"][opportunity.STRATA[1]]
    assert disagreement["task_count"] == 27
    assert agreement["task_count"] == 37
    for summary in (disagreement, agreement):
        assert summary["dynamic_compute_matched_first_query_changes"] == summary[
            "task_count"
        ]
        assert summary["dynamic_relative_brier_gain"] > 0.03
        assert summary["paired"]["mean_brier"]["mean_difference"] < 0
        assert summary["paired"]["mean_log_loss"]["mean_difference"] < 0
        assert summary["dynamic_mean_spearman"] > summary[
            "compute_matched_myopic_mean_spearman"
        ]


def test_build_report_rejects_compute_contract_tamper() -> None:
    task_ids = [f"task-{index:02d}" for index in range(64)]
    result = _stage_result(task_ids)
    result["trees"][0]["root_scores"]["dynamic_depth2"]["a"] += 0.01

    with pytest.raises(ValueError, match="dynamic score"):
        opportunity.build_report(
            stage="development",
            stage_result=result,
            authorization={"verified": True},
            manifest=_manifest(task_ids),
        )


def test_build_report_rejects_manifest_task_drift() -> None:
    task_ids = [f"task-{index:02d}" for index in range(64)]
    manifest = _manifest(task_ids)
    manifest["partitions"]["development"]["rows"][0]["task_id"] = "replacement"

    with pytest.raises(ValueError, match="identities"):
        opportunity.build_report(
            stage="development",
            stage_result=_stage_result(task_ids),
            authorization={"verified": True},
            manifest=manifest,
        )


def test_run_report_authorizes_before_loading_and_writes_once(tmp_path) -> None:
    frozen = opportunity.load_frozen_manifest()
    task_ids = [
        row["task_id"]
        for row in frozen["partitions"]["development"]["rows"]
    ]
    result_path = tmp_path / "COMBINED_RESULT.json"
    result_path.write_text(json.dumps(_stage_result(task_ids)), encoding="utf-8")
    output_path = tmp_path / "OPPORTUNITY.json"
    events = []

    def authorizer(**kwargs):
        assert kwargs["stage"] == "development"
        assert not events
        events.append("authorized")
        return {"verified": True, "stage": "development"}

    def loader(path):
        assert events == ["authorized"]
        events.append("loaded")
        return json.loads(path.read_text(encoding="utf-8"))

    report = opportunity.run_report(
        stage="development",
        result_path=result_path,
        output_path=output_path,
        authorizer=authorizer,
        result_loader=loader,
    )

    assert events == ["authorized", "loaded"]
    assert json.loads(output_path.read_text()) == report
    with pytest.raises(FileExistsError):
        opportunity.run_report(
            stage="development",
            result_path=result_path,
            output_path=output_path,
            authorizer=authorizer,
            result_loader=loader,
        )


def test_run_report_never_loads_unauthorized_result(tmp_path) -> None:
    loaded = False

    def loader(path):
        nonlocal loaded
        loaded = True
        return {}

    with pytest.raises(ValueError, match="authorization failed"):
        opportunity.run_report(
            stage="development",
            result_path=tmp_path / "missing.json",
            output_path=tmp_path / "OPPORTUNITY.json",
            authorizer=lambda **kwargs: {"verified": False},
            result_loader=loader,
        )
    assert loaded is False


def test_build_report_rejects_unauthorized_stage() -> None:
    task_ids = [f"task-{index:02d}" for index in range(64)]
    with pytest.raises(ValueError, match="not authorized"):
        opportunity.build_report(
            stage="development",
            stage_result=deepcopy(_stage_result(task_ids)),
            authorization={"verified": False},
            manifest=_manifest(task_ids),
        )
