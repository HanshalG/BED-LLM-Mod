from copy import deepcopy
import json
import math

import pytest

from scripts import bongard_openworld_compute_matched_control as audit


def _tree(index: int) -> dict:
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
    history_blind = {"a": 0.25, "b": 0.30, "c": 0.20}
    endpoints = {
        "dynamic_depth2": (0.20 + index * 0.001, 0.50 + index * 0.002),
        "compute_matched_myopic_ensemble": (
            0.245 + index * 0.001,
            0.61 + index * 0.002,
        ),
        "shuffled_dynamic_depth2": (0.24 + index * 0.001, 0.60 + index * 0.002),
        "history_blind_depth2": (0.23 + index * 0.001, 0.57 + index * 0.002),
        "myopic_width": (0.25 + index * 0.001, 0.63 + index * 0.002),
    }
    scores = {
        "dynamic_depth2": dynamic,
        "compute_matched_myopic_ensemble": {
            "a": 0.24,
            "b": 0.31,
            "c": 0.22,
        },
        "shuffled_dynamic_depth2": shuffled,
        "history_blind_depth2": history_blind,
        "myopic_width": root,
    }
    policies = {}
    for policy, values in endpoints.items():
        policies[policy] = {
            "first_image_id": max(scores[policy], key=scores[policy].get),
            "final_history_key": f"{policy}-history-{index}",
            "endpoint": {
                "mean_brier": values[0],
                "mean_log_loss": values[1],
            },
        }
    return {
        "task_id": f"task-{index}",
        "root_scores": scores,
        "continuation_values": {
            "dynamic_expected_continuation_utility": dynamic_future,
            "shuffled_expected_continuation_utility": shuffled_future,
        },
        "shuffled_branch_mapping": mapping,
        "policies": policies,
    }


def _stage_result() -> dict:
    return {
        "status": "mechanics_pass",
        "claim_tier": None,
        "trees": [_tree(index) for index in range(4)],
    }


def test_build_report_replays_compute_contract_and_paired_effects() -> None:
    report = audit.build_report(
        stage="mechanics",
        stage_result=_stage_result(),
        authorization={"verified": True, "stage": "mechanics"},
    )

    assert report["status"] == "compute_matched_control_audit_complete"
    assert report["strict_compute_matched_myopic_control"] == (
        "compute_matched_myopic_ensemble"
    )
    assert report["strict_continuation_compute_matched_control"] == (
        "shuffled_dynamic_depth2"
    )
    assert report["matched_request_count_control"] == "history_blind_depth2"
    assert report["online_regeneration_greedy_control"] == "myopic_width"
    assert report["compute_contract_exact"] is True
    assert report["model_calls"] == 0
    assert report["cost_usd"] == 0.0
    assert report["authorizes_paid_calls"] is False
    assert report["changes_claim_tier"] is False
    for control in audit.CONTROLS:
        for metric in audit.METRICS:
            summary = report["comparisons"][control][metric]
            assert summary["n"] == 4
            assert summary["mean_difference"] < 0
            assert summary["bootstrap_draws"] == 20_000
            assert math.isclose(
                summary["standard_error"],
                summary["sample_sd"] / math.sqrt(summary["n"]),
            )
            assert summary["negative_favors"] == "dynamic_depth2"


@pytest.mark.parametrize(
    "mutation",
    ("dynamic_score", "shuffled_value", "mapping", "selected_action"),
)
def test_build_report_fails_closed_on_compute_contract_tamper(mutation: str) -> None:
    result = _stage_result()
    tree = result["trees"][0]
    if mutation == "dynamic_score":
        tree["root_scores"]["dynamic_depth2"]["a"] += 0.01
    elif mutation == "shuffled_value":
        tree["continuation_values"][
            "shuffled_expected_continuation_utility"
        ]["a"] += 0.01
    elif mutation == "mapping":
        tree["shuffled_branch_mapping"]["a"] = "a"
    else:
        tree["policies"]["dynamic_depth2"]["first_image_id"] = "b"

    with pytest.raises(ValueError):
        audit.build_report(
            stage="mechanics",
            stage_result=result,
            authorization={"verified": True},
        )


def test_run_report_authorizes_before_loading_and_writes_once(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(_stage_result()), encoding="utf-8")
    output_path = tmp_path / "AUDIT.json"
    events = []

    def authorizer(**kwargs):
        assert kwargs["stage"] == "mechanics"
        assert not events
        events.append("authorized")
        return {"verified": True, "stage": "mechanics"}

    def loader(path):
        assert events == ["authorized"]
        events.append("loaded")
        return json.loads(path.read_text(encoding="utf-8"))

    report = audit.run_report(
        stage="mechanics",
        result_path=result_path,
        output_path=output_path,
        authorizer=authorizer,
        result_loader=loader,
    )

    assert events == ["authorized", "loaded"]
    assert json.loads(output_path.read_text()) == report
    assert report["stage_result_sha256"] == audit._sha256(result_path)
    with pytest.raises(FileExistsError):
        audit.run_report(
            stage="mechanics",
            result_path=result_path,
            output_path=output_path,
            authorizer=authorizer,
            result_loader=loader,
        )


def test_run_report_never_loads_an_unauthorized_result(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(_stage_result()), encoding="utf-8")
    loaded = False

    def loader(path):
        nonlocal loaded
        loaded = True
        return json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="authorization failed"):
        audit.run_report(
            stage="mechanics",
            result_path=result_path,
            output_path=tmp_path / "AUDIT.json",
            authorizer=lambda **kwargs: {"verified": False},
            result_loader=loader,
        )
    assert loaded is False
