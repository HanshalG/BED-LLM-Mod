from __future__ import annotations

import copy
import hashlib

import pytest

from scripts import regretbench_branch_draw_decision as decision


def _audit() -> dict:
    return {
        "interface_version": "regretbench-branch-draw-fidelity-1",
        "status": "complete_descriptive_non_gating",
        "result_sha256": "result",
        "verification_sha256": "verification",
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
        "metrics": {
            "changed_root_task_count": 24,
            "predicted_to_realized": {
                "draw0": {
                    "spearman": 0.10,
                    "probability_positive": 0.60,
                    "ci95": [-0.2, 0.4],
                    "rmse": 0.10,
                },
                "draw1": {
                    "spearman": 0.05,
                    "probability_positive": 0.55,
                    "ci95": [-0.25, 0.35],
                    "rmse": 0.11,
                },
                "ensemble": {
                    "spearman": 0.12,
                    "probability_positive": 0.70,
                    "ci95": [-0.1, 0.35],
                    "rmse": 0.09,
                },
            },
            "draw_prediction_agreement": {
                "pearson": 0.40,
                "root_mean_square_gap": 0.04,
            },
            "bootstrap": {
                "requested_samples": 20_000,
                "retained_samples": 19_900,
                "ensemble_exceeds_both_probability": 0.60,
            },
        },
    }


def _classify(audit: dict) -> dict:
    return decision.classify(audit, binding_sha256="binding")


def test_draw_variance_region() -> None:
    value = _audit()
    rows = value["metrics"]["predicted_to_realized"]
    rows["draw0"].update(spearman=0.0, probability_positive=0.5, rmse=0.20)
    rows["draw1"].update(spearman=0.02, probability_positive=0.52, rmse=0.18)
    rows["ensemble"].update(spearman=0.35, probability_positive=0.94, rmse=0.15)
    value["metrics"]["bootstrap"]["ensemble_exceeds_both_probability"] = 0.91
    value["metrics"]["draw_prediction_agreement"].update(
        pearson=0.3, root_mean_square_gap=0.08
    )
    assert _classify(value)["region"] == "averaging_reduces_draw_noise"


def test_shared_error_region() -> None:
    value = _audit()
    rows = value["metrics"]["predicted_to_realized"]
    for row in rows.values():
        row.update(spearman=0.05, probability_positive=0.55)
    value["metrics"]["draw_prediction_agreement"].update(
        pearson=0.95, root_mean_square_gap=0.01
    )
    assert _classify(value)["region"] == "shared_ranking_error"


def test_draw_sensitive_region() -> None:
    value = _audit()
    rows = value["metrics"]["predicted_to_realized"]
    rows["draw0"].update(spearman=0.45, probability_positive=0.92)
    rows["draw1"].update(spearman=-0.10, probability_positive=0.30)
    rows["ensemble"].update(spearman=0.12, probability_positive=0.70)
    assert _classify(value)["region"] == "draw_sensitive"


def test_adequate_and_inconclusive_regions() -> None:
    adequate = _audit()
    adequate["metrics"]["predicted_to_realized"]["ensemble"].update(
        spearman=0.25, probability_positive=0.88
    )
    assert _classify(adequate)["region"] == "fidelity_adequate_no_draw_escalation"
    assert _classify(_audit())["region"] == "inconclusive_no_adaptive_repair"


def test_insufficient_information_precedes_other_regions() -> None:
    value = _audit()
    value["metrics"]["changed_root_task_count"] = 15
    rows = value["metrics"]["predicted_to_realized"]
    rows["ensemble"].update(spearman=0.8, probability_positive=0.99, rmse=0.01)
    value["metrics"]["bootstrap"]["ensemble_exceeds_both_probability"] = 0.99
    assert _classify(value)["region"] == "insufficient_changed_roots"


@pytest.mark.parametrize(
    "mutation",
    (
        "pearson",
        "interval",
        "bootstrap_contrast",
    ),
)
def test_unavailable_correlation_evidence_is_insufficient(mutation: str) -> None:
    value = _audit()
    if mutation == "pearson":
        value["metrics"]["draw_prediction_agreement"]["pearson"] = None
    elif mutation == "interval":
        value["metrics"]["predicted_to_realized"]["draw0"]["ci95"] = [None, None]
    else:
        value["metrics"]["bootstrap"]["ensemble_exceeds_both_probability"] = None
    assert _classify(value)["region"] == "insufficient_changed_roots"


def test_mechanics_failure_is_unavailable() -> None:
    value = _audit()
    value["status"] = "unavailable_mechanics_failed"
    value["metrics"] = None
    assert _classify(value)["region"] == "unavailable_mechanics_failed"


def test_non_gating_boundary_is_enforced() -> None:
    value = copy.deepcopy(_audit())
    value["model_calls"] = 1
    with pytest.raises(ValueError, match="non-gating boundary"):
        _classify(value)


def test_markdown_and_execution_binding() -> None:
    result = _classify(_audit())
    text = decision.render_markdown(result)
    assert result["region"] in text
    assert "non-gating" in text
    binding = decision.validate_binding()
    expected = {
        "protocol": decision.PROTOCOL,
        "script": decision.Path(decision.__file__),
        "fidelity_binding": decision.FIDELITY_BINDING,
    }
    for name, path in expected.items():
        assert binding[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
