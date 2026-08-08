from __future__ import annotations

from pathlib import Path

from scripts import bongard_openworld_prompt_role_blindness_audit as audit
from scripts import bongard_openworld_vlm_bed as bed


def test_position_predictor_is_fit_only_on_unobserved_roles() -> None:
    tasks = bed.load_validation_partition_tasks(
        "mechanics", include_endpoint_labels=False
    )
    probabilities = audit.fit_position_predictor(tasks)
    assert len(probabilities) == bed.NUM_IMAGES
    assert all(0.0 < probability < 1.0 for probability in probabilities)


def test_full_frozen_role_blindness_audit_passes(tmp_path: Path) -> None:
    result = audit.run_audit(output_path=tmp_path / "MANIFEST.json")
    assert result["status"] == "prompt_role_blindness_pass"
    assert result["all_gates_pass"] is True
    assert result["authorizes_paid_calls"] is False
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    confirmation = result["position_predictor"]["confirmation"]
    assert confirmation["rows"] == 960
    assert confirmation["endpoint_rows"] == 192
    assert confirmation["auc"] < audit.MAX_CONFIRMATION_AUC
    assert all(
        roles == ["candidate", "endpoint", "initial"]
        for roles in result["role_sets_by_public_position"]
    )
