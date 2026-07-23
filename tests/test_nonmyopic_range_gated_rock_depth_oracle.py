import pytest

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_range_gated_rock_depth_oracle import (
    RangeGatedDepthConfig,
    run_qualification,
)
from scripts.audit_nonmyopic_range_gated_rock_depth_oracle import audit


def test_range_gated_sensor_is_weak_remotely_and_strong_onsite() -> None:
    model = RangeGatedRockDiagnosisModel(get_paper_map("7-8"))
    rock_position = model.map_spec.rock_positions[5]

    assert model.sensor_accuracy(model.map_spec.start_position, 5) == 0.55
    assert model.sensor_accuracy(rock_position, 5) == 0.95
    assert model.expected_information_gain(rock_position, model.initial_belief, "check-5") > (
        model.expected_information_gain(model.map_spec.start_position, model.initial_belief, "check-5")
    )


def test_range_gated_accuracy_contract_is_validated() -> None:
    with pytest.raises(ValueError, match="remote < onsite"):
        RangeGatedRockDiagnosisModel(
            get_paper_map("7-8"), remote_accuracy=0.95, onsite_accuracy=0.55
        )


def test_tiny_exact_qualification_exposes_depth_three_mechanism() -> None:
    summary = run_qualification(
        RangeGatedDepthConfig(
            num_trials=4,
            num_rounds=4,
            bootstrap_replicates=50,
            trial_concurrency=1,
        )
    )

    assert summary["primary_gate_passed"]
    assert summary["truth_log_corroboration_passed"]
    assert all(summary["mechanics"].values())
    assert summary["comparisons"]["d3_minus_d2"]["entropy_auc_gain_mean"] > 0.0
    audited = audit(summary)
    assert audited["passed"]
    assert all(audited["mechanics"].values())
