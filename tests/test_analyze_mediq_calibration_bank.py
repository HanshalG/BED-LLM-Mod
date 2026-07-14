from __future__ import annotations

from pathlib import Path

from core.experiment import run_from_config
from helpers import Config
from scripts.analyze_mediq_calibration_bank import analyze_bank
from tests.test_mediq import FIXTURE, RoutingMediQModel


def test_mediq_calibration_bank_analyzer_accepts_valid_naive_bank(
    tmp_path: Path,
) -> None:
    config = Config(
        task="mediq",
        method_names=["naive"],
        mediq_data_path=str(FIXTURE),
        mediq_verify_official_hash=False,
        mediq_num_trials=2,
        mediq_num_rounds=1,
        mediq_trial_batch_size=2,
        mediq_num_candidates=1,
        mediq_likelihood_mode="data_estimation",
        generation_temperature_simple=0.0,
        answer_temperature=0.0,
    )
    run_from_config(
        config,
        RoutingMediQModel(),
        RoutingMediQModel(),
        output_dir=tmp_path,
    )

    report = analyze_bank(
        tmp_path,
        data_path=FIXTURE,
        task_offset=0,
        expected_tasks=2,
        expected_rounds=1,
        minimum_available_turns=2,
        verify_official_hash=False,
    )

    assert report["status"] == "automated_pass_manual_review_pending"
    assert report["automated_pass"] is True
    assert all(report["checks"].values())
    assert report["available_turns"] == 2
    assert report["endpoint_accuracy_is_ignored"] is True
