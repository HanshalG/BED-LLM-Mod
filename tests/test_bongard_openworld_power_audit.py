from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from scripts import bongard_openworld_power_audit as audit


def test_joint_effect_gate_uses_stricter_observed_threshold() -> None:
    design = audit.DESIGNS[0]
    sd_ratio = 0.20
    evidence = design.evidence_z * sd_ratio / math.sqrt(design.tasks)
    assert evidence < audit.MINIMUM_RELATIVE_GAIN
    assert audit.observed_gain_threshold(
        design, paired_difference_sd_ratio=sd_ratio
    ) == pytest.approx(audit.MINIMUM_RELATIVE_GAIN)
    assert audit.paired_effect_gate_power(
        design,
        true_relative_gain=audit.MINIMUM_RELATIVE_GAIN,
        paired_difference_sd_ratio=sd_ratio,
    ) == pytest.approx(0.5)


def test_confirmation_evidence_gate_can_dominate_three_percent_floor() -> None:
    design = audit.DESIGNS[2]
    threshold = audit.observed_gain_threshold(
        design, paired_difference_sd_ratio=0.20
    )
    assert threshold == pytest.approx(
        NormalDist().inv_cdf(0.975) * 0.20 / math.sqrt(64)
    )
    assert threshold > 0.03


def test_target_gain_inverts_power() -> None:
    for design in audit.DESIGNS:
        for sd_ratio in (0.10, 0.20, 0.30, 0.50):
            gain = audit.true_gain_for_target_power(
                design, paired_difference_sd_ratio=sd_ratio
            )
            assert audit.paired_effect_gate_power(
                design,
                true_relative_gain=gain,
                paired_difference_sd_ratio=sd_ratio,
            ) == pytest.approx(audit.TARGET_POWER)


def test_changed_path_gate_is_exact_on_boundaries() -> None:
    design = audit.DESIGNS[0]
    assert audit.changed_path_gate_probability(
        design, per_task_change_probability=0.0
    ) == 0.0
    assert audit.changed_path_gate_probability(
        design, per_task_change_probability=1.0
    ) == pytest.approx(1.0)


def test_changed_path_gate_matches_bruteforce_small_design() -> None:
    design = audit.Design(
        name="tiny", tasks=4, blocks=2, minimum_changed_paths=2, evidence_z=0.0
    )
    probability = 0.4
    # With two blocks of two, at least one per block already implies >=2 total.
    expected = (1.0 - (1.0 - probability) ** 2) ** 2
    assert audit.changed_path_gate_probability(
        design, per_task_change_probability=probability
    ) == pytest.approx(expected)


def test_candidate_transport_budgets_fit_daily_cap() -> None:
    development = audit.transport_budget(audit.DESIGNS[1])
    confirmation = audit.transport_budget(audit.DESIGNS[3])
    assert development == {
        "tasks_per_block": 16,
        "maximum_accepted_responses": 688,
        "maximum_transport_retries": 14,
        "maximum_http_attempts": 702,
        "maximum_precharged_exposure_usd": pytest.approx(2.808),
    }
    assert confirmation == {
        "tasks_per_block": 24,
        "maximum_accepted_responses": 1032,
        "maximum_transport_retries": 21,
        "maximum_http_attempts": 1053,
        "maximum_precharged_exposure_usd": pytest.approx(4.212),
    }


def test_rendered_audit_is_explicitly_non_forecasting() -> None:
    result = audit.build_audit()
    rendered = audit.render_markdown(result)
    assert result["model_calls_made"] == 0
    assert result["endpoint_data_accessed"] is False
    assert "design-sensitivity calculation, not an outcome forecast" in rendered
    assert "ranking, log-loss, and multi-control conjunctions unmodelled" in rendered
