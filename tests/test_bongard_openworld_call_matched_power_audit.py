from __future__ import annotations

import pytest

from scripts import bongard_openworld_call_matched_power_audit as audit
from scripts import bongard_openworld_power_audit as base


def test_bindings_and_registered_gate_family_are_exact() -> None:
    bindings = audit.verify_bindings()
    assert set(bindings) == set(audit.BOUND_FILES)
    assert len(audit.CALL_MATCHED_CONFIRMATION_GATES) == 7


def test_change_rate_inverts_exact_blockwise_gate_probability() -> None:
    for design in (base.DESIGNS[1], base.DESIGNS[3]):
        rate = audit.change_probability_for_target_power(design)
        probability = base.changed_path_gate_probability(
            design, per_task_change_probability=rate
        )
        assert 0.40 < rate < 0.45
        assert probability == pytest.approx(audit.TARGET_POWER)


def test_standardized_mean_margin_inverts_one_sided_power() -> None:
    assert audit.standardized_mean_margin_for_target_power(64) == pytest.approx(
        0.105202, abs=1e-6
    )
    assert audit.standardized_mean_margin_for_target_power(96) == pytest.approx(
        0.085898, abs=1e-6
    )
    with pytest.raises(ValueError):
        audit.standardized_mean_margin_for_target_power(0)


def test_report_is_zero_call_nonforecasting_and_non_authorizing() -> None:
    result = audit.build_audit()
    rendered = audit.render_markdown(result)
    assert result["status"] == "call_matched_gate_sensitivity_complete"
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["endpoint_data_accessed"] is False
    assert result["changes_scientific_gates"] is False
    assert result["authorizes_paid_calls"] is False
    assert "not joint power and not an outcome forecast" in rendered
    assert "must not trigger threshold relaxation" in rendered
