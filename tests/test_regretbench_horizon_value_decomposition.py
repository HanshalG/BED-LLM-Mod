from __future__ import annotations

import pytest

from scripts import regretbench_horizon_value_decomposition as audit


def _result(
    *,
    mode: str = "supported",
    mechanics_pass: bool = True,
    smc: bool = False,
    confirmation: bool = False,
) -> dict:
    dynamic = "smc_dynamic_depth2" if smc else "dynamic_depth2"
    refresh = "smc_myopic_refresh_brier" if smc else "myopic_refresh_brier"
    tasks = []
    for index in range(64):
        immediate_refresh = 0.20
        if mode == "no_forecast":
            immediate_dynamic = 0.205
            terminal_dynamic = 0.13
            terminal_refresh = 0.132 + index / 100_000
        else:
            immediate_dynamic = 0.25
            terminal_dynamic = 0.10
            terminal_refresh = 0.13 + index / 1_000
        predicted = terminal_refresh - terminal_dynamic
        if mode == "miscalibrated":
            realized = -0.02 - index / 2_000
        elif mode == "partial":
            realized = 0.005 + index / 10_000
        else:
            realized = 0.025 + index / 2_000
        tasks.append(
            {
                "task_id": f"task-{index}",
                "selected_roots": {dynamic: 0, refresh: 1},
                "conditioned_root_risks": [
                    {"brier": terminal_dynamic},
                    {"brier": terminal_refresh},
                    {"brier": 0.8},
                    {"brier": 0.9},
                ],
                "myopic_refresh_brier_root_risks": [
                    {"brier": immediate_dynamic},
                    {"brier": immediate_refresh},
                    {"brier": 0.8},
                    {"brier": 0.9},
                ],
                "policies": {
                    dynamic: {"brier": 0.4 - realized / 2},
                    refresh: {"brier": 0.4 + realized / 2},
                },
            }
        )
    interface = (
        "regretbench-deepseek-smc-confirmation-1"
        if smc and confirmation
        else (
            "regretbench-deepseek-smc-dynamic-depth2-experiment-1"
            if smc
            else (
                "regretbench-deepseek-dynamic-depth2-confirmation-1"
                if confirmation
                else "regretbench-deepseek-dynamic-depth2-policy-1"
            )
        )
    )
    return {
        "interface_version": interface,
        "status": "gated_null" if mechanics_pass else "mechanics_failed",
        "mechanics_gates": {"all_pass": mechanics_pass},
        "tasks": tasks,
    }


def _analyze(result: dict, *, samples: int = 1_000) -> dict:
    return audit.analyze(
        result,
        result_sha256="result",
        verification_sha256="verification",
        binding_sha256="binding",
        samples=samples,
        seed=17,
    )


def test_exact_decomposition_supports_material_realized_horizon_value() -> None:
    result = _analyze(_result())
    metrics = result["metrics"]
    quantities = metrics["quantities"]
    assert result["region"] == "descriptive_horizon_value_supported"
    assert quantities["immediate_penalty"]["mean"] == pytest.approx(0.05)
    assert quantities["differential_horizon_value"]["mean"] == pytest.approx(
        quantities["predicted_terminal_advantage"]["mean"] + 0.05
    )
    assert quantities["realized_terminal_advantage"]["ci95"][0] > 0
    assert metrics["predicted_to_realized"]["spearman"] == pytest.approx(1.0)
    assert metrics["realized_dynamic_vs_refresh"] == {
        "wins": 64,
        "ties": 0,
        "losses": 0,
    }
    assert result["can_change_status_authorization_or_claim_tier"] is False
    assert result["model_calls"] == 0


def test_no_material_horizon_forecast_is_separate_from_bad_realization() -> None:
    result = _analyze(_result(mode="no_forecast"))
    assert result["region"] == "no_material_horizon_forecast"
    differential = result["metrics"]["quantities"][
        "differential_horizon_value"
    ]
    assert differential["mean"] < 0.01


def test_material_forecast_can_be_realized_in_wrong_direction() -> None:
    result = _analyze(_result(mode="miscalibrated"))
    assert result["region"] == "forecast_horizon_not_realized"
    assert result["metrics"]["quantities"]["differential_horizon_value"][
        "mean"
    ] > 0.01
    assert result["metrics"]["quantities"]["realized_terminal_advantage"][
        "mean"
    ] < 0


def test_partial_positive_evidence_does_not_cross_supported_threshold() -> None:
    result = _analyze(_result(mode="partial"))
    assert result["region"] == "partial_horizon_value_evidence"
    assert result["literal_interpretation_gates"][
        "realized_advantage_mean_at_least_002"
    ] is False


def test_mechanics_failure_has_no_efficacy_metrics() -> None:
    result = _analyze(_result(mechanics_pass=False), samples=100)
    assert result["status"] == "unavailable_mechanics_failed"
    assert result["region"] == "unavailable_mechanics_failed"
    assert result["metrics"] is None


@pytest.mark.parametrize(
    ("smc", "confirmation"),
    ((False, True), (True, False), (True, True)),
)
def test_all_registered_result_interfaces(smc: bool, confirmation: bool) -> None:
    result = _analyze(
        _result(smc=smc, confirmation=confirmation), samples=200
    )
    assert result["region"] == "descriptive_horizon_value_supported"


def test_selector_inconsistency_fails_closed() -> None:
    value = _result()
    value["tasks"][0]["myopic_refresh_brier_root_risks"][0]["brier"] = 0.1
    with pytest.raises(ValueError, match="immediate-risk minimizing"):
        _analyze(value, samples=100)


def test_markdown_preserves_non_gating_boundary() -> None:
    rendered = audit.render_markdown(_analyze(_result(), samples=100))
    assert "cannot alter status, authorization, confirmation, or claim tier" in rendered
    assert "differential_horizon_value" in rendered
