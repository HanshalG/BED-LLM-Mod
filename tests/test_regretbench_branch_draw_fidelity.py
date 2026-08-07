from __future__ import annotations

import hashlib
import json

import pytest

from scripts import regretbench_branch_draw_fidelity as audit


def _result(
    *,
    shared_reverse: bool = False,
    mechanics_pass: bool = True,
    smc: bool = False,
    confirmation: bool = False,
) -> dict:
    dynamic = "smc_dynamic_depth2" if smc else "dynamic_depth2"
    refresh = "smc_myopic_refresh_brier" if smc else "myopic_refresh_brier"
    tasks = []
    for index in range(64):
        realized = 0.02 + 0.002 * index
        noise = 0.12 if index % 2 else -0.12
        if shared_reverse:
            draw0 = draw1 = -realized
        else:
            draw0 = realized + noise
            draw1 = realized - noise

        def risks(predicted: float) -> list[dict[str, float]]:
            return [
                {"brier": 0.5},
                {"brier": 0.5 + predicted},
                {"brier": 0.8},
                {"brier": 0.9},
            ]

        ensemble = (draw0 + draw1) / 2.0
        tasks.append(
            {
                "task_id": f"task-{index}",
                "selected_roots": {
                    dynamic: 0,
                    refresh: 1,
                },
                "conditioned_draw_root_risks": [risks(draw0), risks(draw1)],
                "conditioned_root_risks": risks(ensemble),
                "policies": {
                    dynamic: {"brier": 0.5 - realized / 2.0},
                    refresh: {"brier": 0.5 + realized / 2.0},
                },
            }
        )
    return {
        "interface_version": (
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
        ),
        "status": "gated_null" if mechanics_pass else "mechanics_failed",
        "mechanics_gates": {"all_pass": mechanics_pass},
        "tasks": tasks,
    }


def test_ensemble_recovers_ranking_from_opposed_draw_noise() -> None:
    result = audit.analyze(
        _result(),
        result_sha256="result",
        verification_sha256="verification",
        binding_sha256="binding",
        samples=1_000,
        seed=17,
    )
    metrics = result["metrics"]
    assert result["status"] == "complete_descriptive_non_gating"
    assert metrics["changed_root_task_count"] == 64
    assert metrics["predicted_to_realized"]["ensemble"]["spearman"] == pytest.approx(1.0)
    assert metrics["predicted_to_realized"]["ensemble"]["rmse"] < 1e-12
    assert metrics["bootstrap"]["ensemble_exceeds_both_probability"] > 0.99
    assert result["can_change_status_authorization_or_claim_tier"] is False
    assert result["model_calls"] == 0


def test_shared_reversed_draws_expose_common_ranking_failure() -> None:
    result = audit.analyze(
        _result(shared_reverse=True),
        result_sha256="result",
        verification_sha256="verification",
        binding_sha256="binding",
        samples=500,
        seed=19,
    )
    metrics = result["metrics"]
    assert metrics["predicted_to_realized"]["draw0"]["spearman"] == pytest.approx(-1.0)
    assert metrics["predicted_to_realized"]["draw1"]["spearman"] == pytest.approx(-1.0)
    assert metrics["predicted_to_realized"]["ensemble"]["spearman"] == pytest.approx(-1.0)
    assert metrics["draw_prediction_agreement"]["pearson"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("smc", "confirmation"),
    ((False, True), (True, False), (True, True)),
)
def test_all_development_and_confirmation_interfaces(smc: bool, confirmation: bool) -> None:
    result = audit.analyze(
        _result(smc=smc, confirmation=confirmation),
        result_sha256="result",
        verification_sha256="verification",
        binding_sha256="binding",
        samples=100,
        seed=23,
    )
    assert result["status"] == "complete_descriptive_non_gating"
    assert result["metrics"]["changed_root_task_count"] == 64


def test_mechanics_failure_has_no_fidelity_metrics() -> None:
    result = audit.analyze(
        _result(mechanics_pass=False),
        result_sha256="result",
        verification_sha256="verification",
        binding_sha256="binding",
        samples=10,
    )
    assert result["status"] == "unavailable_mechanics_failed"
    assert result["metrics"] is None


def test_verified_input_rejects_result_hash_mismatch(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    verification_path = tmp_path / "VERIFICATION.json"
    result_path.write_text(json.dumps(_result()))
    verification_path.write_text(
        json.dumps(
            {
                "status": "verified",
                "result_status": "gated_null",
                "mismatches": [],
                "checks": {"reported_result_matches_replay": True},
                "artifact_sha256": {"RESULT.json": "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="clean independent verification"):
        audit._validated_inputs(tmp_path)


def test_execution_binding_matches_script_and_protocol() -> None:
    binding = audit.validate_binding()
    for name, path in {"protocol": audit.PROTOCOL, "script": audit.Path(audit.__file__)}.items():
        assert binding[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
