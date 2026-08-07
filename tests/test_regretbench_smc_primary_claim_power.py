from __future__ import annotations

import json

import pytest

from scripts import regretbench_smc_primary_claim_power as power


def test_synthetic_tasks_match_frozen_disagreement_and_gain_targets() -> None:
    scenario = power.SCENARIOS[1]
    rng = power.np.random.default_rng(123)

    tasks = power.synthetic_tasks(rng, scenario)
    summary = power.smc._smc_scientific_summary(tasks, samples=50)

    assert len(tasks) == 64
    assert summary["root_disagreements"]["myopic_refresh_brier"] == (
        scenario.refresh_changed_roots
    )
    assert summary["root_disagreements"]["history_blind_depth2"] == (
        scenario.blind_changed_roots
    )
    assert summary[
        "mean_conditioned_predicted_gain_over_myopic_refresh_brier"
    ] == pytest.approx(scenario.predicted_overall_gain)


def test_small_audit_is_reproducible_and_covers_exact_primary_gates() -> None:
    first = power.build_audit(replicates=3, bootstrap_samples=30, seed=456)
    second = power.build_audit(replicates=3, bootstrap_samples=30, seed=456)

    assert first == second
    assert first["model_calls"] == 0
    assert first["cost_usd"] == 0.0
    assert first["audit_script_sha256"] == power.sha256_file(
        power.Path(power.__file__).resolve()
    )
    assert first["primary_claim_gate_names"] == list(
        power.smc.PRIMARY_CLAIM_GATE_NAMES
    )
    for result in first["results"]:
        assert set(result["gate_pass_rates"]) == set(
            power.smc.PRIMARY_CLAIM_GATE_NAMES
        )
        assert 0.0 <= result["primary_conjunction"]["rate"] <= 1.0
        assert result["primary_conjunction"][
            "nominal_two_cohort_rate_if_independent"
        ] == pytest.approx(result["primary_conjunction"]["rate"] ** 2)


def test_write_audit_emits_replayable_zero_call_artifacts(tmp_path) -> None:
    audit = power.build_audit(replicates=2, bootstrap_samples=20, seed=789)
    output = tmp_path / "audit.json"

    written = power.write_audit(output, audit)

    assert written["model_calls"] == 0
    assert written["cost_usd"] == 0.0
    assert json.loads(output.read_text()) == audit
    markdown = output.with_suffix(".md").read_text()
    assert "exact frozen 13-gate scorer" in markdown
    assert "operating-characteristic audit" in markdown
    assert "100-replicate implementation calibration" in markdown
