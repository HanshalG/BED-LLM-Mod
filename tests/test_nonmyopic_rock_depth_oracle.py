import math

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from scripts.nonmyopic_rock_depth_oracle import (
    DepthOracleConfig,
    exhaustive_action_values,
    run_depth_oracle,
)


def test_initial_depth_three_value_exposes_two_move_enabling_path() -> None:
    model = RockDiagnosisModel(get_paper_map("7-8"))
    values = {
        depth: max(
            exhaustive_action_values(
                model,
                position=model.map_spec.start_position,
                belief=model.initial_belief,
                depth=depth,
            )[0].values()
        )
        for depth in (1, 2, 3)
    }

    assert values[3] > values[2] > values[1] > 0.0
    assert math.isclose(values[3], math.log(2.0), rel_tol=1e-10)


def test_small_depth_oracle_is_paired_legal_finite_and_llm_free() -> None:
    summary = run_depth_oracle(
        DepthOracleConfig(
            num_trials=4,
            num_rounds=3,
            bootstrap_replicates=50,
            trial_concurrency=2,
        )
    )

    assert summary["stage"] == "depth3_exact_qualification"
    assert all(summary["mechanics"].values())
    for comparison in summary["comparisons"].values():
        assert math.isfinite(comparison["entropy_auc_gain_mean"])
        assert all(math.isfinite(value) for value in comparison["entropy_auc_gain_ci95"])
        assert math.isfinite(comparison["truth_log_probability_auc_gain_mean"])
        assert all(math.isfinite(value) for value in comparison["truth_log_probability_auc_gain_ci95"])
    for depth, traces in summary["traces"].items():
        assert int(depth) in (1, 2, 3)
        assert len(traces) == 4
        assert all(len(trace["steps"]) == 3 for trace in traces)
