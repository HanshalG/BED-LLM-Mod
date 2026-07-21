from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from scripts.nonmyopic_rock_auc_depth_oracle import run_auc_depth_oracle
from scripts.nonmyopic_rock_depth_oracle import DepthOracleConfig, exhaustive_action_values


def test_auc_aligned_value_weights_earlier_information_more() -> None:
    model = RockDiagnosisModel(get_paper_map("7-8"))
    terminal, _ = exhaustive_action_values(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        depth=3,
    )
    aligned, _ = exhaustive_action_values(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        depth=3,
        planning_utility="entropy_auc",
    )

    assert max(terminal, key=terminal.get) == "move-SOUTH"
    assert max(aligned, key=aligned.get) == "move-SOUTH"
    assert aligned["check-5"] > terminal["check-5"]


def test_small_auc_depth_oracle_is_paired_legal_and_llm_free() -> None:
    summary = run_auc_depth_oracle(
        DepthOracleConfig(
            num_trials=4,
            num_rounds=3,
            seed=24_076,
            bootstrap_replicates=50,
            trial_concurrency=2,
        )
    )

    assert summary["planning_utility"] == "entropy_auc"
    assert all(summary["mechanics"].values())
    assert set(summary["traces"]) == {"1", "2", "3"}
