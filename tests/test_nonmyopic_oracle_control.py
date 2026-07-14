from pathlib import Path

import numpy as np

from scripts.nonmyopic_oracle_control import (
    OracleControlConfig,
    _select_action,
    candidate_pool,
    expected_information_gain,
    load_frozen_matrix,
    run_oracle_control,
    run_policy,
)


DATA_PATH = Path("data/nonmyopic/uci_zoo.data")


def test_frozen_ucizoo_matrix_is_hashed_and_spot_checked() -> None:
    matrix = load_frozen_matrix(DATA_PATH)

    assert matrix.values.shape == (101, 21)
    assert matrix.names[0] == "aardvark"
    assert "legs_eq_4" in matrix.traits
    assert matrix.source_sha256 == "cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315"


def test_candidate_pool_is_deterministic_nested_and_excludes_asked_traits() -> None:
    history = ((2, True), (7, False))
    narrow = candidate_pool(history, width=3, num_traits=21, seed=1304, trial_index=4)
    wide = candidate_pool(history, width=12, num_traits=21, seed=1304, trial_index=4)

    assert narrow == wide[:3]
    assert narrow == candidate_pool(history, width=3, num_traits=21, seed=1304, trial_index=4)
    assert set(narrow).isdisjoint({2, 7})


def test_exact_eig_rejects_constant_trait_and_policy_never_repeats_actions() -> None:
    matrix = load_frozen_matrix(DATA_PATH)
    support = np.arange(len(matrix.names), dtype=int)
    constant_traits = [index for index in range(len(matrix.traits)) if expected_information_gain(matrix, support, index) == 0.0]
    assert constant_traits == []

    config = OracleControlConfig(num_trials=4, num_rounds=6, bootstrap_replicates=20)
    trace = run_policy(matrix, target=0, width=3, depth=2, config=config, trial_index=0, noise_sd=0.0)
    assert len(trace.actions) == config.num_rounds
    assert len(set(trace.actions)) == len(trace.actions)
    assert len(trace.accuracy) == config.num_rounds


def test_depth_two_truncates_to_one_step_at_the_finite_horizon() -> None:
    matrix = load_frozen_matrix(DATA_PATH)
    config = OracleControlConfig(num_trials=2, num_rounds=4, bootstrap_replicates=20)
    support = np.arange(len(matrix.names), dtype=int)
    history = ((0, True),)

    one_step = _select_action(
        matrix, support, history, width=4, depth=1, config=config, trial_index=1, noise_sd=0.0, remaining_rounds=1
    )
    requested_depth_two = _select_action(
        matrix, support, history, width=4, depth=2, config=config, trial_index=1, noise_sd=0.0, remaining_rounds=1
    )

    assert requested_depth_two == one_step


def test_small_oracle_control_is_exact_and_reports_decision_fields() -> None:
    matrix = load_frozen_matrix(DATA_PATH)
    config = OracleControlConfig(
        num_trials=12,
        num_rounds=4,
        bootstrap_replicates=40,
        restricted_widths=(2, 3),
        primary_width=3,
        score_noise_sds=(0.0, 0.1),
    )

    summary = run_oracle_control(matrix, config)

    assert summary["no_llm_calls"] is True
    assert summary["data"]["num_entities"] == 101
    assert len(summary["restriction"]) == 2
    assert len(summary["noise_frontier"]) == 2
    assert summary["decision"]["verdict"] in {"proceed_to_llm_exploration", "stop_and_discuss"}
