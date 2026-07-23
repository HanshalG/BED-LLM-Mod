import math

import numpy as np
import pytest

from environments.rock_diagnosis import (
    RangeGatedRockDiagnosisModel,
    RockDiagnosisModel,
    get_paper_map,
)
from scripts.nonmyopic_rock_diagnosis_oracle import (
    OracleConfig,
    _candidate_cell,
    run_oracle,
    select_action,
)


def test_paper_maps_and_exact_immediate_eig_are_well_formed() -> None:
    paper_map = get_paper_map("5-7")
    assert paper_map.grid_size == 7
    assert paper_map.rock_positions == ((4, 0), (6, 2), (2, 3), (3, 5), (5, 5))
    assert paper_map.start_position == (0, 3)

    model = RockDiagnosisModel(paper_map)
    belief = model.initial_belief
    assert model.expected_information_gain(paper_map.start_position, belief, "move-EAST") == 0.0
    assert model.expected_information_gain(paper_map.start_position, belief, "check-2") > 0.0


def test_range_gated_variant_uses_on_site_accuracy_only_at_target_rock() -> None:
    model = RangeGatedRockDiagnosisModel(get_paper_map("3-6"))
    start = model.map_spec.start_position
    rock = model.map_spec.rock_positions[0]

    assert model.sensor_accuracy(start, 0) == pytest.approx(0.55)
    assert model.sensor_accuracy(rock, 0) == pytest.approx(0.95)


def test_binary_channel_eig_matches_full_joint_entropy_definition() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    rng = np.random.default_rng(812)
    belief = rng.random(len(model.hidden_states))
    belief /= belief.sum()
    position = (2, 3)

    for action in ("check-0", "check-1", "check-2"):
        expected_entropy = sum(
            model.outcome_probability(position, belief, action, outcome)
            * model.entropy(model.posterior(position, belief, action, outcome))
            for outcome in model.outcomes(action)
        )
        full_joint_eig = model.entropy(belief) - expected_entropy
        assert model.expected_information_gain(position, belief, action) == pytest.approx(
            full_joint_eig, abs=1e-12
        )


def test_factorized_belief_marginals_and_entropy_remain_exact_after_updates() -> None:
    model = RockDiagnosisModel(get_paper_map("5-7"))
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    history = (("check-2", "good"), ("move-EAST", None), ("check-2", "bad"))

    for action, outcome in history:
        belief = model.posterior(position, belief, action, outcome)
        position = model.next_position(position, action)

    full_entropy = -float(np.dot(belief[belief > 0.0], np.log(belief[belief > 0.0])))
    assert model.entropy(belief) == pytest.approx(full_entropy, abs=1e-12)
    for rock_id in range(model.num_rocks):
        joint_marginal = sum(
            probability
            for state, probability in zip(model.hidden_states, belief)
            if str(state[rock_id]).lower() == "good"
        )
        assert model.rock_good_probability(belief, rock_id) == pytest.approx(
            joint_marginal, abs=1e-12
        )


def test_vectorized_likelihood_matches_rocksample_sensor_semantics() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    position = (2, 3)
    rock_id = 1
    accuracy = model.sensor_accuracy(position, rock_id)
    good_likelihood = model.likelihood_vector(position, "check-1", "good")
    bad_likelihood = model.likelihood_vector(position, "check-1", "bad")

    for index, state in enumerate(model.hidden_states):
        expected_good = accuracy if str(state[rock_id]).lower() == "good" else 1.0 - accuracy
        assert good_likelihood[index] == pytest.approx(expected_good)
        assert bad_likelihood[index] == pytest.approx(1.0 - expected_good)


def test_positive_rare_observation_below_epsilon_remains_possible() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    rock_position = model.map_spec.rock_positions[0]
    belief = np.zeros(len(model.hidden_states), dtype=float)
    good_indices = [
        index
        for index, state in enumerate(model.hidden_states)
        if str(state[0]).lower() == "good"
    ]
    bad_indices = [index for index in range(len(model.hidden_states)) if index not in good_indices]
    rare_mass = 1e-14
    belief[good_indices] = rare_mass / len(good_indices)
    belief[bad_indices] = (1.0 - rare_mass) / len(bad_indices)

    probability = model.outcome_probability(rock_position, belief, "check-0", "good")
    posterior = model.posterior(rock_position, belief, "check-0", "good")

    assert 0.0 < probability < 1e-12
    assert np.isfinite(posterior).all()
    assert float(posterior.sum()) == pytest.approx(1.0)
    assert model.rock_good_probability(posterior, 0) == pytest.approx(1.0)


def test_canonical_rocksample_7_8_map_is_transcribed_with_exact_belief() -> None:
    benchmark_map = get_paper_map("7-8")
    assert benchmark_map.grid_size == 7
    assert benchmark_map.rock_positions == (
        (1, 0),
        (5, 1),
        (2, 2),
        (3, 2),
        (6, 3),
        (0, 5),
        (3, 5),
        (2, 6),
    )
    assert benchmark_map.start_position == (0, 3)
    assert benchmark_map.source_citation == "Smith and Simmons (2004)"

    model = RockDiagnosisModel(benchmark_map)
    assert len(model.hidden_states) == 256
    assert math.isclose(float(model.initial_belief.sum()), 1.0)


def test_canonical_rocksample_11_11_map_matches_sarsop_benchmark() -> None:
    benchmark_map = get_paper_map("11-11")
    assert benchmark_map.grid_size == 11
    assert benchmark_map.rock_positions == (
        (0, 3),
        (0, 7),
        (1, 8),
        (2, 4),
        (3, 3),
        (3, 8),
        (4, 3),
        (5, 8),
        (6, 1),
        (9, 3),
        (9, 9),
    )
    assert benchmark_map.start_position == (0, 5)
    assert benchmark_map.source_citation == "SARSOP benchmark repository"

    model = RockDiagnosisModel(benchmark_map)
    assert len(model.hidden_states) == 2_048
    assert len(model.legal_actions(benchmark_map.start_position)) == 14
    assert math.isclose(float(model.initial_belief.sum()), 1.0)


def test_frozen_pobax_rocksample_15_15_map_is_reproducible() -> None:
    benchmark_map = get_paper_map("15-15")
    assert benchmark_map.grid_size == 15
    assert benchmark_map.rock_positions == (
        (13, 9),
        (4, 2),
        (13, 8),
        (2, 6),
        (2, 10),
        (10, 1),
        (14, 10),
        (9, 5),
        (5, 12),
        (13, 7),
        (4, 5),
        (3, 9),
        (0, 0),
        (14, 2),
        (3, 7),
    )
    assert benchmark_map.start_position == (0, 7)
    assert benchmark_map.source_citation == "POBAX RockSample generator (JAX key 24098)"

    model = RockDiagnosisModel(benchmark_map)
    assert len(model.hidden_states) == 32_768
    assert len(model.legal_actions(benchmark_map.start_position)) == 18
    assert math.isclose(float(model.initial_belief.sum()), 1.0)


def test_full_candidate_root_exposes_the_dynamic_lookahead_mechanism() -> None:
    config = OracleConfig(map_name="5-7", candidate_widths=(8,))
    model = RockDiagnosisModel(get_paper_map(config.map_name))
    belief = model.initial_belief
    position = model.map_spec.start_position

    d1 = select_action(
        model,
        position=position,
        belief=belief,
        history=(),
        trial_index=0,
        width=8,
        arm="d1_shared",
        planning_depth=1,
        config=config,
    )
    d2 = select_action(
        model,
        position=position,
        belief=belief,
        history=(),
        trial_index=0,
        width=8,
        arm="d2",
        planning_depth=2,
        config=config,
    )

    assert d1.action == "check-2"
    assert d2.action == "move-EAST"
    assert d2.scores["move-EAST"] > d2.scores["check-2"]


def test_candidate_prefixes_are_nested_and_width_matches_virtual_calls() -> None:
    config = OracleConfig(map_name="5-7")
    model = RockDiagnosisModel(get_paper_map(config.map_name))
    position = model.map_spec.start_position
    belief = model.initial_belief
    narrow = _candidate_cell(
        model,
        position=position,
        history=(),
        trial_index=3,
        width=2,
        config=config,
        label="base",
    )
    wide = _candidate_cell(
        model,
        position=position,
        history=(),
        trial_index=3,
        width=4,
        config=config,
        label="base",
    )
    assert narrow == wide[:2]

    d2 = select_action(
        model,
        position=position,
        belief=belief,
        history=(),
        trial_index=3,
        width=3,
        arm="d2",
        planning_depth=2,
        config=config,
    )
    width = select_action(
        model,
        position=position,
        belief=belief,
        history=(),
        trial_index=3,
        width=3,
        arm="d1_call_matched_width",
        planning_depth=2,
        config=config,
    )
    assert d2.base_candidate_pool == width.base_candidate_pool
    assert set(d2.base_candidate_pool).issubset(width.candidate_pool)
    assert d2.candidate_call_budget == width.candidate_call_budget
    assert width.call_budget_matches_virtual_depth_two


def test_small_oracle_run_is_paired_finite_and_llm_free() -> None:
    summary = run_oracle(
        OracleConfig(
            map_name="3-6",
            num_trials=8,
            num_rounds=3,
            candidate_widths=(2, 3),
            bootstrap_replicates=50,
        )
    )

    assert summary["no_llm_calls"]
    assert set(summary["widths"]) == {"2", "3"}
    for row in summary["widths"].values():
        assert row["mechanics"]["initial_base_candidate_cells_shared"]
        assert row["mechanics"]["initial_width_contains_shared_base"]
        assert row["mechanics"]["width_cells_match_virtual_depth_two_cells"]
        assert row["mechanics"]["all_selected_actions_legal"]
        for comparison in row["comparisons"].values():
            assert math.isfinite(comparison["final_entropy_reduction_mean"])
            assert all(math.isfinite(value) for value in comparison["final_entropy_reduction_ci95"])
        for arm in row["traces"].values():
            assert len(arm) == 8
            assert all(np.isfinite(trace["final_entropy"]) for trace in arm)
