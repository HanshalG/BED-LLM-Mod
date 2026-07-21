import math

import numpy as np

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
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
