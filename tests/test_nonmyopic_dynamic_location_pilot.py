import pytest

from helpers import load_config
from scripts.nonmyopic_dynamic_location_pilot import (
    DeterministicCandidateModel,
    LLMCandidateProvider,
    PilotConfig,
    PromptRandomLegalCandidateModel,
    _grid,
    parse_candidate_action_ids,
    run_pilot,
)


def test_action_parser_accepts_exact_json_fence_and_rejects_infeasible_id() -> None:
    assert parse_candidate_action_ids(
        '```json\n{"action_ids":["q_2","q_5"]}\n```',
        allowed_actions=(2, 5, 8),
        expected_count=2,
    ) == (2, 5)
    with pytest.raises(Exception, match="unknown or infeasible"):
        parse_candidate_action_ids(
            '{"action_ids":["q_2","q_7"]}', allowed_actions=(2, 5, 8), expected_count=2
        )


def test_dynamic_pilot_dry_run_preserves_sharing_legality_and_call_matching() -> None:
    config = PilotConfig(num_trials=2, num_rounds=3, bootstrap_replicates=40)
    provider = LLMCandidateProvider(DeterministicCandidateModel(), _grid(config), config)

    summary = run_pilot(provider, config)

    assert summary["mechanics"]["terminal_candidate_cell_failures"] == 0
    assert summary["mechanics"]["all_selected_actions_legal"]
    assert summary["mechanics"]["all_initial_candidate_cells_shared"]
    assert summary["mechanics"]["width_call_allocation_matches_virtual_depth_two"]
    assert summary["mechanics"]["raw_rejected_candidate_attempts"] == 0
    assert summary["summary"]["d2"]["logical_candidate_calls"] > summary["summary"]["d1_shared"]["logical_candidate_calls"]
    assert all(trace["steps"][0]["fixed_origin"] for trace in summary["traces"]["d2"])


def test_random_legal_candidate_control_is_exactly_legal_and_free() -> None:
    config = PilotConfig(num_trials=2, num_rounds=3, grid_size=21, bootstrap_replicates=40)
    model = PromptRandomLegalCandidateModel()
    provider = LLMCandidateProvider(model, _grid(config), config)

    summary = run_pilot(provider, config)

    assert summary["mechanics"]["terminal_candidate_cell_failures"] == 0
    assert summary["mechanics"]["all_selected_actions_legal"]
    assert summary["mechanics"]["width_call_allocation_matches_virtual_depth_two"]
    assert model.usage_snapshot()["cost_usd"] == 0.0


def test_dynamic_pilot_openrouter_config_is_bounded_nonthinking() -> None:
    config = load_config("configs/config_nonmyopic_dynamic_location_pilot_openrouter.yaml")

    assert config.model_pairs[0].questioner.backend == "openrouter"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_run_budget_usd == 0.50
    assert config.openrouter_projected_cost_usd == 0.15
    assert config.openrouter_max_output_tokens == 128
