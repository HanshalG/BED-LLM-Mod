import math

import pytest

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from helpers import load_config
from scripts.nonmyopic_rock_diagnosis_pilot import (
    CandidateProposalError,
    DeterministicCandidateModel,
    LLMCandidateProvider,
    PilotConfig,
    parse_candidate_action_ids,
    run_pilot,
)


def test_parse_candidate_action_ids_is_strict_about_exact_legal_cells() -> None:
    allowed = ("move-EAST", "check-0", "check-1")
    assert parse_candidate_action_ids(
        '```json\n{"action_ids":["move-EAST","check-1"]}\n```',
        allowed_actions=allowed,
        expected_count=2,
    ) == ("move-EAST", "check-1")
    with pytest.raises(CandidateProposalError, match="not legal"):
        parse_candidate_action_ids(
            '{"action_ids":["move-EAST","check-2"]}', allowed_actions=allowed, expected_count=2
        )
    with pytest.raises(CandidateProposalError, match="distinct"):
        parse_candidate_action_ids(
            '{"action_ids":["move-EAST","move-EAST"]}', allowed_actions=allowed, expected_count=2
        )


def test_dry_run_is_paired_legal_and_call_matched() -> None:
    config = PilotConfig(num_trials=2, num_rounds=3, candidate_width=3, bootstrap_replicates=30)
    environment = RockDiagnosisModel(get_paper_map(config.map_name))
    provider = LLMCandidateProvider(DeterministicCandidateModel(config.candidate_width), environment, config)

    summary = run_pilot(provider, config)

    assert set(summary["traces"]) == {"d1_shared", "d2", "d1_call_matched_width"}
    assert summary["mechanics"]["all_selected_actions_legal"]
    assert summary["mechanics"]["all_initial_candidate_cells_shared"]
    assert summary["mechanics"]["width_call_allocation_matches_virtual_depth_two"]
    assert summary["mechanics"]["physical_candidate_requests"] > 0
    assert summary["summary"]["d1_call_matched_width"]["logical_candidate_calls"] >= summary["summary"]["d1_shared"]["logical_candidate_calls"]
    for comparison in summary["paired"].values():
        assert math.isfinite(comparison["final_entropy_reduction_mean"])
        assert all(math.isfinite(value) for value in comparison["final_entropy_reduction_ci95_descriptive"])


def test_confirmatory_mode_requires_a_minimum_paired_sample() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        PilotConfig(num_trials=11, exploratory_only=False).validate()
    PilotConfig(num_trials=12, exploratory_only=False).validate()


def test_confirmatory_dry_run_uses_the_ci_rule_not_the_exploration_promotion_rule() -> None:
    config = PilotConfig(
        num_trials=12,
        num_rounds=2,
        candidate_width=3,
        bootstrap_replicates=30,
        exploratory_only=False,
    )
    environment = RockDiagnosisModel(get_paper_map(config.map_name))
    provider = LLMCandidateProvider(DeterministicCandidateModel(config.candidate_width), environment, config)

    summary = run_pilot(provider, config)

    assert summary["exploratory_only"] is False
    assert "promotion" not in summary
    assert "confirmed" in summary["confirmation"]


def test_bounded_retry_shows_the_rejected_cell_and_legal_set() -> None:
    class InvalidThenLegalModel:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        def chat_complete(
            self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
        ) -> list[str]:
            del temperature, num_responses
            self.messages.append(messages)
            if len(self.messages) == 1:
                return ['{"action_ids":["move-WEST","check-0","check-1"]}']
            return ['{"action_ids":["move-EAST","check-0","check-1"]}']

    config = PilotConfig(candidate_retries=1)
    environment = RockDiagnosisModel(get_paper_map(config.map_name))
    model = InvalidThenLegalModel()
    provider = LLMCandidateProvider(model, environment, config)

    pool = provider.propose(
        trial_index=0,
        position=environment.map_spec.start_position,
        belief=environment.initial_belief,
        history=(),
        label="base",
    )

    assert pool.action_ids == ("move-EAST", "check-0", "check-1")
    assert len(provider.invalid_responses) == 1
    retry = model.messages[1]
    assert retry[-2]["content"] == '{"action_ids":["move-WEST","check-0","check-1"]}'
    assert "move-WEST" not in retry[-1]["content"].split("legal list: ", maxsplit=1)[1]


def test_config_is_nonthinking_and_has_a_bounded_cost_projection() -> None:
    config = load_config("configs/config_nonmyopic_rock_diagnosis_pilot_openrouter.yaml")

    assert config.openrouter_projected_cost_usd == 0.15
    assert config.openrouter_run_budget_usd == 0.50
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_max_output_tokens == 128
    confirmation = load_config("configs/config_nonmyopic_rock_diagnosis_confirmation_openrouter.yaml")
    assert confirmation.openrouter_projected_cost_usd == 0.18
    assert confirmation.openrouter_run_budget_usd == 0.35
    assert confirmation.model_pairs[0].questioner.thinking is False
