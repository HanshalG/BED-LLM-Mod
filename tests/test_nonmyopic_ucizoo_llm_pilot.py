import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.nonmyopic_oracle_control import load_frozen_matrix
from scripts.nonmyopic_ucizoo_llm_pilot import (
    CandidateProposalError,
    LLMCandidateProvider,
    PilotConfig,
    parse_candidate_trait_ids,
    run_pilot,
    run_policy,
)


DATA_PATH = Path("data/nonmyopic/uci_zoo.data")


class LegalEchoModel:
    """A zero-cost fake that always picks the first currently legal IDs in the prompt."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_complete(self, messages, temperature, num_responses=1):
        del temperature
        assert num_responses == 1
        self.calls += 1
        content = messages[-1]["content"]
        legal = content.split("Unasked legal trait IDs:\n", maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        avoid_marker = "For this width-expansion cell, avoid re-proposing these IDs whenever at least "
        avoid = [] if avoid_marker not in content else content.rsplit("\n", maxsplit=1)[1].split(", ")
        chosen = [trait for trait in legal if trait not in avoid][:3]
        if len(chosen) < 3:
            chosen.extend(trait for trait in legal if trait not in chosen)
        return [json.dumps({"trait_ids": chosen[:3]})]


def _provider(config: PilotConfig) -> tuple[LLMCandidateProvider, LegalEchoModel]:
    model = LegalEchoModel()
    return LLMCandidateProvider(model, load_frozen_matrix(DATA_PATH), config), model


def test_candidate_parser_rejects_unknown_duplicate_and_previously_asked_ids() -> None:
    allowed = ("hair", "feathers", "eggs")

    with pytest.raises(CandidateProposalError, match="unknown"):
        parse_candidate_trait_ids(
            '{"trait_ids":["hair","unknown"]}',
            allowed_traits=allowed,
            asked_actions=set(),
            expected_count=2,
        )
    with pytest.raises(CandidateProposalError, match="distinct"):
        parse_candidate_trait_ids(
            '{"trait_ids":["hair","hair"]}',
            allowed_traits=allowed,
            asked_actions=set(),
            expected_count=2,
        )
    with pytest.raises(CandidateProposalError, match="already asked"):
        parse_candidate_trait_ids(
            '{"trait_ids":["hair","feathers"]}',
            allowed_traits=allowed,
            asked_actions={0},
            expected_count=2,
        )


def test_candidate_parser_accepts_only_an_exact_json_fence_wrapper() -> None:
    allowed = ("hair", "feathers", "eggs")

    assert parse_candidate_trait_ids(
        '```json\n{"trait_ids":["hair","feathers"]}\n```',
        allowed_traits=allowed,
        asked_actions=set(),
        expected_count=2,
    ) == (0, 1)
    with pytest.raises(CandidateProposalError, match="response is not a JSON object"):
        parse_candidate_trait_ids(
            'Proposed traits:\n```json\n{"trait_ids":["hair","feathers"]}\n```',
            allowed_traits=allowed,
            asked_actions=set(),
            expected_count=2,
        )
    with pytest.raises(CandidateProposalError, match="response is not a JSON object"):
        parse_candidate_trait_ids(
            '```\n{"trait_ids":["hair","feathers"]}\n```',
            allowed_traits=allowed,
            asked_actions=set(),
            expected_count=2,
        )
    with pytest.raises(CandidateProposalError, match="incomplete JSON fence"):
        parse_candidate_trait_ids(
            '```json\n{"trait_ids":["hair","feathers"]}',
            allowed_traits=allowed,
            asked_actions=set(),
            expected_count=2,
        )


def test_provider_caches_shared_candidate_cell_without_any_fallback() -> None:
    config = PilotConfig(num_trials=2, num_rounds=2, bootstrap_replicates=20)
    provider, model = _provider(config)
    matrix = load_frozen_matrix(DATA_PATH)

    first = provider.propose(trial_index=0, history=(), support=matrix.values[:, 0].nonzero()[0], label="root")
    second = provider.propose(trial_index=0, history=(), support=matrix.values[:, 0].nonzero()[0], label="root")

    assert model.calls == 1
    assert first.actions == second.actions
    assert not first.cache_hit
    assert second.cache_hit
    assert first.trait_ids == ("hair", "feathers", "eggs")


def test_depth_two_and_matched_width_obey_per_state_call_contract() -> None:
    config = PilotConfig(num_trials=2, num_rounds=3, bootstrap_replicates=20)
    provider, _model = _provider(config)
    matrix = load_frozen_matrix(DATA_PATH)

    depth_two = run_policy(matrix, provider, config, arm="d2", trial_index=0, target=0)
    width = run_policy(matrix, provider, config, arm="d1_matched_width", trial_index=0, target=0)

    assert depth_two.steps[-1].candidate_call_budget == 1
    assert depth_two.steps[-1].virtual_depth_two_call_budget == 1
    assert all(
        step.candidate_call_budget == step.virtual_depth_two_call_budget for step in width.steps
    )
    assert width.steps[0].base_candidate_pool == depth_two.steps[0].base_candidate_pool


def test_pilot_uses_paired_targets_shared_roots_and_reports_mechanics() -> None:
    config = PilotConfig(num_trials=3, num_rounds=3, bootstrap_replicates=40)
    provider, model = _provider(config)
    summary = run_pilot(load_frozen_matrix(DATA_PATH), provider, config)

    assert model.calls == summary["mechanics"]["physical_candidate_requests"]
    assert summary["mechanics"]["no_invalid_llm_candidate_responses"]
    assert summary["mechanics"]["all_initial_candidate_cells_shared"]
    assert summary["mechanics"]["width_call_allocation_matches_virtual_depth_two"]
    assert len(summary["targets"]) == 3
    assert set(summary["traces"]) == {"d1_shared", "d2", "d1_matched_width"}
    assert summary["summary"]["d1_matched_width"]["logical_candidate_calls"] >= summary["summary"]["d1_shared"]["logical_candidate_calls"]


def test_pilot_openrouter_config_is_bounded_nonthinking() -> None:
    config = load_config("configs/config_nonmyopic_ucizoo_llm_pilot_openrouter.yaml")

    assert config.model_pairs[0].questioner.backend == "openrouter"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_run_budget_usd == 0.50
    assert config.openrouter_projected_cost_usd == 0.20
    assert config.openrouter_max_output_tokens == 128
