import json
import math

import pytest

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from helpers import load_config
from scripts.nonmyopic_rock_strategy_prior import (
    ARMS,
    DeterministicStrategyModel,
    L1Config,
    LLMRockStrategyProvider,
    StrategyProposalError,
    parse_strategy_cell,
    parse_width_cell,
    run_l1_anchor,
)


def _deterministic_strategy_response(model: RockDiagnosisModel, count: int) -> str:
    messages = [
        {"role": "system", "content": "test"},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Return exactly {count} distinct strategies using this schema:",
                    f"Rock coordinates by ID: {list(enumerate(model.map_spec.rock_positions))}.",
                ]
            ),
        },
    ]
    return DeterministicStrategyModel().chat_complete(messages, 0.0)[0]


def test_parse_strategy_cell_requires_a_complete_distinct_cell() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    response = _deterministic_strategy_response(model, 3)

    strategies = parse_strategy_cell(response, model=model, expected_count=3)

    assert len(strategies) == 3
    payload = json.loads(response)
    payload["strategies"][1] = payload["strategies"][0]
    with pytest.raises(StrategyProposalError, match="distinct"):
        parse_strategy_cell(json.dumps(payload), model=model, expected_count=3)
    with pytest.raises(StrategyProposalError, match="exactly 3"):
        parse_strategy_cell('{"strategies":[]}', model=model, expected_count=3)


def test_parse_width_cell_requires_every_legal_action_once() -> None:
    allowed = ("move-EAST", "check-0", "check-1")
    assert parse_width_cell(
        '{"action_ids":["check-1","move-EAST","check-0"]}', allowed_actions=allowed
    ) == ("check-1", "move-EAST", "check-0")
    with pytest.raises(StrategyProposalError, match="exactly once"):
        parse_width_cell(
            '{"action_ids":["move-EAST","check-0","check-0"]}', allowed_actions=allowed
        )


def test_provider_uses_one_feedback_retry_for_rollout_illegality() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    valid_response = _deterministic_strategy_response(model, 2)

    class InvalidThenValidModel:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        def chat_complete(
            self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
        ) -> list[str]:
            del temperature, num_responses
            self.messages.append(messages)
            if len(self.messages) == 1:
                bad = {
                    "strategies": [
                        {
                            "name": f"illegal-{index}",
                            "description": "Move outside the west boundary.",
                            "rules": [
                                {"when": [], "action": {"kind": "move", "direction": "WEST"}}
                            ],
                        }
                        for index in range(2)
                    ]
                }
                return [json.dumps(bad)]
            return [valid_response]

    chat_model = InvalidThenValidModel()
    provider = LLMRockStrategyProvider(chat_model, L1Config(num_strategies=2))

    cell = provider.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )

    assert len(cell.strategies) == 2
    assert len(provider.invalid_responses) == 1
    assert len(chat_model.messages) == 2
    assert "compiled illegal action" in chat_model.messages[1][-1]["content"]
    assert chat_model.messages[1][-2]["role"] == "assistant"


def test_provider_feedback_explains_current_root_mix() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    valid_response = _deterministic_strategy_response(model, 2)

    class AllMovesThenValidModel:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, str]]] = []

        def chat_complete(
            self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
        ) -> list[str]:
            del temperature, num_responses
            self.messages.append(messages)
            if len(self.messages) > 1:
                return [valid_response]
            cell = {
                "strategies": [
                    {
                        "name": f"move-{rock_id}",
                        "description": "Move first and check after reaching the target.",
                        "rules": [
                            {
                                "when": [],
                                "action": {
                                    "kind": "target_rock",
                                    "rock_id": rock_id,
                                    "path": "x_first",
                                },
                            }
                        ],
                    }
                    for rock_id in range(2)
                ]
            }
            return [json.dumps(cell)]

    chat_model = AllMovesThenValidModel()
    provider = LLMRockStrategyProvider(chat_model, L1Config(num_strategies=2))

    provider.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )

    assert len(provider.invalid_responses) == 1
    feedback = chat_model.messages[1][-1]["content"]
    assert "compiled root actions were" in feedback
    assert "unconditional check fallback" in feedback


def test_small_dry_anchor_preserves_pairing_and_compute_controls() -> None:
    config = L1Config(
        map_names=("3-6",),
        num_trials_per_map=2,
        num_rounds=3,
        num_strategies=3,
        bootstrap_replicates=30,
    )
    provider = LLMRockStrategyProvider(DeterministicStrategyModel(), config)

    summary = run_l1_anchor(provider, config)

    assert set(summary["traces"]["3-6"]) == set(ARMS)
    assert all(len(summary["traces"]["3-6"][arm]) == 2 for arm in ARMS)
    mechanics = summary["mechanics"]
    assert mechanics["terminal_cell_failures"] == 0
    assert mechanics["all_selected_actions_legal"]
    assert mechanics["initial_strategy_cells_shared_with_d1"]
    assert mechanics["width_logical_llm_calls_match_strategy_eig"]
    assert mechanics["width_exact_scorer_units_match_strategy_eig"]
    assert mechanics["rollout_scoring_llm_calls"] == 0
    for arm in ARMS:
        assert math.isfinite(summary["maps"]["3-6"]["summary"][arm]["final_entropy_mean"])


def test_trial_concurrency_preserves_paired_traces() -> None:
    common = {
        "map_names": ("3-6",),
        "num_trials_per_map": 2,
        "num_rounds": 2,
        "num_strategies": 3,
        "bootstrap_replicates": 30,
    }
    serial_config = L1Config(**common, trial_concurrency=1)
    parallel_config = L1Config(**common, trial_concurrency=2)

    serial = run_l1_anchor(
        LLMRockStrategyProvider(DeterministicStrategyModel(), serial_config), serial_config
    )
    parallel = run_l1_anchor(
        LLMRockStrategyProvider(DeterministicStrategyModel(), parallel_config), parallel_config
    )

    assert serial["traces"] == parallel["traces"]
    assert serial["maps"] == parallel["maps"]


def test_l1_openrouter_config_is_nonthinking_and_bounded() -> None:
    config = load_config("configs/config_nonmyopic_rock_strategy_l1_openrouter.yaml")

    assert config.model_pairs[0].questioner.model == "google/gemma-4-26b-a4b-it"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_max_output_tokens == 4096
    assert config.openrouter_concurrency == 128
    assert config.openrouter_projected_cost_usd == 0.8
    assert config.openrouter_run_budget_usd == 1.0


def test_l1_model_scale_probe_config_has_a_bounded_reasoning_generator() -> None:
    config = load_config("configs/config_nonmyopic_rock_strategy_l1_gemma31b_thinking_openrouter.yaml")
    questioner = config.model_pairs[0].questioner

    assert questioner.model == "google/gemma-4-31b-it"
    assert questioner.thinking is True
    assert questioner.thinking_max_new_tokens == 1024
    assert questioner.thinking_final_max_new_tokens == 256
    assert config.openrouter_projected_cost_usd == 1.5
    assert config.openrouter_run_budget_usd == 2.0
