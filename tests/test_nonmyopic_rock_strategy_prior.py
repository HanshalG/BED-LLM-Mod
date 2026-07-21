import json
import math
from dataclasses import asdict

import pytest

from environments.rock_diagnosis import RockDiagnosisModel, RockStrategyExecutor, get_paper_map
from helpers import load_config
from scripts.nonmyopic_rock_strategy_prior import (
    ARMS,
    DeterministicStrategyModel,
    L1Config,
    LLMRockStrategyProvider,
    StrategyProposalError,
    _posterior_prompt_lines,
    parse_branch_strategy_cell,
    parse_strategy_cell,
    parse_width_cell,
    run_l1_anchor,
)


def test_large_factorized_belief_uses_exact_compact_prompt_summary() -> None:
    model = RockDiagnosisModel(get_paper_map("7-8"))

    lines = _posterior_prompt_lines(model, model.initial_belief)

    assert len(lines) == 1
    assert "factorizes over rocks" in lines[0]
    assert "specify the complete posterior exactly" in lines[0]


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


def test_branch_policy_schema_compiles_explicit_outcome_actions() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    position = model.map_spec.start_position
    response = json.dumps(
        {
            "strategies": [
                {
                    "name": "move then sense",
                    "description": "Move east to improve a subsequent check.",
                    "root_action": "move-EAST",
                    "followups": {"none": "check-0"},
                },
                {
                    "name": "adaptive remote check",
                    "description": "Use the first outcome to choose the second action.",
                    "root_action": "check-0",
                    "followups": {"good": "move-EAST", "bad": "check-1"},
                },
            ]
        }
    )

    move_strategy, check_strategy = parse_branch_strategy_cell(
        response,
        model=model,
        position=position,
        horizon=2,
        expected_count=2,
    )
    executor = RockStrategyExecutor(model)
    assert executor.choose_action(
        move_strategy,
        position=position,
        belief=model.initial_belief,
        history=(),
        strategy_step=0,
    ) == "move-EAST"
    assert executor.choose_action(
        move_strategy,
        position=model.next_position(position, "move-EAST"),
        belief=model.initial_belief,
        history=(("move-EAST", None),),
        strategy_step=1,
    ) == "check-0"
    for outcome, expected in (("good", "move-EAST"), ("bad", "check-1")):
        posterior = model.posterior(position, model.initial_belief, "check-0", outcome)
        assert executor.choose_action(
            check_strategy,
            position=position,
            belief=posterior,
            history=(("check-0", outcome),),
            strategy_step=1,
        ) == expected


def test_branch_policy_uses_the_last_complete_json_fence_after_self_correction() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    position = model.map_spec.start_position
    duplicate = {
        "strategies": [
            {
                "name": f"duplicate-{index}",
                "description": "An invalid first draft with duplicate behavior.",
                "root_action": "check-0",
                "followups": {},
            }
            for index in range(2)
        ]
    }
    corrected = {
        "strategies": [
            {
                "name": "check zero",
                "description": "Check the first rock now.",
                "root_action": "check-0",
                "followups": {},
            },
            {
                "name": "check one",
                "description": "Check the second rock now.",
                "root_action": "check-1",
                "followups": {},
            },
        ]
    }
    response = (
        f"```json\n{json.dumps(duplicate)}\n```\n"
        "I noticed the duplicate and corrected the complete response.\n"
        f"```json\n{json.dumps(corrected)}\n```"
    )

    strategies = parse_branch_strategy_cell(
        response,
        model=model,
        position=position,
        horizon=1,
        expected_count=2,
    )

    assert [json.loads(strategy.raw_text)["root_action"] for strategy in strategies] == [
        "check-0",
        "check-1",
    ]


def test_branch_policy_schema_rejects_missing_branches_and_illegal_child_actions() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    position = model.map_spec.start_position
    base = {
        "name": "invalid branch",
        "description": "A deliberately invalid branch-policy test fixture.",
        "root_action": "check-0",
        "followups": {"good": "check-0"},
    }
    with pytest.raises(StrategyProposalError, match="followups must contain exactly"):
        parse_branch_strategy_cell(
            json.dumps({"strategies": [base]}),
            model=model,
            position=position,
            horizon=2,
            expected_count=1,
        )

    base["followups"] = {"good": {"check-0": "none"}, "bad": "check-1"}
    with pytest.raises(StrategyProposalError, match="must be a string action ID"):
        parse_branch_strategy_cell(
            json.dumps({"strategies": [base]}),
            model=model,
            position=position,
            horizon=2,
            expected_count=1,
        )

    base["root_action"] = "move-NORTH"
    base["followups"] = {"none": "move-WEST"}
    with pytest.raises(StrategyProposalError, match="must be legal"):
        parse_branch_strategy_cell(
            json.dumps({"strategies": [base]}),
            model=model,
            position=position,
            horizon=2,
            expected_count=1,
        )


def test_branch_policy_horizon_one_prompt_requires_unique_roots_and_empty_followups() -> None:
    model = RockDiagnosisModel(get_paper_map("5-7"))
    config = L1Config(num_strategies=6, strategy_schema="branch_policy_v2")
    provider = LLMRockStrategyProvider(DeterministicStrategyModel(), config)

    messages = provider._branch_strategy_messages(
        model,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=1,
    )

    prompt = messages[-1]["content"]
    assert "choose exactly 6 different legal root_action IDs" in prompt
    assert "Every strategy must use followups:{} exactly" in prompt
    assert "even for movement roots" in prompt
    assert "MOVEMENT_ROOT_SLOTS=" not in prompt
    assert "machine-assigned movement slots" not in prompt
    assert "At horizon 2" not in prompt


def test_branch_policy_horizon_one_repairs_only_out_of_horizon_followups() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    response = json.dumps(
        {
            "strategies": [
                {
                    "name": "move now",
                    "description": "Move east as the current action.",
                    "root_action": "move-EAST",
                    "followups": {"none": "check-0"},
                },
                {
                    "name": "check now",
                    "description": "Check the first rock as the current action.",
                    "root_action": "check-0",
                    "followups": {"good": "check-1", "bad": "move-EAST"},
                },
            ]
        }
    )

    class TerminalBranchModel:
        def chat_complete(
            self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
        ) -> list[str]:
            del messages, temperature, num_responses
            return [response]

    config = L1Config(num_strategies=2, strategy_schema="branch_policy_v2")
    provider = LLMRockStrategyProvider(TerminalBranchModel(), config)
    cell = provider.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=1,
    )

    assert len(cell.strategies) == 2
    assert provider.invalid_responses == []
    assert provider.terminal_followup_repairs == 2
    assert all(json.loads(strategy.raw_text)["followups"] == {} for strategy in cell.strategies)


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


def test_branch_feedback_names_omitted_physical_movement_roots() -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    valid_response = DeterministicStrategyModel().chat_complete(
        LLMRockStrategyProvider(
            DeterministicStrategyModel(),
            L1Config(num_strategies=4, strategy_schema="branch_policy_v2"),
        )._strategy_messages(
            model,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            history=(),
            horizon=2,
        ),
        0.0,
    )[0]
    duplicate_response = json.dumps(
        {
            "strategies": [
                {
                    "name": "east zero",
                    "description": "Move east, then check rock zero.",
                    "root_action": "move-EAST",
                    "followups": {"none": "check-0"},
                },
                {
                    "name": "east one",
                    "description": "Move east, then check rock one.",
                    "root_action": "move-EAST",
                    "followups": {"none": "check-1"},
                },
                {
                    "name": "check zero",
                    "description": "Check rock zero now.",
                    "root_action": "check-0",
                    "followups": {"good": "check-0", "bad": "check-1"},
                },
                {
                    "name": "check one",
                    "description": "Check rock one now.",
                    "root_action": "check-1",
                    "followups": {"good": "check-1", "bad": "check-2"},
                },
            ]
        }
    )

    class DuplicateThenValidModel:
        def __init__(self) -> None:
            self.messages = []

        def chat_complete(self, messages, temperature, num_responses=1):
            del temperature, num_responses
            self.messages.append(messages)
            return [duplicate_response if len(self.messages) == 1 else valid_response]

    chat_model = DuplicateThenValidModel()
    provider = LLMRockStrategyProvider(
        chat_model, L1Config(num_strategies=4, strategy_schema="branch_policy_v2")
    )
    provider.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )

    feedback = chat_model.messages[1][-1]["content"]
    assert "Legal movement root IDs are" in feedback
    assert "omitted movement root IDs are ['move-NORTH', 'move-SOUTH']" in feedback
    assert "different target intentions may not repeat one root ID" in feedback


def test_provider_resumes_revalidated_cells_without_model_calls(tmp_path) -> None:
    model = RockDiagnosisModel(get_paper_map("3-6"))
    config = L1Config(
        map_names=("3-6",),
        num_trials_per_map=1,
        num_rounds=2,
        num_strategies=4,
        strategy_schema="branch_policy_v2",
    )
    original = LLMRockStrategyProvider(DeterministicStrategyModel(), config)
    original.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )
    original.propose_width_order(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
    )
    failure_path = tmp_path / "L1_FAILURE.json"
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "L1",
                "status": "failed_closed",
                "error": "later cell failed",
                "config": asdict(config),
                "candidate_requests": original.physical_requests,
                "invalid_responses": [{"error": "preserved"}],
                "usage": {"run_cost_usd": 0.1},
            }
        ),
        encoding="utf-8",
    )

    class NoCallModel:
        def chat_complete(self, *args, **kwargs):
            raise AssertionError("a resumed cell must not call the model")

    resumed = LLMRockStrategyProvider(NoCallModel(), config)
    resume = resumed.load_failure_cache(failure_path)
    strategy = resumed.propose_strategies(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )
    width = resumed.propose_width_order(
        model,
        map_name="3-6",
        trial_index=0,
        position=model.map_spec.start_position,
        belief=model.initial_belief,
        history=(),
    )

    assert strategy.cache_hit and width.cache_hit
    assert resume["accepted_cells_reused"] == 2
    assert resume["rejected_responses_preserved"] == 1
    assert resumed.cache_hits == 2
    assert len(resumed.physical_requests) == 2
    assert resumed.invalid_responses == [{"error": "preserved"}]


def test_provider_resume_requires_an_exact_config_match(tmp_path) -> None:
    config = L1Config(map_names=("3-6",), num_trials_per_map=1)
    payload = {
        "schema_version": 1,
        "stage": "L1",
        "status": "failed_closed",
        "config": {**asdict(config), "seed": config.seed + 1},
        "candidate_requests": [],
        "invalid_responses": [],
    }
    failure_path = tmp_path / "L1_FAILURE.json"
    failure_path.write_text(json.dumps(payload), encoding="utf-8")
    provider = LLMRockStrategyProvider(DeterministicStrategyModel(), config)

    with pytest.raises(ValueError, match="config does not exactly match"):
        provider.load_failure_cache(failure_path)


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


def test_small_branch_policy_anchor_preserves_pairing_and_compute_controls() -> None:
    config = L1Config(
        map_names=("3-6",),
        num_trials_per_map=2,
        num_rounds=3,
        num_strategies=4,
        bootstrap_replicates=30,
        strategy_schema="branch_policy_v2",
        primary_endpoint="entropy_auc",
    )
    provider = LLMRockStrategyProvider(DeterministicStrategyModel(), config)

    summary = run_l1_anchor(provider, config)

    assert summary["mechanics"]["terminal_cell_failures"] == 0
    assert summary["mechanics"]["all_selected_actions_legal"]
    assert summary["mechanics"]["initial_strategy_cells_shared_with_d1"]
    assert summary["mechanics"]["width_exact_scorer_units_match_strategy_eig"]
    assert summary["mechanics"]["random_strategy_cells_have_k_candidates"]
    comparison = summary["maps"]["3-6"]["paired"]["strategy_eig_minus_shared_d1"]
    assert math.isfinite(comparison["entropy_auc_gain_mean"])
    assert len(comparison["entropy_auc_gain_ci95"]) == 2
    assert math.isfinite(comparison["truth_log_probability_auc_gain_mean"])
    assert len(comparison["truth_log_probability_auc_gain_ci95"]) == 2
    for trace in summary["traces"]["3-6"]["random_strategy"]:
        for step in trace["steps"][:-1]:
            move_roots = [root for root in step["candidate_roots"] if root.startswith("move-")]
            assert len(move_roots) == 2
            assert len(set(move_roots)) == 2
            policies = [json.loads(text) for text in step["candidate_strategies"]]
            assert all(
                policy["followups"]["none"].startswith("check-")
                for policy in policies
                if policy["root_action"].startswith("move-")
            )


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
