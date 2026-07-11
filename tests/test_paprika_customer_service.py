from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from core.experiment import run_from_config
from environments.paprika_customer_service import PaprikaAction, load_paprika_tasks
from environments.paprika_customer_service.parsing import parse_distribution
from helpers import Config, load_config
from methods.categorical_eig import categorical_eig


FIXTURE = Path(__file__).parent / "fixtures" / "paprika_customer_service_tiny.json"


class RoutingQuestioner:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_calls = 0
        self.prompt_texts = []

    def chat_complete(self, messages, temperature, num_responses=1):
        del temperature, num_responses
        self.calls += 1
        text = "\n".join(message["content"] for message in messages)
        self.prompt_texts.append(text)
        if '"refined_hypotheses"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"refined_hypotheses": [f"refined cause {self.calls}-{index} with remedy" for index in range(count)]})]
        if '"keep_indices"' in text:
            indices = [int(value) for value in re.findall(r"^(\d+):", text, re.MULTILINE)]
            return [json.dumps({"keep_indices": indices})]
        if '"hypotheses"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"hypotheses": [f"cause {index} and remedy {index}" for index in range(count)]})]
        if '"candidates"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"candidates": [{"query": f"Check diagnostic {index}?", "kind": "diagnostic", "outcomes": ["positive", "negative", "unknown"]} for index in range(count)]})]
        if '"probabilities"' in text:
            return [json.dumps({"probabilities": {"positive": 0.7, "negative": 0.2, "unknown": 0.1}})]
        if '"terminal_consistent"' in text:
            return [json.dumps({"terminal_consistent": True})]
        if '"consistent"' in text:
            return [json.dumps({"consistent": True})]
        if '"outcome"' in text and '"clean"' in text:
            return [json.dumps({"outcome": "positive", "clean": True})]
        if "Reply with <VALID>" in text:
            return ["<NOTVALID>"]
        raise AssertionError(f"Unexpected prompt: {text}")

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=None):
        del block_size, max_new_tokens
        self.batch_calls += 1
        return [self.chat_complete(messages, temperature)[0] for messages in batch_messages]

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("Paprika uses explicit categorical JSON likelihoods")


class RoutingCustomer:
    def __init__(self) -> None:
        self.calls = 0

    def chat_complete(self, messages, temperature, num_responses=1):
        del messages, temperature, num_responses
        self.calls += 1
        return ["The diagnostic result is positive."]


class SolvingCustomer(RoutingCustomer):
    def chat_complete(self, messages, temperature, num_responses=1):
        del messages, temperature, num_responses
        self.calls += 1
        return ["Goal reached"]


class EmbeddedGoalCustomer(RoutingCustomer):
    def chat_complete(self, messages, temperature, num_responses=1):
        del messages, temperature, num_responses
        self.calls += 1
        return ["The device works correctly now. Goal reached."]


class FlakyCandidateQuestioner(RoutingQuestioner):
    def __init__(self) -> None:
        super().__init__()
        self.failed_candidate_once = False

    def chat_complete(self, messages, temperature, num_responses=1):
        text = messages[-1]["content"]
        if '"candidates"' in text and not self.failed_candidate_once:
            self.failed_candidate_once = True
            self.calls += 1
            self.prompt_texts.append(text)
            return ["not json"]
        return super().chat_complete(messages, temperature, num_responses)


class FlakyLikelihoodQuestioner(RoutingQuestioner):
    def __init__(self) -> None:
        super().__init__()
        self.failed_likelihood_once = False

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=None):
        responses = super().chat_complete_messages_batched(
            batch_messages, temperature, block_size, max_new_tokens
        )
        if not self.failed_likelihood_once:
            for index, messages in enumerate(batch_messages):
                if '"probabilities"' in messages[-1]["content"]:
                    responses[index] = "{bad likelihood}"
                    self.failed_likelihood_once = True
                    break
        return responses


class AlwaysValidJudgeQuestioner(RoutingQuestioner):
    def chat_complete(self, messages, temperature, num_responses=1):
        text = "\n".join(message["content"] for message in messages)
        if "Reply with <VALID>" in text:
            self.calls += 1
            self.prompt_texts.append(text)
            return ["<VALID>"]
        return super().chat_complete(messages, temperature, num_responses)


def test_load_released_shape_exposes_private_solution() -> None:
    tasks = load_paprika_tasks(FIXTURE, split="eval")
    assert len(tasks) == 5
    assert tasks[0].scenario == "A refrigerator is beeping and not cooling."
    assert tasks[0].solution == "The door is ajar; closing it restores cooling."
    assert tasks[0].task_id == "customer_service:eval:0000"


def test_official_hash_verification_rejects_fixture() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_paprika_tasks(FIXTURE, verify_official_hash=True)


def test_action_requires_three_to_five_unique_outcomes() -> None:
    with pytest.raises(ValueError, match="3-5"):
        PaprikaAction("Check it", ("yes", "no"), "scenario")
    with pytest.raises(ValueError, match="unique"):
        PaprikaAction("Check it", ("yes", "YES", "unknown"), "scenario")


def test_candidate_kind_downgrades_checks_but_keeps_explicit_corrections() -> None:
    model = RoutingQuestioner()
    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=1,
    )
    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    env = PaprikaCustomerServiceEnvironment(config, RoutingCustomer())
    diagnostic = env._parse_candidates(
        json.dumps({"candidates": [{"query": "Can you check the setting?", "kind": "solution", "outcomes": ["yes", "no", "unknown"]}]}),
        "scenario", [], 1,
    )[0]
    solution = env._parse_candidates(
        json.dumps({"candidates": [{"query": "Please replace the depleted ribbon.", "kind": "diagnostic", "outcomes": ["fixed", "not fixed", "cannot do"]}]}),
        "scenario", [], 1,
    )[0]
    increased = env._parse_candidates(
        json.dumps({"candidates": [{"query": "Increase the printer density setting.", "kind": "diagnostic", "outcomes": ["fixed", "not fixed", "cannot do"]}]}),
        "scenario", [], 1,
    )[0]
    recalibrated = env._parse_candidates(
        json.dumps({"candidates": [{"query": "Perform a full recalibration using test weights.", "kind": "diagnostic", "outcomes": ["fixed", "not fixed", "cannot do"]}]}),
        "scenario", [], 1,
    )[0]
    assert diagnostic.kind == "diagnostic"
    assert solution.kind == "solution"
    assert increased.kind == "solution"
    assert recalibrated.kind == "solution"


def test_candidate_parser_always_adds_uncertainty_outcome() -> None:
    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
    )
    env = PaprikaCustomerServiceEnvironment(config, RoutingCustomer())
    action = env._parse_candidates(
        json.dumps(
            {
                "candidates": [
                    {
                        "query": "Check the indicator?",
                        "kind": "diagnostic",
                        "outcomes": ["green", "red", "off"],
                    }
                ]
            }
        ),
        "scenario",
        [],
        1,
    )[0]
    assert len(action.outcomes) == 4
    assert "Not attempted / cannot determine" in action.outcomes


def test_explicit_observation_cannot_map_to_uncertainty_outcome() -> None:
    class IncorrectUncertaintyMapper(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if '"outcome"' in text and '"clean"' in text:
                return [json.dumps({"outcome": "Not checked / cannot determine", "clean": True})]
            return super().chat_complete(messages, temperature, num_responses)

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
    )
    env = PaprikaCustomerServiceEnvironment(config, RoutingCustomer())
    env.questioner = IncorrectUncertaintyMapper()
    action = PaprikaAction(
        "Log out and back in.",
        ("Data is visible", "Data remains missing", "Not checked / cannot determine"),
        "scenario",
        kind="diagnostic",
    )
    env.answerer.chat_complete = lambda *args, **kwargs: [
        "I tried logging out and back in, but the data is still missing."
    ]
    observation = env.observe(action, load_paprika_tasks(FIXTURE)[0], np.random.default_rng(0))
    assert observation.mapped_outcome is None
    assert observation.mapped_cleanly is False


def test_explicit_observation_is_repaired_to_supported_non_uncertainty_outcome() -> None:
    class RepairingMapper(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if "This is a repair pass" in text:
                return [json.dumps({"outcome": "The sink drains normally", "clean": True})]
            if '"outcome"' in text and '"clean"' in text:
                return [json.dumps({"outcome": None, "clean": False})]
            return super().chat_complete(messages, temperature, num_responses)

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
    )
    env = PaprikaCustomerServiceEnvironment(config, RoutingCustomer())
    env.questioner = RepairingMapper()
    action = PaprikaAction(
        "Run the disposal and check whether the sink drains.",
        ("The sink drains normally", "The sink drains slowly", "Not checked / cannot determine"),
        "scenario",
    )
    env.answerer.chat_complete = lambda *args, **kwargs: [
        "The sink drains fine, but the dishwasher still has water."
    ]
    observation = env.observe(action, load_paprika_tasks(FIXTURE)[0], np.random.default_rng(0))
    assert observation.mapped_outcome == "The sink drains normally"
    assert observation.mapped_cleanly is True


def test_categorical_eig_matches_deterministic_binary_information() -> None:
    value = categorical_eig([0.5, 0.5], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert value == pytest.approx(np.log(2.0))


def test_distribution_treats_explicit_null_and_missing_outcome_as_zero() -> None:
    outcomes = ("yes", "no", "unknown")
    parsed = parse_distribution(
        '{"probabilities":{"yes":0.75,"no":null,"unknown":0.25}}', outcomes
    )
    assert parsed == pytest.approx((0.75, 0.0, 0.25))
    missing = parse_distribution(
        '{"probabilities":{"yes":0.75,"unknown":0.25}}', outcomes
    )
    assert missing == pytest.approx((0.75, 0.0, 0.25))
    with pytest.raises(ValueError, match="positive total mass"):
        parse_distribution('{"probabilities":{}}', outcomes)


def test_distribution_assigns_all_zero_residual_to_guaranteed_uncertainty() -> None:
    outcomes = ("works", "still broken", "Not attempted / cannot determine")
    assert parse_distribution('{"probabilities":{}}', outcomes) == (0.0, 0.0, 1.0)
    assert parse_distribution(
        '{"probabilities":{"works":0,"still broken":null}}', outcomes
    ) == (0.0, 0.0, 1.0)


def test_nested_paprika_config_aliases(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "task: paprika_customer_service\n"
        "environment:\n"
        f"  data_path: {FIXTURE}\n"
        "  split: eval\n"
        "  verify_official_hash: false\n"
        "  num_trials: 5\n"
        "  num_rounds: 2\n"
        "  trial_batch_size: 1\n"
        "  num_hypotheses: 3\n"
        "  num_candidates: 2\n"
        "  shared_call_cache_enabled: true\n"
    )
    config = load_config(str(path))
    assert config.paprika_data_path == str(FIXTURE)
    assert config.paprika_num_trials == 5
    assert config.paprika_num_candidates == 2
    assert config.paprika_shared_call_cache_enabled is True


def test_five_task_runner_smoke_logs_full_answer_coverage(tmp_path: Path) -> None:
    questioner = RoutingQuestioner()
    customer = RoutingCustomer()
    config = Config(
        task="paprika_customer_service",
        method_names=["EIG"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=5,
        paprika_num_rounds=2,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
        paprika_trial_batch_size=1,
        generation_temperature_simple=0.0,
    )
    run_result, summary = run_from_config(config, questioner, customer, output_dir=tmp_path)
    assert len(run_result.trials) == 5
    assert all(len(trial.rounds) == 2 for trial in run_result.trials)
    assert summary.metrics["answer_set_coverage"] == [1.0, 1.0]
    assert summary.metrics["resolved"] == [0.0, 0.0]
    assert customer.calls == 10
    # One batch per candidate/action matrix. Round two also scores newly refined
    # hypotheses against the previous action, so this is 7 batches per task rather
    # than one request per hypothesis.
    assert questioner.batch_calls == 35
    assert all(len(trial.final_belief_state.hypotheses) == 15 for trial in run_result.trials)
    assert all(
        any(hypothesis.startswith("refined cause") for hypothesis in trial.final_belief_state.hypotheses)
        for trial in run_result.trials
    )
    smoke = json.loads((tmp_path / "paprika_smoke.json").read_text())
    assert len(smoke) == 5
    assert all(turn["mapped_cleanly"] for trial in smoke for turn in trial["turns"])


def test_goal_reached_stops_without_categorical_mapping(tmp_path: Path) -> None:
    config = Config(
        task="paprika_customer_service",
        method_names=["EIG"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=3,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )
    run_result, summary = run_from_config(
        config, RoutingQuestioner(), SolvingCustomer(), output_dir=tmp_path
    )
    assert len(run_result.trials[0].rounds) == 1
    assert summary.metrics["resolved"] == [1.0]


def test_embedded_goal_reached_phrase_is_terminal(tmp_path: Path) -> None:
    config = Config(
        task="paprika_customer_service",
        method_names=["naive"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=3,
    )
    run_result, summary = run_from_config(
        config, RoutingQuestioner(), EmbeddedGoalCustomer(), output_dir=tmp_path
    )
    assert len(run_result.trials[0].rounds) == 1
    assert summary.metrics["resolved"] == [1.0]


def test_naive_is_history_only_and_still_uses_native_early_stop(tmp_path: Path) -> None:
    questioner = RoutingQuestioner()
    config = Config(
        task="paprika_customer_service",
        method_names=["naive"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=3,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )
    run_result, summary = run_from_config(
        config, questioner, SolvingCustomer(), output_dir=tmp_path
    )
    assert len(run_result.trials[0].rounds) == 1
    assert run_result.trials[0].final_belief_state.hypotheses == ()
    assert summary.metrics["resolved"] == [1.0]
    assert not any('"hypotheses"' in text for text in questioner.prompt_texts)
    assert questioner.batch_calls == 0


def test_naive_policy_actions_batch_across_trials_without_private_solutions(tmp_path: Path) -> None:
    questioner = RoutingQuestioner()
    customer = RoutingCustomer()
    config = Config(
        task="paprika_customer_service",
        method_names=["naive"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=5,
        paprika_num_rounds=1,
        paprika_trial_batch_size=5,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )

    run_result, _summary = run_from_config(
        config, questioner, customer, output_dir=tmp_path
    )

    assert len(run_result.trials) == 5
    assert questioner.batch_calls == 1
    assert customer.calls == 5
    private_solutions = [task.solution for task in load_paprika_tasks(FIXTURE)]
    policy_prompts = [prompt for prompt in questioner.prompt_texts if '"candidates"' in prompt]
    assert not any(
        solution in prompt
        for solution in private_solutions
        for prompt in policy_prompts
    )


def test_thinking_naive_routes_mapping_to_nonthinking_evaluator() -> None:
    class ThinkingQuestioner(RoutingQuestioner):
        thinking = True

    class NonthinkingEvaluator(RoutingCustomer):
        thinking = False

        def __init__(self) -> None:
            super().__init__()
            self.prompt_texts = []

        def chat_complete(self, messages, temperature, num_responses=1):
            del temperature, num_responses
            text = "\n".join(message["content"] for message in messages)
            self.prompt_texts.append(text)
            if '"consistent"' in text:
                return [json.dumps({"consistent": True})]
            if '"outcome"' in text and '"clean"' in text:
                return [json.dumps({"outcome": "positive", "clean": True})]
            if "Reply with <VALID>" in text:
                return ["<NOTVALID>"]
            self.calls += 1
            return ["The diagnostic result is positive."]

    questioner = ThinkingQuestioner()
    evaluator = NonthinkingEvaluator()
    config = Config(
        task="paprika_customer_service",
        method_names=["naive"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
    )

    run_from_config(config, questioner, evaluator)

    assert any('"candidates"' in prompt for prompt in questioner.prompt_texts)
    assert not any('"outcome"' in prompt and '"clean"' in prompt for prompt in questioner.prompt_texts)
    assert any('"outcome"' in prompt and '"clean"' in prompt for prompt in evaluator.prompt_texts)


def test_thinking_scaffold_routes_only_generation_to_thinking_model() -> None:
    class ThinkingQuestioner(RoutingQuestioner):
        thinking = True

    class NonthinkingEvaluator(RoutingQuestioner):
        thinking = False

        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if "role-play as a customer" in text:
                self.calls += 1
                self.prompt_texts.append(text)
                return ["The diagnostic result is positive."]
            return super().chat_complete(messages, temperature, num_responses)

    questioner = ThinkingQuestioner()
    evaluator = NonthinkingEvaluator()
    config = Config(
        task="paprika_customer_service",
        method_names=["EIG"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
        paprika_num_refresh_hypotheses=2,
    )

    run_from_config(config, questioner, evaluator)

    thinking_prompts = "\n".join(questioner.prompt_texts)
    evaluator_prompts = "\n".join(evaluator.prompt_texts)
    assert '"hypotheses"' in thinking_prompts
    assert '"candidates"' in thinking_prompts
    assert '"refined_hypotheses"' in thinking_prompts
    assert '"probabilities"' not in thinking_prompts
    assert '"keep_indices"' not in thinking_prompts
    assert '"outcome"' not in thinking_prompts
    assert '"probabilities"' in evaluator_prompts
    assert '"keep_indices"' in evaluator_prompts
    assert '"outcome"' in evaluator_prompts


def test_native_success_judge_runs_for_diagnostic_query() -> None:
    questioner = AlwaysValidJudgeQuestioner()
    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )
    run_result, summary = run_from_config(
        config, questioner, RoutingCustomer(), method_name="EIG"
    )
    assert run_result.trials[0].rounds[0].chosen.action.kind == "diagnostic"
    assert summary.metrics["resolved"] == [1.0]
    success_prompts = [
        text for text in questioner.prompt_texts if "Reply with <VALID>" in text
    ]
    assert len(success_prompts) == 1
    assert "The complete conversation begins here" in success_prompts[0]
    assert "Agent: Check diagnostic 0?" in success_prompts[0]
    assert "Customer: The diagnostic result is positive." in success_prompts[0]


def test_failed_solution_attempt_is_rejected_by_success_judge() -> None:
    questioner = RoutingQuestioner()
    customer = RoutingCustomer()
    customer.chat_complete = lambda *args, **kwargs: [
        "I can't connect to any network right now; it still isn't working."
    ]
    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
    )
    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    env = PaprikaCustomerServiceEnvironment(config, customer)
    env.questioner = questioner
    action = PaprikaAction(
        "Enable mobile data or connect to Wi-Fi.",
        ("working", "not working", "Not attempted / cannot determine"),
        "scenario",
        kind="solution",
    )
    observation = env.observe(action, load_paprika_tasks(FIXTURE)[0], np.random.default_rng(0))
    assert observation.goal_reached is False
    assert any("Reply with <VALID>" in text for text in questioner.prompt_texts)


def test_trailer_connector_explicit_fix_is_terminal_on_diagnostic_turn() -> None:
    class EndpointEvaluator(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if '"consistent"' in text:
                return [json.dumps({"consistent": True})]
            if "Reply with <VALID>" in text:
                assert "Agent: Earlier check" in text
                assert "Customer: The connector looked loose." in text
                assert "Agent: Check whether the trailer plug is fully seated." in text
                assert "Customer: It was loose. I tightened it and the lights are working." in text
                return ["<VALID>"]
            return super().chat_complete(messages, temperature, num_responses)

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    customer = RoutingCustomer()
    customer.chat_complete = lambda *args, **kwargs: [
        "It was loose. I tightened it and the lights are working."
    ]
    env = PaprikaCustomerServiceEnvironment(
        Config(task="paprika_customer_service", paprika_data_path=str(FIXTURE), paprika_verify_official_hash=False),
        customer,
    )
    env.questioner = EndpointEvaluator()
    action = PaprikaAction(
        "Check whether the trailer plug is fully seated.",
        ("secure", "loose", "Not attempted / cannot determine"),
        "Trailer lights are not working.",
        transcript=(("Earlier check", "The connector looked loose."),),
        kind="diagnostic",
    )
    observation = env.observe(action, load_paprika_tasks(FIXTURE)[0], np.random.default_rng(0))
    assert observation.goal_reached is True


@pytest.mark.parametrize(
    "first_reply",
    [
        "I cleaned the screen, but it is still not working.",
        "I'll try cleaning it right now.",
    ],
)
def test_kiosk_correct_remedy_contradictions_are_repaired(first_reply: str) -> None:
    class FaithfulnessEvaluator(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if '"consistent"' in text:
                inconsistent = first_reply in text
                return [json.dumps({"consistent": not inconsistent})]
            if "Reply with <VALID>" in text:
                return ["<VALID>" if "Customer: Goal reached" in text else "<NOTVALID>"]
            return super().chat_complete(messages, temperature, num_responses)

    class RepairingCustomer(RoutingCustomer):
        def chat_complete(self, messages, temperature, num_responses=1):
            del temperature, num_responses
            self.calls += 1
            text = "\n".join(message["content"] for message in messages)
            return ["Goal reached" if "contradicted the private ground truth" in text else first_reply]

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_structured_max_retries=2,
    )
    customer = RepairingCustomer()
    env = PaprikaCustomerServiceEnvironment(config, customer)
    env.questioner = FaithfulnessEvaluator()
    action = PaprikaAction(
        "Clean the kiosk display with a microfiber cloth.",
        ("touch works", "still broken", "Not attempted / cannot determine"),
        "The kiosk screen does not respond to touch.",
        kind="solution",
    )
    task = type(load_paprika_tasks(FIXTURE)[0])(
        task_id="customer_service:eval:kiosk",
        scenario="The kiosk screen does not respond to touch.",
        solution="The dirty screen must be cleaned with a microfiber cloth.",
        split="eval",
    )
    observation = env.observe(action, task, np.random.default_rng(0))
    metrics = env._simulator_faithfulness_metrics()
    assert observation.reply == "Goal reached"
    assert observation.goal_reached is True
    assert customer.calls == 2
    assert metrics["simulator_faithfulness_raw_contradictions"] == 1.0
    assert metrics["simulator_faithfulness_repairs"] == 1.0
    assert metrics["simulator_faithfulness_final_inconsistency_rate"] == 0.0


def test_unrepaired_simulator_contradiction_fails_closed() -> None:
    class RejectingEvaluator(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if '"consistent"' in text:
                return [json.dumps({"consistent": False})]
            return super().chat_complete(messages, temperature, num_responses)

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_structured_max_retries=2,
    )
    customer = RoutingCustomer()
    env = PaprikaCustomerServiceEnvironment(config, customer)
    env.questioner = RejectingEvaluator()
    action = PaprikaAction(
        "Clean the kiosk display with a microfiber cloth.",
        ("touch works", "still broken", "Not attempted / cannot determine"),
        "The kiosk screen does not respond to touch.",
        kind="solution",
    )
    task = type(load_paprika_tasks(FIXTURE)[0])(
        task_id="customer_service:eval:kiosk",
        scenario="The kiosk screen does not respond to touch.",
        solution="The dirty screen must be cleaned with a microfiber cloth.",
        split="eval",
    )
    with pytest.raises(ValueError, match="contradicted the private solution"):
        env.observe(action, task, np.random.default_rng(0))
    metrics = env._simulator_faithfulness_metrics()
    assert customer.calls == 3
    assert metrics["simulator_faithfulness_repairs"] == 2.0
    assert metrics["simulator_faithfulness_failures"] == 1.0
    assert metrics["simulator_faithfulness_final_inconsistency_rate"] == 1.0


def test_incorrect_drain_hose_terminal_claim_is_rejected_and_repaired() -> None:
    class TerminalEvaluator(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            del temperature, num_responses
            text = "\n".join(message["content"] for message in messages)
            if '"terminal_consistent"' in text:
                assert "straightening a drain hose is false" in text
                return [json.dumps({"terminal_consistent": False})]
            if '"consistent"' in text:
                return [json.dumps({"consistent": True})]
            if "Reply with <VALID>" in text:
                return ["<NOTVALID>"]
            if '"outcome"' in text and '"clean"' in text:
                return [
                    json.dumps(
                        {
                            "outcome": "The water remains after straightening the hose.",
                            "clean": True,
                        }
                    )
                ]
            return super().chat_complete(messages, 0.0, 1)

    class RepairingCustomer(RoutingCustomer):
        def chat_complete(self, messages, temperature, num_responses=1):
            del temperature, num_responses
            self.calls += 1
            text = "\n".join(message["content"] for message in messages)
            if "contradicted the private ground truth" in text:
                return ["I straightened the hose, but the water is still there."]
            return ["Goal reached"]

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_structured_max_retries=2,
    )
    customer = RepairingCustomer()
    env = PaprikaCustomerServiceEnvironment(config, customer)
    env.questioner = TerminalEvaluator()
    action = PaprikaAction(
        "Straighten the drain hose behind the appliance.",
        (
            "The dishwasher drains correctly now.",
            "The water remains after straightening the hose.",
            "Not attempted / cannot determine",
        ),
        "The dishwasher is leaving water at the bottom.",
        kind="solution",
    )
    task = type(load_paprika_tasks(FIXTURE)[0])(
        task_id="customer_service:eval:dishwasher",
        scenario="The dishwasher is leaving water at the bottom.",
        solution="The drain hose is clogged, and clearing it will restore drainage.",
        split="eval",
    )
    observation = env.observe(action, task, np.random.default_rng(0))
    metrics = env._simulator_faithfulness_metrics()
    assert observation.goal_reached is False
    assert observation.reply == "I straightened the hose, but the water is still there."
    assert customer.calls == 2
    assert metrics["simulator_faithfulness_raw_contradictions"] == 1.0
    assert metrics["simulator_faithfulness_repairs"] == 1.0
    assert metrics["simulator_terminal_claims"] == 1.0
    assert metrics["simulator_terminal_rejections"] == 1.0
    assert metrics["simulator_faithfulness_final_inconsistency_rate"] == 0.0


def test_prospective_attempt_maps_cleanly_to_uncertainty() -> None:
    class UncertaintyMapper(RoutingQuestioner):
        def chat_complete(self, messages, temperature, num_responses=1):
            text = "\n".join(message["content"] for message in messages)
            if '"outcome"' in text and '"clean"' in text:
                return [json.dumps({"outcome": "Not attempted / cannot determine", "clean": True})]
            return super().chat_complete(messages, temperature, num_responses)

    from environments.paprika_customer_service.env import PaprikaCustomerServiceEnvironment

    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
    )
    customer = RoutingCustomer()
    customer.chat_complete = lambda *args, **kwargs: ["I'll try that now."]
    env = PaprikaCustomerServiceEnvironment(config, customer)
    env.questioner = UncertaintyMapper()
    action = PaprikaAction(
        "Check the setting?",
        ("enabled", "disabled", "Not attempted / cannot determine"),
        "scenario",
    )
    observation = env.observe(action, load_paprika_tasks(FIXTURE)[0], np.random.default_rng(0))
    assert observation.mapped_outcome == "Not attempted / cannot determine"
    assert observation.mapped_cleanly is True


def test_full_two_step_expands_each_root_outcome(tmp_path: Path) -> None:
    questioner = RoutingQuestioner()
    config = Config(
        task="paprika_customer_service",
        method_names=["Full2StepEIG"],
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )
    run_result, _summary = run_from_config(
        config, questioner, RoutingCustomer(), output_dir=tmp_path
    )
    chosen = run_result.trials[0].rounds[0].chosen
    assert chosen.extras["planning_depth"] == 2
    assert chosen.extras["expanded_branch_counts"] == [3, 3]
    assert len(chosen.extras["candidate_scores"]) == 2
    # Root likelihoods, all branch proposals, all follow-up likelihoods, and
    # post-observation refined-support likelihoods are each one batched call.
    assert questioner.batch_calls == 4


def test_paired_methods_reuse_identical_root_candidates_and_customer_reply() -> None:
    questioner = RoutingQuestioner()
    customer = RoutingCustomer()
    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
        paprika_shared_call_cache_enabled=True,
    )
    eig_run, _ = run_from_config(config, questioner, customer, method_name="EIG")
    two_run, two_summary = run_from_config(
        config, questioner, customer, method_name="Full2StepEIG"
    )
    eig_round = eig_run.trials[0].rounds[0]
    two_round = two_run.trials[0].rounds[0]
    assert eig_round.candidates == two_round.candidates
    assert eig_round.observation == two_round.observation
    assert customer.calls == 1
    assert two_summary.metrics["shared_call_cache_hits"][0] > 0


@pytest.mark.parametrize(
    "questioner_type", [FlakyCandidateQuestioner, FlakyLikelihoodQuestioner]
)
def test_structured_output_repair_is_bounded_and_logged(questioner_type) -> None:
    config = Config(
        task="paprika_customer_service",
        paprika_data_path=str(FIXTURE),
        paprika_verify_official_hash=False,
        paprika_num_trials=1,
        paprika_num_rounds=1,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
        paprika_structured_max_retries=2,
    )
    _run, summary = run_from_config(
        config, questioner_type(), RoutingCustomer(), method_name="EIG"
    )
    assert summary.metrics["structured_parse_retries"] == [1.0]
    assert summary.metrics["structured_parse_failures"] == [0.0]
