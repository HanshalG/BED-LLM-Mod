from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from core.experiment import run_from_config
from environments.paprika_customer_service import PaprikaAction, load_paprika_tasks
from helpers import Config, load_config
from methods.categorical_eig import categorical_eig


FIXTURE = Path(__file__).parent / "fixtures" / "paprika_customer_service_tiny.json"


class RoutingQuestioner:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_calls = 0

    def chat_complete(self, messages, temperature, num_responses=1):
        del temperature, num_responses
        self.calls += 1
        text = messages[-1]["content"]
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
            return [json.dumps({"candidates": [{"query": f"Check diagnostic {index}?", "outcomes": ["positive", "negative", "unknown"]} for index in range(count)]})]
        if '"probabilities"' in text:
            return [json.dumps({"probabilities": {"positive": 0.7, "negative": 0.2, "unknown": 0.1}})]
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


def test_categorical_eig_matches_deterministic_binary_information() -> None:
    value = categorical_eig([0.5, 0.5], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert value == pytest.approx(np.log(2.0))


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
