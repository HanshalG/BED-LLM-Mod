from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from core import BeliefState
from core.experiment import run_from_config
from environments.mediq import (
    MediQAction,
    MediQObservation,
    load_mediq_tasks,
    load_mediq_tasks_with_report,
)
from environments.mediq.env import (
    UNAVAILABLE_OUTCOME,
    MediQEnvironment,
    _ensure_unavailable,
)
from helpers import Config, load_config


FIXTURE = Path(__file__).parent / "fixtures" / "mediq_tiny.jsonl"


class RoutingMediQModel:
    def __init__(self) -> None:
        self.batches: list[list[list[dict[str, str]]]] = []
        self.thinking = False

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        self.batches.append(batch_messages)
        return [self._route(messages) for messages in batch_messages]

    @staticmethod
    def _route(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "calibrated clinical multiple-choice judge" in system:
            return json.dumps(
                {"probabilities": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}}
            )
        if "generate atomic patient questions" in system:
            match = re.search(r"Return exactly (\d+) candidate", user)
            count = int(match.group(1)) if match else 1
            branch_offset = 10 if "Conversation so far:\nNone" not in user else 0
            candidates = []
            for index in range(count):
                candidates.append(
                    {
                        "query": (
                            "What relevant diagnostic finding number "
                            f"{branch_offset + index + 1} is present?"
                        ),
                        "outcomes": [
                            "Finding present",
                            "Finding absent",
                            "Information unavailable / not in record",
                        ],
                    }
                )
            return json.dumps({"candidates": candidates})
        if "strict MediQ candidate-space auditor" in system:
            return json.dumps({"valid": True, "reason": "valid atomic partition"})
        if "calibrated clinical generative model" in system:
            outcomes = json.loads(
                re.search(r"Response categories: (\[[^\n]+\])", user).group(1)
            )
            hypothesis = re.search(r"Assume the correct answer is ([A-Z]):", user).group(1)
            if hypothesis == "A":
                values = [0.8, 0.1, 0.1]
            else:
                values = [0.1, 0.8, 0.1]
            return json.dumps(
                {"probabilities": dict(zip(outcomes, values, strict=True))}
            )
        if "official-style MediQ Fact-Select patient" in system:
            return json.dumps({"fact_indices": [1], "cannot_answer": False})
        if "strict MediQ explicit-entailment auditor" in system:
            return json.dumps(
                {"relevant": True, "reason": "the selected fact explicitly answers it"}
            )
        if "Map an already-validated" in system:
            outcomes = json.loads(
                re.search(r"Response categories: (\[[^\n]+\])", user).group(1)
            )
            return json.dumps({"clean": True, "outcome": outcomes[0]})
        raise AssertionError(f"Unexpected MediQ prompt: {system}")


class CandidateRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            if "failed structural validation" in user:
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "query": "What is the body temperature?",
                                "outcomes": [
                                    "Below 38 C",
                                    "At least 38 C",
                                    UNAVAILABLE_OUTCOME,
                                ],
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": "What is the blood glucose value?",
                            "outcomes": [
                                "Below 50 mg/dL",
                                "Above 500 mg/dL",
                                UNAVAILABLE_OUTCOME,
                            ],
                        }
                    ]
                }
            )
        if "strict MediQ candidate-space auditor" in system:
            if "blood glucose" in user:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": "numeric outcomes leave a gap from 50 to 500 mg/dL",
                    }
                )
            return json.dumps({"valid": True, "reason": "complete numeric partition"})
        return super()._route(messages)


class CompoundQuestionRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            query = (
                "What is the blood smear result?"
                if "contains 'and' or 'or'" in user
                else "Do you have fever or productive cough?"
            )
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": query,
                            "outcomes": ["Present", "Absent", UNAVAILABLE_OUTCOME],
                        }
                    ]
                }
            )
        return super()._route(messages)


class RejectingRelevanceModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        if "strict MediQ explicit-entailment auditor" in messages[0]["content"]:
            return json.dumps(
                {
                    "relevant": False,
                    "reason": "age and sex do not answer the requested diagnostic finding",
                }
            )
        return super()._route(messages)


class IrrelevantThenUnavailablePatient(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        if "official-style MediQ Fact-Select patient" in messages[0]["content"]:
            repaired = any(
                "selected facts did not directly answer" in message["content"].casefold()
                for message in messages
            )
            if repaired:
                return json.dumps({"fact_indices": [], "cannot_answer": True})
            return json.dumps({"fact_indices": [0], "cannot_answer": False})
        return super()._route(messages)


def _config(**overrides: object) -> Config:
    values = {
        "task": "mediq",
        "method_names": ["EIG"],
        "mediq_data_path": str(FIXTURE),
        "mediq_verify_official_hash": False,
        "mediq_num_trials": 2,
        "mediq_num_rounds": 1,
        "mediq_trial_batch_size": 2,
        "mediq_num_candidates": 2,
        "generation_temperature_diverse": 0.0,
        "answer_temperature": 0.0,
    }
    values.update(overrides)
    return Config(**values)


def test_load_mediq_tasks_preserves_target_and_hides_context() -> None:
    tasks = load_mediq_tasks(FIXTURE, dataset="imedqa")
    assert len(tasks) == 2
    assert tasks[0].task_id == "mediq:imedqa:0"
    assert tasks[0].initial_info == "A 28-year-old woman presents with fever."
    assert tasks[0].context[2] == "A blood smear shows ring forms within erythrocytes."
    assert tasks[0].facts[2] == "A blood smear shows ring forms within erythrocytes."
    assert tasks[0].answer_idx == "A"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mediq_tasks(FIXTURE, dataset="imedqa", verify_official_hash=True)


def test_load_mediq_tasks_explicitly_reports_unusable_rows(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    unusable = dict(rows[0], id=224, context=[], facts=[])
    path = tmp_path / "with-unusable.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [*rows, unusable]) + "\n")
    tasks, excluded, raw_count = load_mediq_tasks_with_report(path)
    assert len(tasks) == 2
    assert excluded == ("224",)
    assert raw_count == 3
    with pytest.raises(ValueError, match="requires non-empty context"):
        load_mediq_tasks(path, skip_unusable_tasks=False)


def test_mediq_nested_config_aliases_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "mediq.yaml"
    path.write_text(
        "\n".join(
            [
                "task: mediq",
                "method_names: [EIG]",
                "environment:",
                f"  data_path: {FIXTURE}",
                "  dataset: imedqa",
                "  verify_official_hash: false",
                "  skip_unusable_tasks: false",
                "  num_trials: 2",
                "  num_rounds: 3",
                "  trial_batch_size: 2",
                "  task_offset: 0",
                "  seed: 1304",
                "  num_candidates: 4",
                "  max_patient_facts: 2",
                "  probability_floor: 0.02",
                "  shared_call_cache_enabled: true",
                "  structured_max_retries: 1",
            ]
        )
    )
    config = load_config(str(path))
    assert config.mediq_data_path == str(FIXTURE)
    assert config.mediq_skip_unusable_tasks is False
    assert config.mediq_num_trials == 2
    assert config.mediq_num_rounds == 3
    assert config.mediq_trial_batch_size == 2
    assert config.mediq_seed == 1304
    assert config.mediq_num_candidates == 4
    assert config.mediq_probability_floor == 0.02
    assert config.mediq_config.num_candidates == 4
    with pytest.raises(ValueError, match="mediq_probability_floor"):
        Config(task="mediq", mediq_probability_floor=0.25)
    with pytest.raises(ValueError, match="mediq_num_candidates"):
        Config(task="mediq", mediq_num_candidates=0)
    with pytest.raises(ValueError, match="mediq_skip_unusable_tasks"):
        Config(task="mediq", mediq_skip_unusable_tasks="yes")


def test_mediq_update_applies_latest_likelihood_once() -> None:
    model = RoutingMediQModel()
    config = _config(mediq_num_trials=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.tasks[0]
    action = MediQAction(
        query="What finding is present?",
        outcomes=(
            "Finding present",
            "Finding absent",
            "Information unavailable / not in record",
        ),
        task=task,
        prior_probabilities=(0.25, 0.25, 0.25, 0.25),
    )
    observation = MediQObservation(
        reply=task.facts[1],
        mapped_outcome="Finding present",
        mapped_cleanly=True,
        selected_fact_indices=(1,),
        grounded=True,
        relevant=True,
        cannot_answer=False,
    )
    prior = BeliefState.uniform(task.option_labels)
    updated = env.update_belief_state(prior, [(action, observation)], model, config)
    assert updated.probabilities[0] > updated.probabilities[1]
    assert np.isclose(sum(updated.probabilities), 1.0)
    updated_twice = env.update_belief_state(
        updated, [(action, observation)], model, config
    )
    assert updated_twice.probabilities[0] > updated.probabilities[0]


def test_mediq_outcomes_have_one_canonical_unavailable_category() -> None:
    outcomes = _ensure_unavailable(
        [
            "Low",
            "Normal",
            "High",
            "Not recorded",
            "Unknown",
            "Information unavailable / not in record",
        ]
    )
    assert outcomes == ("Low", "Normal", "High", UNAVAILABLE_OUTCOME)
    assert sum(outcome == UNAVAILABLE_OUTCOME for outcome in outcomes) == 1


def test_mediq_compound_candidate_is_rejected() -> None:
    model = RoutingMediQModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    belief = BeliefState.uniform(task.option_labels)
    response = json.dumps(
        {
            "candidates": [
                {
                    "query": "Do you have fever or productive cough?",
                    "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="Expected 1 valid MediQ candidates") as exc_info:
        env._parse_candidates(response, task, belief, [], 1)
    assert "contains 'and' or 'or'; ask one variable only" in str(exc_info.value)


def test_mediq_semantic_candidate_validation_regenerates_invalid_set() -> None:
    model = CandidateRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == ["What is the body temperature?"]
    assert env._candidate_metrics() == {
        "candidate_validation_checks": 2.0,
        "candidate_validation_retries": 1.0,
        "candidate_validation_failures": 0.0,
    }


def test_mediq_compound_candidate_repair_receives_specific_feedback() -> None:
    model = CompoundQuestionRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == ["What is the blood smear result?"]
    assert env._structured_parse_retries == 1


def test_mediq_relevance_is_category_blind_and_repairs_fact_selection() -> None:
    questioner = RejectingRelevanceModel()
    answerer = IrrelevantThenUnavailablePatient()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, answerer).configure_for_run(config)
    env.set_questioner(questioner)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    action = MediQAction(
        query="What diagnostic finding confirms the infection?",
        outcomes=("Ring forms present", "Ring forms absent", UNAVAILABLE_OUTCOME),
        task=task,
    )
    observation = env.observe(action, task, np.random.default_rng(0))
    assert observation.cannot_answer is True
    assert observation.mapped_outcome == UNAVAILABLE_OUTCOME
    assert observation.selected_fact_indices == ()
    relevance_prompts = [
        messages
        for batch in questioner.batches
        for messages in batch
        if "strict MediQ explicit-entailment auditor" in messages[0]["content"]
    ]
    assert len(relevance_prompts) == 1
    assert "Response categories" not in relevance_prompts[0][-1]["content"]
    patient_metrics = env._patient_metrics()
    assert patient_metrics["patient_raw_irrelevant_selections"] == 1.0
    assert patient_metrics["patient_relevance_repairs"] == 1.0
    assert patient_metrics["patient_relevance_failures"] == 0.0


def test_mediq_relevant_fact_cannot_map_to_unavailable() -> None:
    model = RoutingMediQModel()
    config = _config(mediq_num_trials=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    task = env.tasks[0]
    action = MediQAction(
        query="What is the blood smear result?",
        outcomes=("Ring forms present", "Ring forms absent", UNAVAILABLE_OUTCOME),
        task=task,
    )
    response = json.dumps({"clean": True, "outcome": UNAVAILABLE_OUTCOME})
    with pytest.raises(ValueError, match="cannot map to unavailable"):
        env._parse_patient_mapping(response, action)


def test_mediq_eig_integration_uses_grounded_patient_and_writes_artifact(
    tmp_path: Path,
) -> None:
    questioner = RoutingMediQModel()
    answerer = RoutingMediQModel()
    _run_result, summary = run_from_config(
        _config(),
        questioner,
        answerer,
        output_dir=tmp_path,
    )
    for metric in (
        "accuracy",
        "correct_option_mass",
        "belief_entropy",
        "answer_set_coverage",
        "patient_grounding_rate",
        "patient_relevance_rate",
        "realized_entropy_drop",
        "selected_eig",
    ):
        assert len(summary.metrics[metric]) == 1
        assert np.isfinite(summary.metrics[metric][0])
    assert summary.metrics["patient_grounding_rate"] == [1.0]
    assert summary.metrics["patient_relevance_rate"] == [1.0]
    assert summary.metrics["answer_set_coverage"] == [1.0]
    assert sum(len(batch) for batch in questioner.batches) == 28
    assert sum(len(batch) for batch in answerer.batches) == 2
    records = json.loads((tmp_path / "mediq_interactions.json").read_text())
    assert len(records) == 2
    assert records[0]["turns"][0]["reply"] == records[0]["facts"][1]
    assert records[0]["turns"][0]["grounded"] is True
    manifest = json.loads((tmp_path / "mediq_data_manifest.json").read_text())
    assert manifest["raw_row_count"] == 2
    assert manifest["excluded_source_ids"] == []
    assert manifest["selected_source_ids"] == ["0", "1"]
    candidate = records[0]["turns"][0]["candidate_details"][0]
    assert set(candidate["likelihoods"]) == {"A", "B", "C", "D"}
    assert np.isclose(sum(candidate["predictive_outcome_probabilities"].values()), 1.0)
    assert np.isfinite(
        records[0]["turns"][0]["metrics"][
            "realized_truth_log_probability_gain"
        ]
    )


def test_mediq_naive_is_belief_free_but_decodes_each_round(tmp_path: Path) -> None:
    questioner = RoutingMediQModel()
    answerer = RoutingMediQModel()
    run_result, summary = run_from_config(
        _config(method_names=["naive"]),
        questioner,
        answerer,
        output_dir=tmp_path,
    )
    assert summary.metrics["accuracy"] == [1.0]
    assert all(not trial.final_belief_state.hypotheses for trial in run_result.trials)
    assert summary.metrics["patient_grounding_rate"] == [1.0]


def test_mediq_full_two_step_expands_synthetic_branches_only(tmp_path: Path) -> None:
    questioner = RoutingMediQModel()
    answerer = RoutingMediQModel()
    run_result, summary = run_from_config(
        _config(
            method_names=["Full2StepEIG"],
            mediq_num_trials=1,
            mediq_trial_batch_size=1,
            mediq_num_candidates=1,
        ),
        questioner,
        answerer,
        output_dir=tmp_path,
    )
    chosen = run_result.trials[0].rounds[0].chosen
    assert chosen.extras["planning_depth"] == 2
    assert chosen.extras["expanded_branch_counts"] == [3]
    assert summary.metrics["patient_observations"] == [1.0]
    assert len(answerer.batches) == 1
