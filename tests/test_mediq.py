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
    _project_joint_to_marginals,
)
from helpers import Config, load_config
from scripts.run_icraft_profile_gates import (
    run_calibration as run_profile_calibration_gate,
    run_smoke as run_profile_smoke_gate,
    run_structural as run_profile_structural_gate,
)


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
                            "Is relevant diagnostic finding number "
                            f"{branch_offset + index + 1} present?"
                        ),
                        "outcomes": [
                            "Yes",
                            "No",
                            "Information unavailable / not in record",
                        ],
                    }
                )
            return json.dumps({"candidates": candidates})
        if "strict MediQ candidate-space auditor" in system:
            return json.dumps({"valid": True, "reason": "valid atomic partition"})
        if "strict MediQ candidate-set deduplication auditor" in system:
            return json.dumps(
                {"duplicate_groups": [], "reason": "all candidate queries are distinct"}
            )
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


class FactoredLikelihoodModel(RoutingMediQModel):
    @staticmethod
    def _route(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "estimate record coverage for the MediQ benchmark" in system:
            return json.dumps(
                {
                    "probabilities": {
                        "Answerable from record": 0.75,
                        "Not answerable from record": 0.25,
                    }
                }
            )
        if "counterfactual clinical-record model" in system:
            hypothesis = re.search(
                r"correct answer is ([A-Z]):", user
            ).group(1)
            values = [0.8, 0.2] if hypothesis == "A" else [0.2, 0.8]
            return json.dumps(
                {"probabilities": {"Yes": values[0], "No": values[1]}}
            )
        return RoutingMediQModel._route(messages)


class DataEstimationModel(RoutingMediQModel):
    @staticmethod
    def _route(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "predictive model for the MediQ patient interface" in system:
            return json.dumps(
                {
                    "probabilities": {
                        "Yes": 0.4,
                        "No": 0.4,
                        UNAVAILABLE_OUTCOME: 0.2,
                    }
                }
            )
        if "hypothetical-evidence clinical judge" in system:
            if "Patient response category: Yes" in user:
                values = {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
            else:
                values = {"A": 0.1, "B": 0.5, "C": 0.2, "D": 0.2}
            return json.dumps({"probabilities": values})
        return RoutingMediQModel._route(messages)


class ProfileSupportModel(RoutingMediQModel):
    @staticmethod
    def _route(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "counterfactual patient profiles" in system:
            count = int(re.search(r"exactly (\d+) distinct", user).group(1))
            label = re.search(r"answer label ([A-Z]):", user).group(1)
            return json.dumps({"profiles": [f"Profile {index} for label {label}" for index in range(count)]})
        if "counterfactual patient-profile auditor" in system:
            return json.dumps({"valid": True, "reason": "concrete compatible profile"})
        if "profile-conditioned clinical model" in system:
            label = re.search(r"answer label ([A-Z]):", user).group(1)
            yes = 0.8 if label == "A" else 0.2
            return json.dumps({"probabilities": {"Yes": yes, "No": 1.0 - yes}})
        if "estimate record coverage for the MediQ benchmark" in system:
            return json.dumps({"probabilities": {"Answerable from record": 0.75, "Not answerable from record": 0.25}})
        return RoutingMediQModel._route(messages)


class CandidateRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            if "Rejected queries:" in user:
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "query": "Is the body temperature at least 38 C?",
                                "outcomes": [
                                    "Yes",
                                    "No",
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
                            "query": "Is the fictitious serum protein A elevated?",
                            "outcomes": [
                                "Yes",
                                "No",
                                UNAVAILABLE_OUTCOME,
                            ],
                        }
                    ]
                }
            )
        if "strict MediQ candidate-space auditor" in system:
            if "fictitious serum protein A" in user:
                return json.dumps(
                    {
                        "valid": False,
                        "reason": "the query invents a clinically nonsensical variable",
                    }
                )
            return json.dumps({"valid": True, "reason": "valid binary predicate"})
        return super()._route(messages)


class NaiveRetryDiversityModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            if "Rejected queries:" in user:
                queries = [
                    "Has the patient made physical contact with the clinician?",
                    "Is the patient currently experiencing a fever?",
                    "Has the patient recently traveled to an endemic region?",
                    "Is a new skin rash present?",
                ]
            else:
                queries = ["Has the patient made sexual advances toward the clinician?"]
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": query,
                            "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                        }
                        for query in queries
                    ]
                }
            )
        if "strict MediQ candidate-space auditor" in system:
            valid = not any(
                phrase in user
                for phrase in ("sexual advances", "physical contact")
            )
            return json.dumps(
                {
                    "valid": valid,
                    "reason": (
                        "valid new patient finding"
                        if valid
                        else "semantic paraphrase of a prior unavailable variable"
                    ),
                }
            )
        return super()._route(messages)


class CompoundQuestionRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            query = (
                "Is a ring form present on the blood smear?"
                if "contains 'and' or 'or'" in user
                else "Do you have fever or productive cough?"
            )
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": query,
                            "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                        }
                    ]
                }
            )
        return super()._route(messages)


class PartialCandidateRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            if "Already accepted queries:" in user:
                queries = [
                    "Is the blood smear positive for ring forms?",
                    "Is the body temperature above 38 C?",
                ]
            else:
                queries = [
                    "Is recent endemic travel documented?",
                    "Is the fictitious serum protein A elevated?",
                ]
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": query,
                            "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                        }
                        for query in queries
                    ]
                }
            )
        if "strict MediQ candidate-space auditor" in system:
            valid = "fictitious serum protein A" not in user
            return json.dumps(
                {
                    "valid": valid,
                    "reason": (
                        "valid binary predicate"
                        if valid
                        else "the query invents a clinically nonsensical variable"
                    ),
                }
            )
        return super()._route(messages)


class SynonymCandidateRepairModel(RoutingMediQModel):
    def _route(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "generate atomic patient questions" in system:
            queries = (
                [
                    "Does the patient have a history of urinary tract obstruction?",
                    "Does the patient currently have a fever?",
                ]
                if "Already accepted queries:" in user
                else [
                    "Does the patient have a history of renal calculi?",
                    "Does the patient have a history of nephrolithiasis?",
                ]
            )
            return json.dumps(
                {
                    "candidates": [
                        {
                            "query": query,
                            "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                        }
                        for query in queries
                    ]
                }
            )
        if "strict MediQ candidate-set deduplication auditor" in system:
            if '"query": "Does the patient have a history of nephrolithiasis?' in user:
                return json.dumps(
                    {
                        "duplicate_groups": [[0, 1]],
                        "reason": "renal calculi and nephrolithiasis are synonyms",
                    }
                )
            return json.dumps(
                {"duplicate_groups": [], "reason": "the remaining queries are distinct"}
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


def test_icraft_preserves_answer_text_disagreements_but_uses_indexed_target(
    tmp_path: Path,
) -> None:
    row = json.loads(FIXTURE.read_text().splitlines()[0])
    row["answer"] = "An alternate raw diagnosis string"
    path = tmp_path / "icraft-mismatch.jsonl"
    path.write_text(json.dumps(row) + "\n")
    task = load_mediq_tasks(path, dataset="icraft_md")[0]
    assert task.answer_idx == "A"
    assert task.answer == "An alternate raw diagnosis string"
    assert task.answer_option_text == task.option_text("A")
    assert task.answer_text_matches_option is False
    with pytest.raises(ValueError, match="answer text does not match"):
        load_mediq_tasks(path, dataset="imedqa")


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
                "  likelihood_mode: factored_record",
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
    assert config.mediq_likelihood_mode == "factored_record"
    assert config.mediq_config.num_candidates == 4
    assert config.mediq_config.likelihood_mode == "factored_record"
    assert Config(
        task="mediq", mediq_likelihood_mode="data_estimation"
    ).mediq_config.likelihood_mode == "data_estimation"
    with pytest.raises(ValueError, match="mediq_probability_floor"):
        Config(task="mediq", mediq_probability_floor=0.25)
    with pytest.raises(ValueError, match="mediq_num_candidates"):
        Config(task="mediq", mediq_num_candidates=0)
    with pytest.raises(ValueError, match="mediq_skip_unusable_tasks"):
        Config(task="mediq", mediq_skip_unusable_tasks="yes")
    with pytest.raises(ValueError, match="mediq_likelihood_mode"):
        Config(task="mediq", mediq_likelihood_mode="option_words_are_diagnoses")


def test_mediq_profile_support_uses_fixed_profiles_and_neutral_missingness() -> None:
    model = ProfileSupportModel()
    config = _config(
        mediq_dataset="icraft_md",
        mediq_likelihood_mode="profile_support",
        mediq_profiles_per_option=3,
        mediq_num_trials=1,
    )
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    prior = env.initial_belief_state(model, config)
    assert len(prior.hypotheses) == 12
    diagnosis_prior, labels = env._diagnosis_probabilities(
        prior.hypotheses, prior.probabilities, task
    )
    assert labels == task.option_labels
    assert diagnosis_prior == pytest.approx((0.4, 0.3, 0.2, 0.1))
    action = env.generate_candidate_actions(prior, [], model, config)[0]
    assert action.support_hypotheses == prior.hypotheses
    likelihoods = env.outcome_likelihoods(prior.hypotheses, action)
    assert likelihoods.shape == (12, 3)
    assert np.allclose(likelihoods[:, 2], likelihoods[0, 2])
    unavailable = MediQObservation(
        reply="The patient cannot answer this question from the supplied record.",
        mapped_outcome=UNAVAILABLE_OUTCOME,
        mapped_cleanly=True,
        selected_fact_indices=(),
        grounded=True,
        relevant=True,
        cannot_answer=True,
    )
    updated = env.update_belief_state(prior, [(action, unavailable)], model, config)
    assert updated.probabilities == pytest.approx(prior.probabilities)


def test_mediq_profile_support_config_aliases_and_source_selection(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "\n".join(
            [
                "task: mediq",
                "method_names: [EIG]",
                "environment:",
                f"  data_path: {FIXTURE}",
                "  dataset: icraft_md",
                "  verify_official_hash: false",
                "  likelihood_mode: profile_support",
                "  profiles_per_option: 4",
                "  source_ids: ['1']",
                "  num_trials: 1",
            ]
        )
    )
    config = load_config(str(path))
    assert config.mediq_source_ids == ["1"]
    assert config.mediq_profiles_per_option == 4
    assert config.mediq_config.source_ids == ["1"]
    env = MediQEnvironment(config, ProfileSupportModel()).configure_for_run(config)
    assert [task.source_id for task in env.tasks] == ["1"]
    with pytest.raises(ValueError, match="profiles_per_option"):
        Config(task="mediq", mediq_dataset="icraft_md", mediq_likelihood_mode="profile_support", mediq_profiles_per_option=1)
    with pytest.raises(ValueError, match="require mediq_dataset=icraft_md"):
        Config(task="mediq", mediq_likelihood_mode="profile_support")


def test_mediq_profile_support_integration_reports_diagnosis_level_artifacts(
    tmp_path: Path,
) -> None:
    questioner = ProfileSupportModel()
    _result, summary = run_from_config(
        _config(
            mediq_dataset="icraft_md",
            mediq_likelihood_mode="profile_support",
            mediq_num_trials=1,
            mediq_trial_batch_size=1,
        ),
        questioner,
        RoutingMediQModel(),
        output_dir=tmp_path,
    )
    assert np.isfinite(summary.metrics["correct_option_mass"][0])
    records = json.loads((tmp_path / "mediq_interactions.json").read_text())
    candidate = records[0]["turns"][0]["candidate_details"][0]
    assert set(candidate["prior"]) == {"A", "B", "C", "D"}
    assert set(candidate["likelihoods"]) == {"A", "B", "C", "D"}
    assert len(candidate["profile_support"]) == 12
    assert set(records[0]["final_belief"]) == {"A", "B", "C", "D"}


def test_icraft_profile_gate_runner_exercises_all_preregistered_paths() -> None:
    smoke_config = _config(
        mediq_dataset="icraft_md",
        mediq_likelihood_mode="profile_support",
        mediq_num_trials=1,
        mediq_trial_batch_size=1,
        mediq_num_candidates=2,
        mediq_source_ids=["0"],
    )
    smoke = run_profile_smoke_gate(
        smoke_config, ProfileSupportModel(), RoutingMediQModel()
    )
    assert smoke["passed"] is True
    assert smoke["likelihood_shapes"] == [[12, 3], [12, 3]]

    gate_config = _config(
        mediq_dataset="icraft_md",
        mediq_likelihood_mode="profile_support",
        mediq_num_trials=2,
        mediq_trial_batch_size=2,
        mediq_num_candidates=4,
        mediq_source_ids=["0", "1"],
    )
    calibration = run_profile_calibration_gate(
        gate_config, ProfileSupportModel(), RoutingMediQModel()
    )
    assert len(calibration["rows"]) == 8
    assert calibration["max_unavailable_posterior_move"] == pytest.approx(0.0)
    assert calibration["max_branch_update_error"] == pytest.approx(0.0)

    structural = run_profile_structural_gate(
        gate_config, ProfileSupportModel(), RoutingMediQModel()
    )
    assert len(structural["rows"]) == 2
    assert all(row["two_step_best_value"] >= row["one_step_best_value"] for row in structural["rows"])


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


def test_mediq_factored_likelihood_makes_missingness_label_independent() -> None:
    model = FactoredLikelihoodModel()
    config = _config(
        mediq_num_trials=1,
        mediq_likelihood_mode="factored_record",
    )
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.tasks[0]
    action = MediQAction(
        query="Is a ring form present on the blood smear?",
        outcomes=("Yes", "No", UNAVAILABLE_OUTCOME),
        task=task,
        prior_probabilities=(0.25, 0.25, 0.25, 0.25),
    )
    likelihoods = env.outcome_likelihoods(task.option_labels, action)
    assert likelihoods.shape == (4, 3)
    assert np.allclose(likelihoods[:, 2], likelihoods[0, 2])
    assert np.allclose(likelihoods.sum(axis=1), 1.0)
    assert likelihoods[0, 0] > likelihoods[1, 0]
    assert env._record_availability_cache[action] == pytest.approx((0.75, 0.25))

    prior = BeliefState(task.option_labels, (0.4, 0.3, 0.2, 0.1))
    unavailable = MediQObservation(
        reply="The patient cannot answer this question from the supplied record.",
        mapped_outcome=UNAVAILABLE_OUTCOME,
        mapped_cleanly=True,
        selected_fact_indices=(),
        grounded=True,
        relevant=True,
        cannot_answer=True,
    )
    updated = env.update_belief_state(
        prior, [(action, unavailable)], model, config
    )
    assert updated.probabilities == pytest.approx(prior.probabilities)

    coverage_prompts = [
        messages
        for batch in model.batches
        for messages in batch
        if "estimate record coverage for the MediQ benchmark"
        in messages[0]["content"]
    ]
    binary_prompts = [
        messages
        for batch in model.batches
        for messages in batch
        if "counterfactual clinical-record model" in messages[0]["content"]
    ]
    assert len(coverage_prompts) == 1
    assert len(binary_prompts) == 4
    assert "must not depend on which multiple-choice option is correct" in (
        coverage_prompts[0][-1]["content"]
    )
    assert "findings associated with other options may coexist" in (
        binary_prompts[0][-1]["content"]
    )


def test_joint_projection_matches_both_requested_marginals() -> None:
    matrix = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    rows = np.asarray([0.2, 0.3, 0.5])
    columns = np.asarray([0.45, 0.55])
    projected = _project_joint_to_marginals(matrix, rows, columns)
    assert np.allclose(projected.sum(axis=1), rows, atol=1e-12)
    assert np.allclose(projected.sum(axis=0), columns, atol=1e-12)


def test_mediq_data_estimation_builds_a_coherent_joint() -> None:
    model = DataEstimationModel()
    config = _config(
        mediq_num_trials=1,
        mediq_likelihood_mode="data_estimation",
    )
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.tasks[0]
    prior_values = (0.4, 0.3, 0.2, 0.1)
    action = MediQAction(
        query="Is a ring form present on the blood smear?",
        outcomes=("Yes", "No", UNAVAILABLE_OUTCOME),
        task=task,
        prior_probabilities=prior_values,
    )
    likelihoods = env.outcome_likelihoods(task.option_labels, action)
    prior = np.asarray(prior_values)
    assert np.allclose(likelihoods.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(prior @ likelihoods, [0.4, 0.4, 0.2], atol=1e-12)
    assert np.allclose(likelihoods[:, 2], 0.2, atol=1e-12)
    assert env._data_estimation_projection_residuals[action] <= 1e-12

    unavailable = MediQObservation(
        reply="The patient cannot answer this question from the supplied record.",
        mapped_outcome=UNAVAILABLE_OUTCOME,
        mapped_cleanly=True,
        selected_fact_indices=(),
        grounded=True,
        relevant=True,
        cannot_answer=True,
    )
    belief = BeliefState(task.option_labels, prior_values)
    updated = env.update_belief_state(
        belief, [(action, unavailable)], model, config
    )
    assert updated.probabilities == pytest.approx(prior_values)

    marginal_prompts = [
        messages
        for batch in model.batches
        for messages in batch
        if "predictive model for the MediQ patient interface"
        in messages[0]["content"]
    ]
    posterior_prompts = [
        messages
        for batch in model.batches
        for messages in batch
        if "hypothetical-evidence clinical judge" in messages[0]["content"]
    ]
    assert len(marginal_prompts) == 1
    assert len(posterior_prompts) == 2
    assert "Do not condition this prediction on any answer option" in (
        marginal_prompts[0][-1]["content"]
    )
    assert "findings associated with different options can coexist" in (
        posterior_prompts[0][-1]["content"]
    )


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


def test_mediq_candidate_parser_rejects_decode_repeat_and_derived_query() -> None:
    model = RoutingMediQModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=3)
    env = MediQEnvironment(config, model).configure_for_run(config)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    belief = BeliefState.uniform(task.option_labels)
    previous_action = MediQAction(
        query="Does the patient report feelings of excessive worry?",
        outcomes=("Yes", "No", UNAVAILABLE_OUTCOME),
        task=task,
    )
    previous_observation = MediQObservation(
        reply="The patient cannot answer this question from the supplied record.",
        mapped_outcome=UNAVAILABLE_OUTCOME,
        mapped_cleanly=True,
        selected_fact_indices=(),
        grounded=True,
        relevant=True,
        cannot_answer=True,
    )
    response = json.dumps(
        {
            "candidates": [
                {
                    "query": "Does the patient report feeling excessive worry?",
                    "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                },
                {
                    "query": "Was the patient treated with an antibiotic?",
                    "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                },
                {
                    "query": "Is the patient hemodynamically stable?",
                    "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                },
            ]
        }
    )
    with pytest.raises(ValueError) as exc_info:
        env._parse_candidates(
            response,
            task,
            belief,
            [(previous_action, previous_observation)],
            3,
        )
    error = str(exc_info.value)
    assert "semantically repeats an earlier query" in error
    assert "management decision rather than patient evidence" in error
    assert "derived clinical judgment" in error


def test_mediq_semantic_candidate_validation_regenerates_invalid_set() -> None:
    model = CandidateRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == [
        "Is the body temperature at least 38 C?"
    ]
    assert env._candidate_metrics() == {
        "candidate_validation_checks": 2.0,
        "candidate_validation_retries": 1.0,
        "candidate_validation_failures": 0.0,
        "candidate_set_validation_checks": 0.0,
        "candidate_set_validation_rejections": 0.0,
    }


def test_mediq_naive_retry_requests_multiple_replacement_concepts() -> None:
    model = NaiveRetryDiversityModel()
    config = _config(
        method_names=["naive"],
        mediq_num_trials=1,
        mediq_num_candidates=1,
    )
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    action = env.generate_naive_action(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert action.query == "Is the patient currently experiencing a fever?"
    generation_prompts = [
        messages[-1]["content"]
        for batch in model.batches
        for messages in batch
        if "generate atomic patient questions" in messages[0]["content"]
    ]
    assert "Generate exactly 4 replacement candidate(s)" in generation_prompts[-1]
    assert "Each replacement must test a different observable" in generation_prompts[-1]


def test_mediq_compound_candidate_repair_receives_specific_feedback() -> None:
    model = CompoundQuestionRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=1)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == [
        "Is a ring form present on the blood smear?"
    ]
    assert env._structured_parse_retries == 1


def test_mediq_candidate_repair_retains_valid_queries_and_fills_deficit() -> None:
    model = PartialCandidateRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=2)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == [
        "Is recent endemic travel documented?",
        "Is the blood smear positive for ring forms?",
    ]
    assert env._candidate_metrics() == {
        "candidate_validation_checks": 4.0,
        "candidate_validation_retries": 1.0,
        "candidate_validation_failures": 0.0,
        "candidate_set_validation_checks": 1.0,
        "candidate_set_validation_rejections": 0.0,
    }


def test_mediq_candidate_set_audit_replaces_medical_synonym() -> None:
    model = SynonymCandidateRepairModel()
    config = _config(mediq_num_trials=1, mediq_num_candidates=2)
    env = MediQEnvironment(config, model).configure_for_run(config)
    env.set_questioner(model)
    task = env.sample_hidden_state_for_trial(0, np.random.default_rng(0))
    actions = env.generate_candidate_actions(
        BeliefState.uniform(task.option_labels), [], model, config
    )
    assert [action.query for action in actions] == [
        "Does the patient have a history of renal calculi?",
        "Does the patient have a history of urinary tract obstruction?",
    ]
    assert env._candidate_metrics() == {
        "candidate_validation_checks": 4.0,
        "candidate_validation_retries": 1.0,
        "candidate_validation_failures": 0.0,
        "candidate_set_validation_checks": 2.0,
        "candidate_set_validation_rejections": 1.0,
    }


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
    assert sum(len(batch) for batch in questioner.batches) == 30
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


def test_mediq_data_estimation_integration_logs_joint_components(
    tmp_path: Path,
) -> None:
    questioner = DataEstimationModel()
    answerer = RoutingMediQModel()
    _run_result, summary = run_from_config(
        _config(mediq_likelihood_mode="data_estimation"),
        questioner,
        answerer,
        output_dir=tmp_path,
    )
    assert summary.metrics["selected_joint_projection_residual"][0] <= 1e-10
    records = json.loads((tmp_path / "mediq_interactions.json").read_text())
    candidate = records[0]["turns"][0]["candidate_details"][0]
    assert candidate["data_estimation_marginal"] == pytest.approx(
        {"Yes": 0.4, "No": 0.4, UNAVAILABLE_OUTCOME: 0.2}
    )
    assert set(candidate["data_estimation_elicited_posteriors"]) == {
        "Yes",
        "No",
    }
    assert candidate["joint_projection_residual"] <= 1e-10


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
