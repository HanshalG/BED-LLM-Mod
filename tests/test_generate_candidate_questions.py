import math
import re
import sys
import types

import pytest

fake_wandb_module = types.ModuleType("wandb")
fake_wandb_module.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", fake_wandb_module)

fake_model_module = types.ModuleType("model")


class _ModelBase:
    pass


fake_model_module.Model = _ModelBase
sys.modules.setdefault("model", fake_model_module)

import environments.animals.questions as gcq
from core import BeliefState
from helpers import Config


class FakeModel(_ModelBase):
    def __init__(self, probabilities: dict[tuple[str, str], dict[str, float]]):
        self.probabilities = probabilities
        self.messages: list[list[dict[str, str]]] = []

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        raise AssertionError("chat_complete should not be used in these tests")

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        self.messages.extend(messages)
        results = []
        for conversation in messages:
            entity, question = _extract_entity_and_question(conversation)
            try:
                results.append(self.probabilities[(entity, question)])
            except KeyError as exc:
                raise AssertionError(f"Missing probability fixture for {(entity, question)}") from exc
        return results


class RecordingQuestionModel(_ModelBase):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.user_prompts: list[str] = []

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        self.user_prompts.append(messages[-1]["content"])
        if not self.responses:
            raise AssertionError("No more responses configured")
        return [self.responses.pop(0)]

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        raise AssertionError("chat_probabilities_messages_batched should not be used in this test")


def _extract_entity(system_prompt: str) -> str:
    match = re.search(r"Your chosen entity is:\s*(.*?)\s*When asked", system_prompt, re.DOTALL)
    if match is None:
        raise AssertionError(f"Could not parse entity from prompt: {system_prompt}")
    return match.group(1).strip()


def _extract_entity_and_question(conversation: list[dict[str, str]]) -> tuple[str, str]:
    user_prompt = conversation[-1]["content"]
    entity_match = re.search(
        r"Hypothesized target animal:\n(.*?)\n\nQuestion:\n",
        user_prompt,
        re.DOTALL,
    )
    question_match = re.search(
        r"\n\nQuestion:\n(.*?)\n\nAllowed answer labels:",
        user_prompt,
        re.DOTALL,
    )
    if entity_match is not None and question_match is not None:
        return entity_match.group(1).strip(), question_match.group(1).strip()

    return _extract_entity(conversation[0]["content"]), user_prompt


def _make_config() -> Config:
    return Config(
        generation_temperature_diverse=0.0,
        answer_temperature=1.0,
        target_num_questions=2,
        num_mc_samples=10,
        batched_block_size=16,
        threshold_rejection_probability=0.2,
    )


def _belief_names(beliefs):
    if isinstance(beliefs, BeliefState):
        return list(beliefs.hypotheses)
    return beliefs


def test_evaluate_questions_forward_search_depth_1_matches_batched():
    beliefs = ["cat", "dog", "wolf"]
    cand_questions = ["Is it feline?", "Does it bark?"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
            ("wolf", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
            ("cat", "Does it bark?"): {"Yes": 0.0, "No": 1.0},
            ("dog", "Does it bark?"): {"Yes": 1.0, "No": 0.0},
            ("wolf", "Does it bark?"): {"Yes": 1.0, "No": 0.0},
        }
    )

    direct_scores = gcq.evaluate_questions_batched(
        beliefs,
        cand_questions,
        eig=True,
        deterministic=False,
        questioner=model,
        answer_temperature=config.answer_temperature,
        num_mc_samples=config.num_mc_samples,
        block_size=config.batched_block_size,
    )
    forward_scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        cand_questions,
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=1,
    )

    assert forward_scores == pytest.approx(direct_scores)


def test_evaluate_questions_batched_uses_belief_probabilities_when_available():
    beliefs = BeliefState(
        hypotheses=["cat", "dog"],
        probabilities=[0.9, 0.1],
    )
    cand_questions = ["Is it feline?", "Does it bark?"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
            ("cat", "Does it bark?"): {"Yes": 0.2, "No": 0.8},
            ("dog", "Does it bark?"): {"Yes": 1.0, "No": 0.0},
        }
    )

    scores = gcq.evaluate_questions_batched(
        beliefs,
        cand_questions,
        eig=True,
        deterministic=False,
        questioner=model,
        answer_temperature=config.answer_temperature,
        num_mc_samples=config.num_mc_samples,
        block_size=config.batched_block_size,
    )

    expected_first = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
    expected_second = (
        -0.28 * math.log(0.28) - 0.72 * math.log(0.72)
        - (0.9 * (-0.2 * math.log(0.2) - 0.8 * math.log(0.8)))
    )
    assert scores == pytest.approx([expected_first, expected_second])


def test_evaluate_questions_batched_builds_dedicated_likelihood_messages():
    beliefs = ["Wolverine"]
    cand_questions = ["Is it native to North America?"]
    config = _make_config()
    model = FakeModel(
        {
            ("Wolverine", "Is it native to North America?"): {"Yes": 0.95, "No": 0.05},
        }
    )

    gcq.evaluate_questions_batched(
        beliefs,
        cand_questions,
        eig=True,
        deterministic=True,
        questioner=model,
        answer_temperature=config.answer_temperature,
        num_mc_samples=config.num_mc_samples,
        block_size=config.batched_block_size,
    )

    assert len(model.messages) == 1
    system_prompt = model.messages[0][0]["content"]
    user_prompt = model.messages[0][1]["content"]
    assert "You estimate answer likelihoods" in system_prompt
    assert "reply exactly" not in system_prompt
    assert "Hypothesized target animal:\nWolverine" in user_prompt
    assert "Question:\nIs it native to North America?" in user_prompt
    assert "values must sum to 1" in user_prompt


def test_generate_candidate_questions_uses_weighted_backfill_prompt_for_categorical_state():
    beliefs = BeliefState(
        hypotheses=["cat", "dog", "wolf"],
        probabilities=[0.8, 0.15, 0.05],
    )
    model = RecordingQuestionModel(["Is it feline?", "Does it bark?"])

    questions = gcq.generate_candidate_questions(
        beliefs,
        [],
        model,
        generation_temperature=0.0,
        num_questions=2,
    )

    assert questions == ["Is it feline?", "Does it bark?"]
    assert "beliefs list with probabilities" in model.user_prompts[0]
    assert "beliefs list with probabilities" in model.user_prompts[1]
    assert "current candidate questions: ['Is it feline?']" in model.user_prompts[1]


def test_generate_candidate_questions_logs_weighted_summary_for_categorical_state(capsys):
    beliefs = BeliefState(
        hypotheses=["cat", "dog", "wolf"],
        probabilities=[0.8, 0.15, 0.05],
    )
    model = RecordingQuestionModel(["Is it feline?", "Does it bark?"])

    gcq.generate_candidate_questions(
        beliefs,
        [],
        model,
        generation_temperature=0.0,
        num_questions=2,
    )

    captured = capsys.readouterr()
    assert "[categorical] Candidate generation weights: 3 belief(s): [cat (0.800), dog (0.150), wolf (0.050)]" in captured.out
    assert "[categorical] Candidate questions after conditional pass (1): ['Is it feline?']" in captured.out
    assert "[categorical] Candidate questions after backfill (2): ['Is it feline?', 'Does it bark?']" in captured.out


def test_generate_candidate_questions_logs_all_questions_without_truncation(capsys):
    beliefs = BeliefState(
        hypotheses=["cat", "dog", "wolf"],
        probabilities=[0.8, 0.15, 0.05],
    )
    model = RecordingQuestionModel(
        [
            "\n".join(
                [
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5",
                    "Q6",
                ]
            )
        ]
    )

    questions = gcq.generate_candidate_questions(
        beliefs,
        [],
        model,
        generation_temperature=0.0,
        num_questions=6,
    )

    captured = capsys.readouterr()
    assert questions == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert "[categorical] Candidate questions after conditional pass (6): ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']" in captured.out


def test_generate_candidate_question_naive_includes_prior_when_provided():
    prior = BeliefState(
        hypotheses=["cat", "dog"],
        probabilities=[0.8, 0.2],
    )
    model = RecordingQuestionModel(["Is it feline?"])

    question = gcq.generate_candidate_question_naive(
        [],
        model,
        generation_temperature=0.0,
        prior_beliefs=prior,
    )

    assert question == "Is it feline?"
    assert "prior distribution over possible target animals" in model.user_prompts[0]
    assert "cat: 0.800" in model.user_prompts[0]
    assert "dog: 0.200" in model.user_prompts[0]


def test_generate_candidate_question_naive_can_label_current_posterior_belief_state():
    beliefs = BeliefState(
        hypotheses=["cat", "dog"],
        probabilities=[0.8, 0.2],
    )
    model = RecordingQuestionModel(["Is it feline?"])

    question = gcq.generate_candidate_question_naive(
        [],
        model,
        generation_temperature=0.0,
        prior_beliefs=beliefs,
        belief_context_label="current posterior belief state",
    )

    assert question == "Is it feline?"
    assert "current posterior belief state over possible target animals" in model.user_prompts[0]
    assert "Use these probabilities as context" in model.user_prompts[0]


def test_generate_candidate_question_naive_uses_plain_prompt_without_prior():
    model = RecordingQuestionModel(["Is it feline?"])

    gcq.generate_candidate_question_naive([], model, generation_temperature=0.0)

    assert "prior distribution over possible target animals" not in model.user_prompts[0]


def test_evaluate_questions_forward_search_depth_2_adds_expected_future_value(monkeypatch):
    beliefs = ["cat", "lion", "dog", "wolf"]
    primary_question = "Is it feline?"
    config = _make_config()
    model = FakeModel(
        {
            ("cat", primary_question): {"Yes": 1.0, "No": 0.0},
            ("lion", primary_question): {"Yes": 1.0, "No": 0.0},
            ("dog", primary_question): {"Yes": 0.0, "No": 1.0},
            ("wolf", primary_question): {"Yes": 0.0, "No": 1.0},
            ("cat", "Does it meow?"): {"Yes": 1.0, "No": 0.0},
            ("lion", "Does it meow?"): {"Yes": 0.0, "No": 1.0},
            ("dog", "Does it bark?"): {"Yes": 1.0, "No": 0.0},
            ("wolf", "Does it bark?"): {"Yes": 0.0, "No": 1.0},
        }
    )

    def fake_generate_candidate_questions(beliefs, history_questioner, questioner, generation_temperature, num_questions, **kwargs):
        assert num_questions == config.target_num_questions
        assert history_questioner[-2]["content"] == primary_question
        if history_questioner[-1]["content"] == "Yes":
            assert _belief_names(beliefs) == ["cat", "lion"]
            return ["Does it meow?"]
        assert _belief_names(beliefs) == ["dog", "wolf"]
        return ["Does it bark?"]

    def fake_update_beliefs_batched(history, beliefs, questioner, deterministic, config):
        if history[-1]["content"] == "Yes":
            return ["cat", "lion"]
        return ["dog", "wolf"]

    monkeypatch.setattr(gcq, "generate_candidate_questions", fake_generate_candidate_questions)
    monkeypatch.setattr(gcq, "update_beliefs_batched", fake_update_beliefs_batched)
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        [primary_question],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([2 * math.log(2)])


def test_evaluate_questions_forward_search_depth_3_adds_recursive_future_value(monkeypatch):
    beliefs = ["cat", "lion", "tiger", "leopard", "dog", "wolf", "fox", "hyena"]
    root_question = "Is it feline?"
    feline_question = "Is it a small feline?"
    canine_question = "Does it howl?"
    config = _make_config()
    model = FakeModel(
        {
            ("cat", root_question): {"Yes": 1.0, "No": 0.0},
            ("lion", root_question): {"Yes": 1.0, "No": 0.0},
            ("tiger", root_question): {"Yes": 1.0, "No": 0.0},
            ("leopard", root_question): {"Yes": 1.0, "No": 0.0},
            ("dog", root_question): {"Yes": 0.0, "No": 1.0},
            ("wolf", root_question): {"Yes": 0.0, "No": 1.0},
            ("fox", root_question): {"Yes": 0.0, "No": 1.0},
            ("hyena", root_question): {"Yes": 0.0, "No": 1.0},
            ("cat", feline_question): {"Yes": 1.0, "No": 0.0},
            ("lion", feline_question): {"Yes": 1.0, "No": 0.0},
            ("tiger", feline_question): {"Yes": 0.0, "No": 1.0},
            ("leopard", feline_question): {"Yes": 0.0, "No": 1.0},
            ("dog", canine_question): {"Yes": 1.0, "No": 0.0},
            ("wolf", canine_question): {"Yes": 1.0, "No": 0.0},
            ("fox", canine_question): {"Yes": 0.0, "No": 1.0},
            ("hyena", canine_question): {"Yes": 0.0, "No": 1.0},
            ("cat", "Is it cat?"): {"Yes": 1.0, "No": 0.0},
            ("lion", "Is it cat?"): {"Yes": 0.0, "No": 1.0},
            ("tiger", "Is it tiger?"): {"Yes": 1.0, "No": 0.0},
            ("leopard", "Is it tiger?"): {"Yes": 0.0, "No": 1.0},
            ("dog", "Is it dog?"): {"Yes": 1.0, "No": 0.0},
            ("wolf", "Is it dog?"): {"Yes": 0.0, "No": 1.0},
            ("fox", "Is it fox?"): {"Yes": 1.0, "No": 0.0},
            ("hyena", "Is it fox?"): {"Yes": 0.0, "No": 1.0},
        }
    )
    seen_update_histories = []

    def fake_generate_candidate_questions(beliefs, history_questioner, questioner, generation_temperature, num_questions, **kwargs):
        assert num_questions == config.target_num_questions
        last_question = history_questioner[-2]["content"]
        last_answer = history_questioner[-1]["content"]
        if last_question == root_question:
            return [feline_question] if last_answer == "Yes" else [canine_question]
        if last_question == feline_question:
            return ["Is it cat?"] if last_answer == "Yes" else ["Is it tiger?"]
        if last_question == canine_question:
            return ["Is it dog?"] if last_answer == "Yes" else ["Is it fox?"]
        raise AssertionError(f"Unexpected history: {history_questioner}")

    def fake_update_beliefs_batched(history, beliefs, questioner, deterministic, config):
        seen_update_histories.append(history)
        last_question = history[-2]["content"]
        last_answer = history[-1]["content"]
        if last_question == root_question:
            if last_answer == "Yes":
                return ["cat", "lion", "tiger", "leopard"]
            return ["dog", "wolf", "fox", "hyena"]
        if last_question == feline_question:
            return ["cat", "lion"] if last_answer == "Yes" else ["tiger", "leopard"]
        if last_question == canine_question:
            return ["dog", "wolf"] if last_answer == "Yes" else ["fox", "hyena"]
        raise AssertionError(f"Unexpected update history: {history}")

    monkeypatch.setattr(gcq, "generate_candidate_questions", fake_generate_candidate_questions)
    monkeypatch.setattr(gcq, "update_beliefs_batched", fake_update_beliefs_batched)
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        [root_question],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=3,
    )

    assert scores == pytest.approx([3 * math.log(2)])
    assert seen_update_histories == [
        [{"role": "assistant", "content": root_question}, {"role": "user", "content": "Yes"}],
        [
            {"role": "assistant", "content": root_question},
            {"role": "user", "content": "Yes"},
            {"role": "assistant", "content": feline_question},
            {"role": "user", "content": "Yes"},
        ],
        [
            {"role": "assistant", "content": root_question},
            {"role": "user", "content": "Yes"},
            {"role": "assistant", "content": feline_question},
            {"role": "user", "content": "No"},
        ],
        [{"role": "assistant", "content": root_question}, {"role": "user", "content": "No"}],
        [
            {"role": "assistant", "content": root_question},
            {"role": "user", "content": "No"},
            {"role": "assistant", "content": canine_question},
            {"role": "user", "content": "Yes"},
        ],
        [
            {"role": "assistant", "content": root_question},
            {"role": "user", "content": "No"},
            {"role": "assistant", "content": canine_question},
            {"role": "user", "content": "No"},
        ],
    ]


def test_evaluate_questions_forward_search_uses_full_update_for_stochastic_branches(monkeypatch):
    beliefs = ["cat", "dog"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )
    seen_histories = []

    def fake_update_beliefs_batched(history, beliefs, questioner, deterministic, config):
        seen_histories.append(history)
        return []

    monkeypatch.setattr(gcq, "update_beliefs_batched", fake_update_beliefs_batched)
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)
    history = [{"role": "assistant", "content": "Existing question"}, {"role": "user", "content": "Yes"}]

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        history,
        ["Is it feline?"],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([math.log(2)])
    assert seen_histories == [
        history + [{"role": "assistant", "content": "Is it feline?"}, {"role": "user", "content": "Yes"}],
        history + [{"role": "assistant", "content": "Is it feline?"}, {"role": "user", "content": "No"}],
    ]


def test_candidate_coverage_dynamics_uses_production_branch_updates_without_target_leakage(monkeypatch):
    beliefs = ["cat", "dog"]
    truth = "Secret Animal"
    questions = ["Question A?", "Question B?"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Question A?"): {"Yes": 0.5, "No": 0.5},
            ("dog", "Question A?"): {"Yes": 0.5, "No": 0.5},
            ("cat", "Question B?"): {"Yes": 0.5, "No": 0.5},
            ("dog", "Question B?"): {"Yes": 0.5, "No": 0.5},
        }
    )
    seen_histories = []
    seen_beliefs = []

    def fake_update_beliefs_many(histories, branch_beliefs, questioner, deterministic, config):
        seen_histories.extend(histories)
        seen_beliefs.append(_belief_names(branch_beliefs))
        updated = []
        for history in histories:
            question = history[-2]["content"]
            answer = history[-1]["content"]
            if question == "Question A?":
                updated.append([truth] if answer == "Yes" else ["dog"])
            else:
                updated.append([truth] if answer == "No" else ["cat"])
        return updated

    monkeypatch.setattr(gcq, "_update_beliefs_many", fake_update_beliefs_many)

    dynamics = gcq.evaluate_candidate_coverage_dynamics(
        beliefs,
        [],
        questions,
        truth,
        deterministic=False,
        questioner=model,
        config=config,
    )

    assert [entry.question for entry in dynamics] == questions
    assert [entry.immediate_eig for entry in dynamics] == pytest.approx([0.0, 0.0])
    assert [entry.expected_truth_coverage for entry in dynamics] == pytest.approx([0.5, 0.5])
    assert [entry.expected_current_support_retention for entry in dynamics] == pytest.approx([0.25, 0.25])
    assert [entry.expected_surviving_map_mass for entry in dynamics] == pytest.approx([0.25, 0.25])
    assert [(entry.truth_covered_if_yes, entry.truth_covered_if_no) for entry in dynamics] == [
        (True, False),
        (False, True),
    ]
    assert seen_histories == [
        [{"role": "assistant", "content": "Question A?"}, {"role": "user", "content": "Yes"}],
        [{"role": "assistant", "content": "Question A?"}, {"role": "user", "content": "No"}],
        [{"role": "assistant", "content": "Question B?"}, {"role": "user", "content": "Yes"}],
        [{"role": "assistant", "content": "Question B?"}, {"role": "user", "content": "No"}],
    ]
    assert all(truth not in message["content"] for history in seen_histories for message in history)
    assert seen_beliefs == [beliefs]
    assert all(truth not in branch_beliefs for branch_beliefs in seen_beliefs)


def test_evaluate_questions_forward_search_uses_full_update_for_deterministic_branches(monkeypatch):
    beliefs = ["cat", "dog"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )
    seen_histories = []

    def fake_update_beliefs_batched(history, beliefs, questioner, deterministic, config):
        seen_histories.append(history)
        return ["placeholder"]

    monkeypatch.setattr(gcq, "update_beliefs_batched", fake_update_beliefs_batched)
    monkeypatch.setattr(gcq, "generate_candidate_questions", lambda *args, **kwargs: [])
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    history = [{"role": "assistant", "content": "Existing question"}, {"role": "user", "content": "Yes"}]
    gcq.evaluate_questions_forward_search(
        beliefs,
        history,
        ["Is it feline?"],
        eig=True,
        deterministic=True,
        questioner=model,
        config=config,
        depth=2,
    )

    assert seen_histories == [
        history + [{"role": "assistant", "content": "Is it feline?"}, {"role": "user", "content": "Yes"}],
        history + [{"role": "assistant", "content": "Is it feline?"}, {"role": "user", "content": "No"}],
    ]


def test_evaluate_questions_forward_search_handles_empty_future_beliefs(monkeypatch):
    beliefs = ["cat", "dog"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )

    monkeypatch.setattr(gcq, "update_beliefs_batched", lambda *args, **kwargs: BeliefState([], []))
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        ["Is it feline?"],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([math.log(2)])


def test_evaluate_questions_forward_search_handles_empty_future_questions(monkeypatch):
    beliefs = ["cat", "dog"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )

    monkeypatch.setattr(gcq, "update_beliefs_batched", lambda *args, **kwargs: ["cat"])
    monkeypatch.setattr(gcq, "generate_candidate_questions", lambda *args, **kwargs: [])
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        ["Is it feline?"],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([math.log(2)])


def test_evaluate_questions_forward_search_handles_empty_future_scores(monkeypatch):
    beliefs = ["cat", "dog"]
    config = _make_config()
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )

    monkeypatch.setattr(gcq, "update_beliefs_batched", lambda *args, **kwargs: ["cat"])
    monkeypatch.setattr(gcq, "generate_candidate_questions", lambda *args, **kwargs: ["Follow-up?"])
    monkeypatch.setattr(gcq, "evaluate_questions_batched", lambda *args, **kwargs: [])
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        ["Is it feline?"],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([math.log(2)])


def test_evaluate_questions_forward_search_depth_3_skips_empty_questions_and_zero_probability_branches(monkeypatch):
    beliefs = ["cat", "dog"]
    root_question = "Is it feline?"
    future_question = "Always yes?"
    config = _make_config()
    model = FakeModel(
        {
            ("cat", root_question): {"Yes": 1.0, "No": 0.0},
            ("dog", root_question): {"Yes": 0.0, "No": 1.0},
            ("cat", future_question): {"Yes": 1.0, "No": 0.0},
            ("dog", future_question): {"Yes": 1.0, "No": 0.0},
        }
    )
    seen_histories = []

    def fake_update_beliefs_batched(history, beliefs, questioner, deterministic, config):
        seen_histories.append(history)
        return ["cat", "dog"]

    def fake_generate_candidate_questions(beliefs, history_questioner, questioner, generation_temperature, num_questions, **kwargs):
        if history_questioner[-2]["content"] == root_question and history_questioner[-1]["content"] == "Yes":
            return []
        if history_questioner[-2]["content"] == root_question and history_questioner[-1]["content"] == "No":
            return [future_question]
        if history_questioner[-2]["content"] == future_question:
            return []
        raise AssertionError(f"Unexpected history: {history_questioner}")

    monkeypatch.setattr(gcq, "update_beliefs_batched", fake_update_beliefs_batched)
    monkeypatch.setattr(gcq, "generate_candidate_questions", fake_generate_candidate_questions)
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        [root_question],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=3,
    )

    assert scores == pytest.approx([math.log(2)])
    assert seen_histories == [
        [{"role": "assistant", "content": root_question}, {"role": "user", "content": "Yes"}],
        [{"role": "assistant", "content": root_question}, {"role": "user", "content": "No"}],
        [
            {"role": "assistant", "content": root_question},
            {"role": "user", "content": "No"},
            {"role": "assistant", "content": future_question},
            {"role": "user", "content": "Yes"},
        ],
    ]


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_evaluate_questions_forward_search_rejects_invalid_depth(depth):
    config = _make_config()
    model = FakeModel({})

    with pytest.raises(ValueError, match="search depth must be a positive integer"):
        gcq.evaluate_questions_forward_search(
            ["cat"],
            [],
            ["Is it feline?"],
            eig=True,
            deterministic=False,
            questioner=model,
            config=config,
            depth=depth,
        )


def test_evaluate_questions_forward_search_logs_to_configured_run_file(monkeypatch, tmp_path):
    beliefs = ["cat", "dog"]
    question = "Is it feline?"
    config = _make_config()
    config.log_path = tmp_path / "custom-run.log"
    model = FakeModel(
        {
            ("cat", question): {"Yes": 1.0, "No": 0.0},
            ("dog", question): {"Yes": 0.0, "No": 1.0},
        }
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gcq, "update_beliefs_batched", lambda *args, **kwargs: [])
    monkeypatch.setattr(gcq, "generate_candidate_questions", lambda *args, **kwargs: [])

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        [question],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    assert scores == pytest.approx([math.log(2)])
    assert "Optimal immediate question: Is it feline?" in config.log_path.read_text(encoding="utf-8")
    assert not (tmp_path / "logs" / "log1.txt").exists()


def test_evaluate_questions_forward_search_logs_categorical_branch_summaries(monkeypatch, tmp_path):
    beliefs = BeliefState(
        hypotheses=["cat", "dog"],
        probabilities=[0.6, 0.4],
    )
    question = "Is it feline?"
    config = _make_config()
    config.belief_state_mode = "categorical"
    config.log_path = tmp_path / "categorical-run.log"
    model = FakeModel(
        {
            ("cat", question): {"Yes": 1.0, "No": 0.0},
            ("dog", question): {"Yes": 0.0, "No": 1.0},
        }
    )

    branch_states = {
        "Yes": BeliefState(hypotheses=["cat", "lion"], probabilities=[0.7, 0.3]),
        "No": BeliefState(hypotheses=["dog", "wolf"], probabilities=[0.8, 0.2]),
    }

    monkeypatch.setattr(
        gcq,
        "_future_beliefs_for_answer",
        lambda beliefs, history_questioner, branch_question, answer, questioner, deterministic, config: branch_states[answer],
    )
    monkeypatch.setattr(gcq, "generate_candidate_questions", lambda *args, **kwargs: [])

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        [question],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    expected_score = -0.6 * math.log(0.6) - 0.4 * math.log(0.4)
    assert scores == pytest.approx([expected_score])
    log_text = config.log_path.read_text(encoding="utf-8")
    assert "[categorical] Branch 'Is it feline?' -> Yes (p=0.600): 2 belief(s): [cat (0.700), lion (0.300)]" in log_text
    assert "[categorical] Branch 'Is it feline?' -> No (p=0.400): 2 belief(s): [dog (0.800), wolf (0.200)]" in log_text


def test_evaluate_questions_forward_search_preserves_weighted_future_states(monkeypatch):
    beliefs = BeliefState(
        hypotheses=["cat", "dog"],
        probabilities=[0.6, 0.4],
    )
    config = _make_config()
    config.belief_state_mode = "categorical"
    model = FakeModel(
        {
            ("cat", "Is it feline?"): {"Yes": 1.0, "No": 0.0},
            ("dog", "Is it feline?"): {"Yes": 0.0, "No": 1.0},
        }
    )
    weighted_branch_states = [
        BeliefState(hypotheses=["cat", "lion"], probabilities=[0.8, 0.2]),
        BeliefState(hypotheses=["dog", "wolf"], probabilities=[0.3, 0.7]),
    ]
    generated_probabilities = []
    scored_probabilities = []

    def fake_generate_candidate_questions(beliefs, history_questioner, questioner, generation_temperature, num_questions, **kwargs):
        generated_probabilities.append(list(beliefs.probabilities))
        return ["Follow-up?"]

    def fake_evaluate_questions_batched(beliefs, cand_questions, eig, deterministic, questioner, answer_temperature, num_mc_samples, block_size, **kwargs):
        scored_probabilities.append(list(beliefs.probabilities))
        return [1.0]

    monkeypatch.setattr(gcq, "update_beliefs_batched", lambda *args, **kwargs: weighted_branch_states.pop(0))
    monkeypatch.setattr(gcq, "generate_candidate_questions", fake_generate_candidate_questions)
    monkeypatch.setattr(gcq, "evaluate_questions_batched", fake_evaluate_questions_batched)
    monkeypatch.setattr(gcq, "write_to_log", lambda *args, **kwargs: None)

    scores = gcq.evaluate_questions_forward_search(
        beliefs,
        [],
        ["Is it feline?"],
        eig=True,
        deterministic=False,
        questioner=model,
        config=config,
        depth=2,
    )

    expected_score = -0.6 * math.log(0.6) - 0.4 * math.log(0.4) + 1.0
    assert scores == pytest.approx([expected_score])
    assert generated_probabilities == [[0.8, 0.2], [0.3, 0.7]]
    assert scored_probabilities == [[0.8, 0.2], [0.3, 0.7]]
