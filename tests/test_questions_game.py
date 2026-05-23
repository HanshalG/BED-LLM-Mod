import sys
import types

import pytest


fake_model_module = types.ModuleType("model")


class _ModelBase:
    pass


fake_model_module.Model = _ModelBase
sys.modules.setdefault("model", fake_model_module)

fake_wandb_module = types.ModuleType("wandb")
fake_wandb_module.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", fake_wandb_module)

import questions_game as qg
from helpers import BeliefState, Config, is_guess_correct_via_answerer


class FakeAnswererModel(_ModelBase):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "num_responses": num_responses,
            }
        )
        if not self.responses:
            raise AssertionError("No more responses configured")
        return [self.responses.pop(0)]


class StubQuestionerModel(_ModelBase):
    def chat_complete(self, messages, temperature=0.0, num_responses=1):
        return [""]

    def chat_complete_messages_batched(
        self, batch_messages, temperature=0.0, block_size=50, max_new_tokens=8192
    ):
        return [""] * len(batch_messages)

    def chat_probabilities_messages_batched(
        self, messages, responses, temperature=0.0, block_size=50
    ):
        return [{label: 1.0 / len(responses) for label in responses} for _ in messages]


_BED_PATCH_TARGETS = {
    "generate_original_beliefs": "helpers.generate_original_beliefs",
    "initialize_belief_state": "update_beliefs.initialize_belief_state",
    "generate_candidate_questions": "generate_candidate_questions.generate_candidate_questions",
    "generate_candidate_question_naive": "generate_candidate_questions.generate_candidate_question_naive",
    "evaluate_questions_forward_search": "generate_candidate_questions.evaluate_questions_forward_search",
    "get_question_answered": "helpers.get_question_answered",
    "update_beliefs_batched": "update_beliefs.update_beliefs_batched",
    "sample_beliefs": "sample_beliefs.sample_beliefs",
    "sample_beliefs_naive": "sample_beliefs.sample_beliefs_naive",
    "is_guess_correct_via_answerer": "helpers.is_guess_correct_via_answerer",
    "format_belief_state": "helpers.format_belief_state",
    "write_to_log": "helpers.write_to_log",
    "print_and_log": "helpers.print_and_log",
}

# AnimalsBEDEnvironment binds these at import time; patch both the source module and the adapter.
_ANIMALS_ENV_SYMBOLS = frozenset(_BED_PATCH_TARGETS) - {"format_belief_state", "write_to_log", "print_and_log"}


def _bed_patch(monkeypatch, name: str, value) -> None:
    target = _BED_PATCH_TARGETS.get(name, f"questions_game.{name}")
    monkeypatch.setattr(target, value)
    if name in _ANIMALS_ENV_SYMBOLS:
        import environments.animals.env as animals_env

        if hasattr(animals_env, name):
            monkeypatch.setattr(animals_env, name, value)
    if name in {"evaluate_questions_forward_search", "generate_candidate_questions"}:
        import methods.animals_special as animals_methods

        if hasattr(animals_methods, name):
            monkeypatch.setattr(animals_methods, name, value)


@pytest.fixture(autouse=True)
def _suppress_file_logging_unless_configured(monkeypatch):
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)
    _bed_patch(monkeypatch, "print_and_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(qg, "write_to_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(qg, "print_and_log", lambda *args, **kwargs: None)


def _make_config() -> Config:
    return Config(
        belief_state_mode="uniform",
        generation_temperature_diverse=1.0,
        generation_temperature_simple=1.0,
        answer_temperature=0.7,
        search_depth=1,
        target_num_questions=1,
        log_path=None,
    )


def test_is_guess_correct_via_answerer_returns_true_only_for_correct():
    answerer = FakeAnswererModel(["Correct!", "Yes"])

    assert is_guess_correct_via_answerer("Common badger", "European badger", answerer, 0.7) is True
    assert is_guess_correct_via_answerer("Otter", "European badger", answerer, 0.7) is False
    assert answerer.calls[0]["messages"][-1]["content"] == "Is it Common badger?"
    assert answerer.calls[1]["messages"][-1]["content"] == "Is it Otter?"


def test_complex_guess_fast_path_skips_answerer_validation(monkeypatch):
    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat"], [1.0]),
    )
    _bed_patch(monkeypatch, "format_belief_state", lambda beliefs: str(beliefs))
    _bed_patch(monkeypatch, "generate_candidate_questions", lambda *args, **kwargs: ["Is it a mammal?"])
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch,
        "update_beliefs_batched",
        lambda *args, **kwargs: BeliefState(["Jerboa"], [1.0]),
    )
    _bed_patch(monkeypatch, "sample_beliefs", lambda *args, **kwargs: "Jerboa")
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    helper_calls: list[tuple[str, str]] = []

    def fake_validate(guess, goal, answerer, temperature):
        helper_calls.append((guess, goal))
        return False

    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", fake_validate)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="Jerboa",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=_make_config(),
    )

    assert result[0] == 1
    assert helper_calls == []


def test_complex_guess_fallback_uses_answerer_validation_when_exact_match_fails(monkeypatch):
    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat"], [1.0]),
    )
    _bed_patch(monkeypatch, "format_belief_state", lambda beliefs: str(beliefs))
    _bed_patch(monkeypatch, "generate_candidate_questions", lambda *args, **kwargs: ["Is it a mammal?"])
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch,
        "update_beliefs_batched",
        lambda *args, **kwargs: BeliefState(["Common badger"], [1.0]),
    )
    _bed_patch(monkeypatch, "sample_beliefs", lambda *args, **kwargs: "Common badger")
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: True)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="European badger",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=_make_config(),
    )

    assert result[0] == 1


def test_naive_guess_fallback_uses_answerer_validation_when_exact_match_fails(monkeypatch):
    _bed_patch(monkeypatch, "generate_candidate_question_naive", lambda *args, **kwargs: "Is it nocturnal?")
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch, "sample_beliefs_naive", lambda *args, **kwargs: "Common badger")
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: True)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    config = _make_config()
    result = qg.twenty_questions_animals_single_naive(
        goal_animal="European badger",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result[0] == 1
    assert result.correct_belief_mass == [0.0] * config.animals_num_rounds


def test_naive_guess_fallback_does_not_mark_correct_for_yes_response(monkeypatch):
    _bed_patch(monkeypatch, "generate_candidate_question_naive", lambda *args, **kwargs: "Is it nocturnal?")
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch, "sample_beliefs_naive", lambda *args, **kwargs: "Common badger")
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: False)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    config = _make_config()
    result = qg.twenty_questions_animals_single_naive(
        goal_animal="European badger",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result[0] == 0
    assert result.correct_belief_mass == [0.0] * config.animals_num_rounds


def test_naive_correct_answer_does_not_create_belief_mass(monkeypatch):
    _bed_patch(monkeypatch, "generate_candidate_question_naive", lambda *args, **kwargs: "Is it cat?")
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "Correct!")
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    config = _make_config()
    result = qg.twenty_questions_animals_single_naive(
        goal_animal="cat",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result.correct_guess == [1] * config.animals_num_rounds
    assert result.correct_belief_mass == [0.0] * config.animals_num_rounds


def test_naive_game_passes_configured_prior_to_question_and_guess(monkeypatch):
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.5

    question_priors = []

    def fake_generate_question(
        history_questioner,
        questioner,
        generation_temperature,
        prior_beliefs=None,
        **kwargs,
    ):
        question_priors.append(prior_beliefs)
        return "Is it feline?"

    _bed_patch(monkeypatch, "generate_candidate_question_naive", fake_generate_question)
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch, "sample_beliefs", lambda *args, **kwargs: "cat")
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: True)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals_single_naive(
        goal_animal="cat",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result[0] == 1
    assert question_priors
    assert question_priors[0].beliefs == ["cat", "dog"]


def test_naive_belief_registered_and_callable():
    assert qg.extraction_methods["naive+belief"] is qg.twenty_questions_animals_single_naive_belief


def test_naive_belief_generates_naive_question_from_beliefs_updates_beliefs_and_skips_forward_search(monkeypatch):
    config = _make_config()

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat", "dog"], [0.5, 0.5]),
    )
    _bed_patch(monkeypatch, "format_belief_state", lambda beliefs: str(beliefs))
    _bed_patch(monkeypatch,
        "evaluate_questions_forward_search",
        lambda *args, **kwargs: pytest.fail("naive+belief should not score candidate questions"),
    )
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    seen_prior_beliefs = []
    seen_question_histories = []

    seen_belief_context_labels = []

    def fake_generate_candidate_question_naive(
        history,
        questioner,
        temperature,
        prior_beliefs=None,
        belief_context_label="prior distribution",
    ):
        seen_prior_beliefs.append(prior_beliefs)
        seen_belief_context_labels.append(belief_context_label)
        seen_question_histories.append(list(history))
        return "Is it feline?"

    answers = iter(["Yes", "Correct!"])
    seen_update_histories = []

    def fake_update_beliefs(history, beliefs, questioner, deterministic, config):
        seen_update_histories.append(list(history))
        assert deterministic is False
        return BeliefState(["cat", "dog"], [0.75, 0.25])

    _bed_patch(monkeypatch, "generate_candidate_question_naive", fake_generate_candidate_question_naive)
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: next(answers))
    _bed_patch(monkeypatch, "update_beliefs_batched", fake_update_beliefs)
    _bed_patch(monkeypatch, "sample_beliefs", lambda *args, **kwargs: "dog")
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: False)

    result = qg.twenty_questions_animals_single_naive_belief(
        goal_animal="cat",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert [beliefs.beliefs for beliefs in seen_prior_beliefs] == [["cat", "dog"], ["cat", "dog"]]
    assert [beliefs.probabilities for beliefs in seen_prior_beliefs] == [[0.5, 0.5], [0.75, 0.25]]
    assert seen_belief_context_labels == [
        "current posterior belief state",
        "current posterior belief state",
    ]
    assert seen_question_histories[0] == []
    assert seen_question_histories[1] == [
        {"role": "assistant", "content": "Is it feline?"},
        {"role": "user", "content": "Yes"},
    ]
    assert seen_update_histories[0] == seen_question_histories[1]
    assert result.correct_belief_mass[0] == pytest.approx(0.75)
    assert result.correct_belief_mass[1:] == [1.0] * (config.animals_num_rounds - 1)


def test_naive_belief_uses_prior_support_when_generation_disabled(monkeypatch):
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.5
    config.belief_generation_enabled = False

    initialized_beliefs = []

    def fake_initialize_belief_state(beliefs, history, questioner, config):
        initialized_beliefs.append(list(beliefs))
        return BeliefState(list(beliefs), [0.5, 0.5])

    _bed_patch(monkeypatch,
        "generate_original_beliefs",
        lambda *args, **kwargs: pytest.fail("generation should be skipped when belief_generation_enabled=false"),
    )
    _bed_patch(monkeypatch, "initialize_belief_state", fake_initialize_belief_state)
    _bed_patch(monkeypatch, "format_belief_state", lambda beliefs: str(beliefs))
    _bed_patch(monkeypatch, "generate_candidate_question_naive", lambda *args, **kwargs: "Is it cat?")
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "Correct!")
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals_single_naive_belief(
        goal_animal="cat",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert initialized_beliefs == [["cat", "dog"]]
    assert result.correct_guess == [1] * config.animals_num_rounds
    assert result.correct_belief_mass == [1.0] * config.animals_num_rounds


def test_naive_belief_categorical_guesses_top_probability_belief(monkeypatch):
    config = _make_config()
    config.belief_state_mode = "categorical"

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["dog", "cat"], [0.6, 0.4]),
    )
    _bed_patch(monkeypatch, "format_belief_state", lambda beliefs: str(beliefs))
    _bed_patch(monkeypatch, "generate_candidate_question_naive", lambda *args, **kwargs: "Is it feline?")
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "No")
    _bed_patch(monkeypatch,
        "update_beliefs_batched",
        lambda *args, **kwargs: BeliefState(["cat", "dog"], [0.8, 0.2]),
    )
    _bed_patch(monkeypatch,
        "sample_beliefs",
        lambda *args, **kwargs: pytest.fail("categorical naive+belief should use the top weighted belief"),
    )
    _bed_patch(monkeypatch,
        "evaluate_questions_forward_search",
        lambda *args, **kwargs: pytest.fail("naive+belief should not score candidate questions"),
    )
    _bed_patch(monkeypatch, "is_guess_correct_via_answerer", lambda guess, goal, answerer, temperature: False)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals_single_naive_belief(
        goal_animal="cat",
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result.correct_guess[0] == 1
    assert result.correct_belief_mass[0] == pytest.approx(0.8)


def test_complex_game_uses_configured_search_depth(monkeypatch):
    config = _make_config()
    config.search_depth = 2

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat", "dog"], [0.5, 0.5]),
    )
    _bed_patch(monkeypatch, "generate_candidate_questions", lambda *args, **kwargs: ["Is it feline?", "Does it bark?"])
    _bed_patch(monkeypatch, "get_question_answered", lambda question, goal, answerer, temp: "Correct!")
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    seen_depths = []

    def fake_evaluate_questions_forward_search(*args, **kwargs):
        seen_depths.append(kwargs["depth"])
        return [0.1, 0.9]

    _bed_patch(monkeypatch, "evaluate_questions_forward_search", fake_evaluate_questions_forward_search)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="cat",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert result[0] == 1
    assert seen_depths == [2]


def test_complex_categorical_guess_threshold_asks_identity_question(monkeypatch):
    config = _make_config()
    config.belief_state_mode = "categorical"
    config.belief_guess_threshold = 0.99

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat", "dog"], [0.99, 0.01]),
    )
    _bed_patch(monkeypatch,
        "generate_candidate_questions",
        lambda *args, **kwargs: pytest.fail("threshold guessing should skip candidate generation"),
    )
    _bed_patch(monkeypatch,
        "evaluate_questions_forward_search",
        lambda *args, **kwargs: pytest.fail("threshold guessing should skip EIG scoring"),
    )
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    asked_questions = []

    def fake_get_question_answered(question, goal, answerer, temp):
        asked_questions.append(question)
        return "Correct!"

    _bed_patch(monkeypatch, "get_question_answered", fake_get_question_answered)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="cat",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert asked_questions == ["Is it cat?"]
    assert result.correct_guess == [1] * config.animals_num_rounds
    assert result.correct_belief_mass == [1.0] * config.animals_num_rounds


def test_complex_categorical_below_guess_threshold_uses_eig_path(monkeypatch):
    config = _make_config()
    config.belief_state_mode = "categorical"
    config.belief_guess_threshold = 0.99

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat", "dog"], [0.98, 0.02]),
    )
    _bed_patch(monkeypatch, "generate_candidate_questions", lambda *args, **kwargs: ["Is it feline?", "Does it bark?"])
    _bed_patch(monkeypatch, "evaluate_questions_forward_search", lambda *args, **kwargs: [0.1, 0.9])
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    asked_questions = []

    def fake_get_question_answered(question, goal, answerer, temp):
        asked_questions.append(question)
        return "Correct!"

    _bed_patch(monkeypatch, "get_question_answered", fake_get_question_answered)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="cat",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert asked_questions == ["Does it bark?"]
    assert result.correct_guess == [1] * config.animals_num_rounds


def test_complex_categorical_guess_threshold_none_disables_identity_question(monkeypatch):
    config = _make_config()
    config.belief_state_mode = "categorical"
    config.belief_guess_threshold = None

    _bed_patch(monkeypatch, "generate_original_beliefs", lambda questioner, config: ["cat", "dog"])
    _bed_patch(monkeypatch,
        "initialize_belief_state",
        lambda beliefs, history, questioner, config: BeliefState(["cat", "dog"], [1.0, 0.0]),
    )
    _bed_patch(monkeypatch, "generate_candidate_questions", lambda *args, **kwargs: ["Is it feline?", "Does it bark?"])
    _bed_patch(monkeypatch, "evaluate_questions_forward_search", lambda *args, **kwargs: [0.1, 0.9])
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    asked_questions = []

    def fake_get_question_answered(question, goal, answerer, temp):
        asked_questions.append(question)
        return "Correct!"

    _bed_patch(monkeypatch, "get_question_answered", fake_get_question_answered)

    result = qg.twenty_questions_animals_single_complex(
        goal_animal="cat",
        eig=True,
        deterministic=False,
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        config=config,
    )

    assert asked_questions == ["Does it bark?"]
    assert result.correct_guess == [1] * config.animals_num_rounds


def test_twenty_questions_samples_answerer_targets_from_seeded_prior(monkeypatch):
    config = _make_config()
    config.animals = [["cat", "dog", "wolf"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.5
    config.answerer_sample_from_prior = True
    config.answerer_num_prior_trials = 4
    config.answerer_prior_seed = 7

    seen_targets = []

    def fake_trial(goal_animal, method_name, questioner, answerer, config):
        seen_targets.append(goal_animal)
        return qg.GameMetrics(correct_guess=[1] * config.animals_num_rounds, correct_belief_mass=[0.0] * config.animals_num_rounds)

    monkeypatch.setattr(qg, "_run_single_animal_trial", fake_trial)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals(
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        target_animals=["ignored"],
        extraction_method_name="EIG",
        config=config,
    )

    assert len(seen_targets) == 4
    assert set(seen_targets) <= {"cat", "dog", "wolf"}
    assert result == [1.0] * config.animals_num_rounds


def test_twenty_questions_can_sample_answerer_from_exponential_while_questioner_prior_is_uniform(monkeypatch):
    config = _make_config()
    config.animals = [["cat", "dog", "wolf"]]
    config.belief_prior_mode = "uniform"
    config.answerer_sample_from_prior = True
    config.answerer_prior_mode = "exponential_rank"
    config.answerer_prior_exponential_rate = 0.5
    config.answerer_num_prior_trials = 4
    config.answerer_prior_seed = 7

    seen_targets = []
    seen_questioner_probabilities = []

    def fake_trial(goal_animal, method_name, questioner, answerer, config):
        seen_targets.append(goal_animal)
        questioner_prior = qg.get_configured_prior(config)
        seen_questioner_probabilities.append(questioner_prior.probabilities)
        assert config.active_prior_animals is None
        assert config.active_answerer_prior_animals is None
        return qg.GameMetrics(correct_guess=[1] * config.animals_num_rounds, correct_belief_mass=[0.0] * config.animals_num_rounds)

    monkeypatch.setattr(qg, "_run_single_animal_trial", fake_trial)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    result = qg.twenty_questions_animals(
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        target_animals=["ignored"],
        extraction_method_name="EIG",
        config=config,
    )

    assert len(seen_targets) == 4
    assert set(seen_targets) <= {"cat", "dog", "wolf"}
    for probabilities in seen_questioner_probabilities:
        assert probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result == [1.0] * config.animals_num_rounds


def test_twenty_questions_uses_fixed_targets_by_default(monkeypatch):
    config = _make_config()
    seen_targets = []

    def fake_trial(goal_animal, method_name, questioner, answerer, config):
        seen_targets.append(goal_animal)
        return qg.GameMetrics(correct_guess=[0] * config.animals_num_rounds, correct_belief_mass=[0.0] * config.animals_num_rounds)

    monkeypatch.setattr(qg, "_run_single_animal_trial", fake_trial)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    qg.twenty_questions_animals(
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        target_animals=["cat", "dog"],
        extraction_method_name="EIG",
        config=config,
    )

    assert seen_targets == ["cat", "dog"]


def test_twenty_questions_randomized_answerer_prior_order_does_not_change_questioner_prior(monkeypatch):
    config = _make_config()
    config.animals = [["cat", "dog", "wolf"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.5
    config.answerer_sample_from_prior = True
    config.answerer_randomize_prior_order_per_trial = True
    config.answerer_num_prior_trials = 3
    config.answerer_prior_seed = 7

    seen_questioner_prior_orders = []

    def fake_trial(goal_animal, method_name, questioner, answerer, config):
        questioner_prior = qg.get_configured_prior(config)
        seen_questioner_prior_orders.append(questioner_prior.beliefs)
        assert config.active_prior_animals is None
        assert config.active_answerer_prior_animals is None
        return qg.GameMetrics(
            correct_guess=[0] * config.animals_num_rounds,
            correct_belief_mass=[0.0] * config.animals_num_rounds,
        )

    monkeypatch.setattr(qg, "_run_single_animal_trial", fake_trial)
    _bed_patch(monkeypatch, "write_to_log", lambda *args, **kwargs: None)

    qg.twenty_questions_animals(
        questioner=StubQuestionerModel(),
        answerer=FakeAnswererModel([]),
        target_animals=["ignored"],
        extraction_method_name="EIG",
        config=config,
    )

    assert seen_questioner_prior_orders == [
        ["cat", "dog", "wolf"],
        ["cat", "dog", "wolf"],
        ["cat", "dog", "wolf"],
    ]
    assert config.active_prior_animals is None
    assert config.active_answerer_prior_animals is None


def test_run_single_animal_trial_uses_animals_num_rounds_from_config(monkeypatch):
    config = _make_config()
    config.animals_num_rounds = 7
    captured: dict[str, int] = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured["num_rounds"] = kwargs["num_rounds"]

        def run_single_trial(self, trial_index):
            from core.bed_runner import TrialResult

            return TrialResult(
                trial_index=trial_index,
                hidden_state="cat",
                rounds=(),
                final_metrics={},
            )

    monkeypatch.setattr("core.BEDRunner", _CapturingRunner)
    monkeypatch.setattr("core.build_method", lambda env_name, name, cfg: object())
    monkeypatch.setattr("core.defaults.register_defaults", lambda: None)

    result = qg._run_single_animal_trial(
        "cat",
        "EIG",
        StubQuestionerModel(),
        FakeAnswererModel([]),
        config,
    )

    assert captured["num_rounds"] == 7
    assert len(result.correct_guess) == 7
    assert len(result.correct_belief_mass) == 7
