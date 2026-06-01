import sys
import types

import pytest

import helpers as helpers_module


fake_model_module = types.ModuleType("model")


class _ModelBase:
    pass


fake_model_module.Model = _ModelBase
sys.modules.setdefault("model", fake_model_module)

from helpers import Config
from environments.animals.beliefs import build_belief_state, filter_valid_animal_names_batched, generate_new_beliefs, initialize_belief_state, update_beliefs_batched


class FakeBeliefScoringModel(_ModelBase):
    def __init__(self, completions: list[str] | list[list[str]] | None = None,
                 batched_completions: list[list[str]] | None = None):
        self.completions = list(completions or [])
        self.batched_completions = list(batched_completions or [])
        self.calls: list[dict[str, object]] = []
        self.batched_calls: list[dict[str, object]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "num_responses": num_responses,
            }
        )
        if not self.completions:
            raise AssertionError("No more completions configured")
        next_completion = self.completions.pop(0)
        if isinstance(next_completion, list):
            return next_completion
        return [next_completion]

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        self.batched_calls.append(
            {
                "batch_messages": batch_messages,
                "temperature": temperature,
                "block_size": block_size,
                "max_new_tokens": max_new_tokens,
            }
        )
        if not self.batched_completions:
            raise AssertionError("No more batched completions configured")
        return self.batched_completions.pop(0)

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used in these tests")


def _make_config() -> Config:
    return Config(
        belief_state_mode="categorical",
        belief_probability_temperature=0.0,
        belief_distribution_num_calls=1,
        batched_block_size=16,
    )


class FakeBeliefGenerationModel(_ModelBase):
    def __init__(self, completion: str):
        self.completion = completion
        self.calls: list[dict[str, object]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "num_responses": num_responses,
            }
        )
        return [self.completion]

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        raise AssertionError("chat_complete_messages_batched should not be used in these tests")

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used in these tests")


class FakeBeliefValidationModel(_ModelBase):
    def __init__(self, batched_completions: list[str]):
        self.batched_completions = list(batched_completions)
        self.batched_calls: list[dict[str, object]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        raise AssertionError("chat_complete should not be used in these tests")

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        self.batched_calls.append(
            {
                "batch_messages": batch_messages,
                "temperature": temperature,
                "block_size": block_size,
                "max_new_tokens": max_new_tokens,
            }
        )
        return list(self.batched_completions)

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used in these tests")


class FakeProbabilityModel(_ModelBase):
    def __init__(self, probability_batches: list[list[dict[str, float]]]):
        self.probability_batches = list(probability_batches)
        self.complete_calls: list[dict[str, object]] = []
        self.probability_calls: list[dict[str, object]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.complete_calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "num_responses": num_responses,
            }
        )
        raise AssertionError("chat_complete should not be used in these tests")

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        raise AssertionError("chat_complete_messages_batched should not be used in these tests")

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        self.probability_calls.append(
            {
                "messages": messages,
                "responses": responses,
                "temperature": temperature,
                "block_size": block_size,
            }
        )
        if not self.probability_batches:
            raise AssertionError("No more probability batches configured")
        return self.probability_batches.pop(0)


def test_build_belief_state_scores_categorical_probabilities_from_single_json_call():
    config = _make_config()
    model = FakeBeliefScoringModel(['{"cat": 2, "dog": 8}'])

    belief_state = build_belief_state(["cat", "dog"], [], model, config)

    assert len(model.calls) == 1
    assert len(model.batched_calls) == 0
    assert model.calls[0]["num_responses"] == 1
    assert belief_state.hypotheses == ("dog", "cat",)
    assert belief_state.probabilities == pytest.approx([0.8, 0.2])


def test_build_belief_state_logs_categorical_summary_to_stdout_and_run_log(tmp_path, capsys):
    config = _make_config()
    config.log_path = tmp_path / "logs" / "run.log"
    model = FakeBeliefScoringModel(['{"cat": 2, "dog": 8}'])

    build_belief_state(["cat", "dog"], [], model, config)

    captured = capsys.readouterr()
    expected = "[categorical] Scored belief distribution (1/1 valid): 2 belief(s): [dog (0.800), cat (0.200)]"
    assert expected in captured.out
    assert expected in (
        config.log_path.read_text(encoding="utf-8")
    )


def test_build_belief_state_ignores_extra_keys_and_assigns_zero_to_missing_keys():
    config = _make_config()
    model = FakeBeliefScoringModel(['{"cat": 3, "fox": 9}'])

    belief_state = build_belief_state(["cat", "dog", "wolf"], [], model, config)

    assert belief_state.hypotheses == ("cat", "dog", "wolf",)
    assert belief_state.probabilities == pytest.approx([1.0, 0.0, 0.0])


def test_build_belief_state_averages_multiple_valid_distribution_completions_from_one_call():
    config = _make_config()
    config.belief_distribution_num_calls = 3
    model = FakeBeliefScoringModel([[
        '{"cat": 1, "dog": 3}',
        '{"cat": 3, "dog": 1}',
        '{"cat": 2, "dog": 2}',
    ]])

    belief_state = build_belief_state(["cat", "dog"], [], model, config)

    assert len(model.calls) == 1
    assert len(model.batched_calls) == 0
    assert model.calls[0]["num_responses"] == 3
    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_build_belief_state_averages_only_valid_distribution_completions():
    config = _make_config()
    config.belief_distribution_num_calls = 3
    model = FakeBeliefScoringModel([[
        "not json",
        '{"cat": 1, "dog": 3}',
        '{"cat": 3, "dog": 1}',
    ]])

    belief_state = build_belief_state(["cat", "dog"], [], model, config)

    assert len(model.calls) == 1
    assert len(model.batched_calls) == 0
    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_build_belief_state_falls_back_to_uniform_when_all_distribution_completions_are_invalid():
    config = _make_config()
    config.belief_distribution_num_calls = 3
    model = FakeBeliefScoringModel([[
        "not json",
        '{"cat": "oops", "dog": 1}',
        '{"cat": -1, "dog": 1}',
    ]])

    belief_state = build_belief_state(["cat", "dog"], [], model, config)

    assert len(model.calls) == 1
    assert len(model.batched_calls) == 0
    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_build_belief_state_falls_back_to_uniform_for_multi_label_zero_mass_payload():
    config = _make_config()
    config.belief_distribution_num_calls = 3
    model = FakeBeliefScoringModel([[
        """{
  "Northern raccoon": 0.0,
  "Striped skunk": 0.0,
  "Virginia raccoon": 0.0,
  "North American raccoon": 0.0,
  "Northern skunk": 0.0,
  "Common raccoon": 0.0,
  "Raccoon": 0.0,
  "American Black Bear": 0.0,
  "Black Bear": 0.0
}""",
        "not json",
        '{"American Black Bear": "oops"}',
    ]])

    beliefs = [
        "Northern raccoon",
        "Striped skunk",
        "Virginia raccoon",
        "North American raccoon",
        "Northern skunk",
        "Common raccoon",
        "Raccoon",
        "American Black Bear",
        "Black Bear",
    ]
    belief_state = build_belief_state(beliefs, [], model, config)

    assert len(model.calls) == 1
    assert len(model.batched_calls) == 0
    assert belief_state.hypotheses == tuple(beliefs)
    assert belief_state.probabilities == pytest.approx([1 / len(beliefs)] * len(beliefs))


def test_build_belief_state_logs_valid_distribution_count_for_partial_batch(tmp_path, capsys):
    config = _make_config()
    config.belief_distribution_num_calls = 3
    config.log_path = tmp_path / "logs" / "run.log"
    model = FakeBeliefScoringModel([[
        '{"cat": 1, "dog": 3}',
        "not json",
        '{"cat": 3, "dog": 1}',
    ]])

    build_belief_state(["cat", "dog"], [], model, config)

    captured = capsys.readouterr()
    expected = "[categorical] Scored belief distribution (2/3 valid): 2 belief(s): [cat (0.500), dog (0.500)]"
    assert expected in captured.out
    assert expected in config.log_path.read_text(encoding="utf-8")


def test_build_belief_state_permutation_mode_batches_one_completion_per_permuted_history(monkeypatch):
    config = _make_config()
    config.belief_distribution_num_calls = 3
    config.belief_distribution_permute_history = True
    history = [
        {"role": "assistant", "content": "Question 1?"},
        {"role": "user", "content": "Yes"},
        {"role": "assistant", "content": "Question 2?"},
        {"role": "user", "content": "No"},
    ]
    model = FakeBeliefScoringModel(
        batched_completions=[[
            '{"cat": 1, "dog": 3}',
            '{"cat": 3, "dog": 1}',
            '{"cat": 2, "dog": 2}',
        ]]
    )
    permutations = iter([[1, 0], [0, 1], [1, 0]])
    monkeypatch.setattr(
        helpers_module.np.random,
        "permutation",
        lambda size: next(permutations),
    )

    belief_state = build_belief_state(["cat", "dog"], history, model, config)

    assert len(model.calls) == 0
    assert len(model.batched_calls) == 1
    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])

    batched_call = model.batched_calls[0]
    assert batched_call["block_size"] == config.batched_block_size
    assert batched_call["max_new_tokens"] == 8192

    histories = [messages[1:-1] for messages in batched_call["batch_messages"]]
    assert histories[0] == [
        {"role": "assistant", "content": "Question 2?"},
        {"role": "user", "content": "No"},
        {"role": "assistant", "content": "Question 1?"},
        {"role": "user", "content": "Yes"},
    ]
    assert histories[1] == history
    assert histories[2] == [
        {"role": "assistant", "content": "Question 2?"},
        {"role": "user", "content": "No"},
        {"role": "assistant", "content": "Question 1?"},
        {"role": "user", "content": "Yes"},
    ]
    for permuted_history in histories:
        assert permuted_history[0]["role"] == "assistant"
        assert permuted_history[1]["role"] == "user"
        assert permuted_history[2]["role"] == "assistant"
        assert permuted_history[3]["role"] == "user"


def test_build_belief_state_permutation_mode_averages_only_valid_samples_and_logs_mode(tmp_path, capsys):
    config = _make_config()
    config.belief_distribution_num_calls = 3
    config.belief_distribution_permute_history = True
    config.log_path = tmp_path / "logs" / "run.log"
    history = [
        {"role": "assistant", "content": "Question 1?"},
        {"role": "user", "content": "Yes"},
        {"role": "assistant", "content": "Question 2?"},
        {"role": "user", "content": "No"},
    ]
    model = FakeBeliefScoringModel(
        batched_completions=[[
            '{"cat": 1, "dog": 3}',
            "not json",
            '{"cat": 3, "dog": 1}',
        ]]
    )

    belief_state = build_belief_state(["cat", "dog"], history, model, config)

    assert len(model.calls) == 0
    assert len(model.batched_calls) == 1
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])
    captured = capsys.readouterr()
    expected = "[categorical] Scored belief distribution (permuted-history, 2/3 valid): 2 belief(s): [cat (0.500), dog (0.500)]"
    assert expected in captured.out
    assert expected in config.log_path.read_text(encoding="utf-8")


def test_build_belief_state_permutation_mode_falls_back_to_uniform_when_all_samples_are_invalid():
    config = _make_config()
    config.belief_distribution_num_calls = 3
    config.belief_distribution_permute_history = True
    history = [
        {"role": "assistant", "content": "Question 1?"},
        {"role": "user", "content": "Yes"},
        {"role": "assistant", "content": "Question 2?"},
        {"role": "user", "content": "No"},
    ]
    model = FakeBeliefScoringModel(
        batched_completions=[[
            "not json",
            '{"cat": "oops", "dog": 1}',
            '{"cat": -1, "dog": 1}',
        ]]
    )

    belief_state = build_belief_state(["cat", "dog"], history, model, config)

    assert len(model.calls) == 0
    assert len(model.batched_calls) == 1
    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_build_belief_state_uniform_mode_emits_no_categorical_trace(tmp_path, capsys):
    config = Config(
        belief_state_mode="uniform",
        log_path=tmp_path / "logs" / "run.log",
    )
    model = FakeBeliefScoringModel(['{"cat": 2, "dog": 8}'])

    belief_state = build_belief_state(["cat", "dog"], [], model, config)

    captured = capsys.readouterr()
    assert belief_state.hypotheses == ("cat", "dog",)
    assert "[categorical]" not in captured.out
    assert not config.log_path.exists()
    assert len(model.calls) == 0


def test_generate_new_beliefs_drops_structurally_invalid_candidates_and_logs_cleanup(tmp_path, capsys):
    config = _make_config()
    config.log_path = tmp_path / "logs" / "run.log"
    model = FakeBeliefGenerationModel(
        "Cassowary\nGround squirrel (specific African species)\nCapybara (No, has fur) -> Green Iguana\nBlue dragon sea slug.\n"
        "Since the logic of the previous answers creates a contradiction"
    )

    beliefs = generate_new_beliefs({"role": "system", "content": "system"}, [], model, 0.8, config)

    assert beliefs == ["Cassowary", "Ground squirrel", "Blue dragon sea slug"]
    captured = capsys.readouterr()
    assert "[beliefs] Generated 5 raw belief(s)" in captured.out
    assert "[beliefs] 3 belief(s) remain after structural cleanup" in captured.out
    assert "[categorical] Structural cleanup retained 3/5 generated belief(s)" in (
        config.log_path.read_text(encoding="utf-8")
    )


def test_filter_valid_animal_names_batched_keeps_only_exact_yes_completions():
    model = FakeBeliefValidationModel(["Yes", "No", "Maybe", " Yes "])

    beliefs = filter_valid_animal_names_batched(
        ["Cassowary", "reasoning text", "Blue dragon sea slug", "Southern cassowary"],
        model,
        block_size=8,
    )

    assert beliefs == ["Cassowary", "Southern cassowary"]
    assert len(model.batched_calls) == 1
    assert model.batched_calls[0]["temperature"] == 0.0
    assert model.batched_calls[0]["max_new_tokens"] == 8192


def test_filter_valid_animal_names_batched_preserves_clean_variants():
    model = FakeBeliefValidationModel(["Yes", "Yes", "Yes"])

    beliefs = filter_valid_animal_names_batched(
        ["Cassowary", "Southern cassowary", "Blue dragon sea slug"],
        model,
        block_size=8,
    )

    assert beliefs == ["Cassowary", "Southern cassowary", "Blue dragon sea slug"]


def test_initialize_belief_state_filters_opening_beliefs_before_building_state():
    config = Config(
        belief_state_mode="uniform",
        batched_block_size=16,
    )
    model = FakeBeliefValidationModel(["Yes", "No"])

    belief_state = initialize_belief_state(
        [
            "Cassowary",
            "year: new String();",
            "Blue dragon sea slug",
            "Sand cat?",
        ],
        [],
        model,
        config,
    )

    assert belief_state.hypotheses == ("Cassowary",)
    assert belief_state.probabilities == pytest.approx([1.0])


def test_build_belief_state_with_prior_uses_bayesian_likelihoods():
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.0
    history = [
        {"role": "assistant", "content": "Does it bark?"},
        {"role": "user", "content": "Yes"},
    ]
    model = FakeProbabilityModel([[
        {"Yes": 0.2, "No": 0.8},
        {"Yes": 0.8, "No": 0.2},
    ]])

    belief_state = build_belief_state(["cat", "dog"], history, model, config)

    assert belief_state.hypotheses == ("dog", "cat",)
    assert belief_state.probabilities == pytest.approx([0.8, 0.2])
    assert len(model.probability_calls) == 1
    assert len(model.probability_calls[0]["messages"]) == 2
    first_message = model.probability_calls[0]["messages"][0]
    assert "You estimate answer likelihoods" in first_message[0]["content"]
    assert "reply exactly" not in first_message[0]["content"]
    assert "Hypothesized target animal:\ncat" in first_message[1]["content"]
    assert "Question:\nDoes it bark?" in first_message[1]["content"]


def test_build_belief_state_with_prior_averages_multiple_likelihood_samples():
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_prior_exponential_rate = 0.0
    config.belief_distribution_num_calls = 2
    history = [
        {"role": "assistant", "content": "Does it bark?"},
        {"role": "user", "content": "Yes"},
    ]
    model = FakeProbabilityModel([[
        {"Yes": 0.2, "No": 0.8},
        {"Yes": 0.8, "No": 0.2},
        {"Yes": 0.6, "No": 0.4},
        {"Yes": 0.4, "No": 0.6},
    ]])

    belief_state = build_belief_state(["cat", "dog"], history, model, config)

    assert belief_state.hypotheses == ("dog", "cat",)
    assert belief_state.probabilities == pytest.approx([0.6, 0.4])
    assert len(model.probability_calls) == 1
    assert len(model.probability_calls[0]["messages"]) == 4


def test_update_beliefs_generation_disabled_filters_prior_without_generation_calls():
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_generation_enabled = False
    history = [
        {"role": "assistant", "content": "Does it bark?"},
        {"role": "user", "content": "Yes"},
    ]
    model = FakeProbabilityModel([
        [
            {"Yes": 0.1, "No": 0.9},
            {"Yes": 0.9, "No": 0.1},
        ],
        [
            {"Yes": 0.9, "No": 0.1},
        ],
    ])

    belief_state = update_beliefs_batched(history, ["cat", "dog"], model, deterministic=False, config=config)

    assert model.complete_calls == []
    assert belief_state.hypotheses == ("dog",)
    assert belief_state.probabilities == pytest.approx([1.0])
    assert len(model.probability_calls) == 2
    first_filter_message = model.probability_calls[1]["messages"][0]
    assert "You estimate answer likelihoods" in first_filter_message[0]["content"]
    assert "Hypothesized target animal:\ndog" in first_filter_message[1]["content"]


def test_update_beliefs_generation_disabled_can_skip_filtering():
    config = _make_config()
    config.animals = [["cat", "dog"]]
    config.belief_prior_mode = "exponential_rank"
    config.belief_generation_enabled = False
    config.belief_filtering_enabled = False
    history = [
        {"role": "assistant", "content": "Does it bark?"},
        {"role": "user", "content": "Yes"},
    ]
    model = FakeProbabilityModel([[
        {"Yes": 0.1, "No": 0.9},
        {"Yes": 0.9, "No": 0.1},
    ]])

    belief_state = update_beliefs_batched(history, ["cat", "dog"], model, deterministic=False, config=config)

    assert model.complete_calls == []
    assert belief_state.hypotheses == ("dog", "cat",)
    assert belief_state.probabilities == pytest.approx([0.9, 0.1])
    assert len(model.probability_calls) == 1
