import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch


if "model" not in sys.modules:
    fake_model_module = types.ModuleType("model")

    class Model:
        pass

    fake_model_module.Model = Model
    sys.modules["model"] = fake_model_module

if "wandb" not in sys.modules:
    fake_wandb_module = types.ModuleType("wandb")
    fake_wandb_module.init = lambda *args, **kwargs: None
    fake_wandb_module.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = fake_wandb_module

import generate_candidate_questions as gcq
import questions_game
import update_beliefs as ub
from helpers import BeliefState, Config, ModelSpec, build_output_stem, format_config_for_log, load_config


class DummyQuestioner:
    def chat_complete(self, messages, temperature, num_responses=1):
        raise AssertionError("chat_complete should not be used in this test")

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        raise AssertionError("chat_complete_messages_batched should not be used in this test")

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used in this test")


class BatchedQuestioner(DummyQuestioner):
    def __init__(self, completions=None, probabilities=None):
        self.completions = list(completions or [])
        self.probabilities = list(probabilities or [])
        self.complete_calls = []
        self.batched_complete_calls = []
        self.probability_calls = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.complete_calls.append(
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
        self.batched_complete_calls.append(
            {
                "batch_messages": batch_messages,
                "temperature": temperature,
                "block_size": block_size,
                "max_new_tokens": max_new_tokens,
            }
        )
        if not self.completions:
            raise AssertionError("No more batched completions configured")
        return self.completions.pop(0)

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        self.probability_calls.append(
            {
                "messages": messages,
                "responses": responses,
                "temperature": temperature,
                "block_size": block_size,
            }
        )
        if not self.probabilities:
            raise AssertionError("No more probabilities configured")
        return self.probabilities.pop(0)


class DepthSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            answer_temperature=0.3,
            batched_block_size=4,
            generation_temperature_diverse=0.9,
            target_num_questions=3,
            num_mc_samples=5,
            threshold_rejection_probability=0.2,
            search_depth=2,
        )
        self.beliefs = BeliefState(["Otter", "Fox"], [0.6, 0.4])
        self.history = [{"role": "assistant", "content": "Does it swim?"}, {"role": "user", "content": "Yes"}]
        self.questioner = DummyQuestioner()

    def test_depth_1_uses_batched_scoring_path(self) -> None:
        with patch.object(gcq, "evaluate_questions_batched", return_value=[0.1, 0.9]) as mock_batched:
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it furry?", "Does it live in water?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=1,
            )

        self.assertEqual(values, [0.1, 0.9])
        mock_batched.assert_called_once()

    def test_depth_2_uses_full_branch_updates_and_expected_future_value(self) -> None:
        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter", "Fox"], [0.5, 0.5])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([1.0], [0.25], [0.75])),
            patch.object(
                gcq,
                "_update_beliefs_many",
                return_value=[BeliefState(["Seal"], [1.0]), BeliefState(["Wolf"], [1.0])],
            ) as mock_update_many,
            patch.object(gcq, "_generate_future_candidate_questions_batched", return_value=[["Ask about flippers?"], ["Ask about howling?"]]),
            patch.object(gcq, "_score_future_questions_batched", return_value=[[2.0], [4.0]]),
            patch.object(gcq, "write_to_log"),
        ):
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=2,
            )

        self.assertEqual(values, [4.5])
        self.assertEqual(mock_update_many.call_count, 1)
        self.assertEqual(
            mock_update_many.call_args.args[0],
            [
                self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "Yes"}],
                self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "No"}],
            ],
        )

    def test_depth_2_empty_future_beliefs_contribute_zero(self) -> None:
        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter"], [1.0])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([0.7], [0.4], [0.6])),
            patch.object(gcq, "_update_beliefs_many", return_value=[BeliefState([], [])]),
            patch.object(gcq, "write_to_log"),
        ):
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=2,
            )

        self.assertEqual(values, [0.7])

    def test_depth_2_empty_future_questions_contribute_zero(self) -> None:
        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter"], [1.0])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([0.8], [0.5], [0.5])),
            patch.object(gcq, "_update_beliefs_many", return_value=[BeliefState(["Seal"], [1.0])]),
            patch.object(gcq, "_generate_future_candidate_questions_batched", return_value=[[]]),
            patch.object(gcq, "write_to_log"),
        ):
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=2,
            )

        self.assertEqual(values, [0.8])

    def test_depth_2_empty_future_scores_contribute_zero(self) -> None:
        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter"], [1.0])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([0.9], [0.5], [0.5])),
            patch.object(gcq, "_update_beliefs_many", return_value=[BeliefState(["Seal"], [1.0])]),
            patch.object(gcq, "_generate_future_candidate_questions_batched", return_value=[["Ask about fins?"]]),
            patch.object(gcq, "_score_future_questions_batched", return_value=[[]]),
            patch.object(gcq, "write_to_log"),
        ):
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=2,
            )

        self.assertEqual(values, [0.9])

    def test_invalid_depth_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports depth=1 or depth=2"):
            gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=3,
            )

    def test_depth_2_accepts_weighted_future_belief_states(self) -> None:
        weighted_future_states = [
            BeliefState(["Seal", "Otter"], [0.8, 0.2]),
            BeliefState(["Wolf", "Dog"], [0.3, 0.7]),
        ]

        generated_probabilities: list[list[float]] = []
        scored_probabilities: list[list[float]] = []

        def fake_generate_candidate_questions(beliefs, history, questioner, generation_temperature, num_questions):
            generated_probabilities.append(list(beliefs.probabilities))
            return ["Follow-up?"]

        def fake_evaluate_questions_batched(beliefs, cand_questions, eig, deterministic, questioner, answer_temperature, num_mc_samples, block_size):
            scored_probabilities.append(list(beliefs.probabilities))
            return [1.0]

        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter"], [1.0])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([0.4], [0.5], [0.5])),
            patch.object(gcq, "_update_beliefs_many", return_value=weighted_future_states),
            patch.object(gcq, "_generate_future_candidate_questions_batched", side_effect=lambda beliefs, histories, questioner, generation_temperature, num_questions, block_size: [fake_generate_candidate_questions(belief, history, questioner, generation_temperature, num_questions) for belief, history in zip(beliefs, histories)]),
            patch.object(gcq, "_score_future_questions_batched", side_effect=lambda beliefs, future_questions, eig, deterministic, questioner, answer_temperature, num_mc_samples, block_size: [fake_evaluate_questions_batched(belief, questions, eig, deterministic, questioner, answer_temperature, num_mc_samples, block_size) for belief, questions in zip(beliefs, future_questions)]),
            patch.object(gcq, "write_to_log"),
        ):
            categorical_config = Config(
                answer_temperature=0.3,
                batched_block_size=4,
                generation_temperature_diverse=0.9,
                target_num_questions=3,
                num_mc_samples=5,
                threshold_rejection_probability=0.2,
                belief_state_mode="categorical",
                search_depth=2,
            )
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=categorical_config,
                depth=2,
            )

        self.assertEqual(values, [1.4])
        self.assertEqual(generated_probabilities, [[0.8, 0.2], [0.3, 0.7]])
        self.assertEqual(scored_probabilities, [[0.8, 0.2], [0.3, 0.7]])

    def test_generate_future_candidate_questions_batched_preserves_branch_order_and_modes(self) -> None:
        model = BatchedQuestioner(
            completions=[
                ["Q1a\nQ1b", "Q2a"],
                ["Q2b"],
            ]
        )
        branch_beliefs = [
            BeliefState([], []),
            BeliefState(["Seal"], [1.0]),
            BeliefState(["Otter", "Seal", "Whale"], [1 / 3, 1 / 3, 1 / 3]),
            BeliefState(["Wolf", "Dog", "Fox"], [0.7, 0.2, 0.1]),
        ]
        branch_histories = [
            [],
            [{"role": "assistant", "content": "Q?"}, {"role": "user", "content": "A"}],
            [{"role": "assistant", "content": "Q1"}, {"role": "user", "content": "A1"}],
            [{"role": "assistant", "content": "Q2"}, {"role": "user", "content": "A2"}],
        ]

        future_questions = gcq._generate_future_candidate_questions_batched(
            branch_beliefs,
            branch_histories,
            model,
            generation_temperature=0.5,
            num_questions=2,
            block_size=4,
        )

        self.assertEqual(
            future_questions,
            [
                [],
                ["Is it Seal?"],
                ["Q1a", "Q1b"],
                ["Q2a", "Q2b"],
            ],
        )
        self.assertEqual(len(model.batched_complete_calls), 2)

    def test_score_future_questions_batched_matches_per_branch_values(self) -> None:
        model = BatchedQuestioner(
            probabilities=[
                [
                    {"Yes": 1.0, "No": 0.0},
                    {"Yes": 0.0, "No": 1.0},
                    {"Yes": 0.2, "No": 0.8},
                    {"Yes": 0.8, "No": 0.2},
                    {"Yes": 0.8, "No": 0.2},
                    {"Yes": 0.2, "No": 0.8},
                ]
            ]
        )
        branch_beliefs = [
            BeliefState(["Cat", "Dog"], [0.5, 0.5]),
            BeliefState(["Otter", "Seal"], [0.75, 0.25]),
        ]
        branch_future_questions = [
            ["Is it feline?"],
            ["Does it swim?", "Does it bark?"],
        ]

        scores = gcq._score_future_questions_batched(
            branch_beliefs,
            branch_future_questions,
            eig=True,
            deterministic=True,
            questioner=model,
            answer_temperature=0.3,
            num_mc_samples=5,
            block_size=8,
        )

        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0][0], 0.6931471805599453, places=6)
        self.assertEqual(len(scores[1]), 2)

    def test_update_beliefs_many_matches_sequential_uniform_path(self) -> None:
        histories = [
            self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "Yes"}],
            self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "No"}],
        ]
        config = Config(
            belief_state_mode="uniform",
            generation_temperature_diverse=0.9,
            max_num_samples=5,
            min_num_samples=2,
            batched_block_size=4,
            threshold_rejection_probability=0.2,
        )
        with (
            patch.object(ub, "_generate_new_beliefs_many", return_value=[["Seal", "Otter"], ["Wolf"]]),
            patch.object(ub, "_filter_valid_animal_names_many", side_effect=lambda beliefs, checker, block_size: beliefs),
            patch.object(ub, "_check_beliefs_many", side_effect=[
                [["Seal"], ["Wolf"]],
                [["Otter"], ["Fox"]],
            ]),
        ):
            states = ub._update_beliefs_many(histories, self.beliefs, self.questioner, False, config)

        self.assertEqual([state.beliefs for state in states], [["Seal", "Otter"], ["Wolf", "Fox"]])
        self.assertEqual(states[0].probabilities, [0.5, 0.5])

    def test_update_beliefs_many_matches_sequential_categorical_path(self) -> None:
        histories = [
            self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "Yes"}],
            self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "No"}],
        ]
        categorical_config = Config(
            belief_state_mode="categorical",
            generation_temperature_diverse=0.9,
            max_num_samples=5,
            min_num_samples=2,
            batched_block_size=4,
            threshold_rejection_probability=0.2,
        )
        scored_states = [
            BeliefState(["Seal", "Otter"], [0.8, 0.2]),
            BeliefState(["Fox", "Wolf"], [0.6, 0.4]),
        ]
        with (
            patch.object(ub, "_generate_new_beliefs_many", return_value=[["Seal"], ["Wolf"]]),
            patch.object(ub, "_filter_valid_animal_names_many", side_effect=lambda beliefs, checker, block_size: beliefs),
            patch.object(ub, "_check_beliefs_many", side_effect=[
                [["Seal"], ["Wolf"]],
                [["Otter"], ["Fox"]],
            ]),
            patch.object(ub, "_build_belief_states_many", return_value=scored_states),
        ):
            states = ub._update_beliefs_many(histories, self.beliefs, self.questioner, False, categorical_config)

        self.assertEqual(states, scored_states)

    def test_update_beliefs_many_skips_retry_cross_branch_leakage(self) -> None:
        histories = [
            self.history + [{"role": "assistant", "content": "Is it aquatic?"}, {"role": "user", "content": "Yes"}],
            self.history + [{"role": "assistant", "content": "Is it nocturnal?"}, {"role": "user", "content": "No"}],
        ]
        config = Config(
            belief_state_mode="uniform",
            generation_temperature_diverse=0.9,
            max_num_samples=5,
            min_num_samples=2,
            batched_block_size=4,
            threshold_rejection_probability=0.2,
        )
        with (
            patch.object(ub, "_generate_new_beliefs_many", side_effect=[
                [["Seal"], ["Fox", "Wolf"]],
                [["Otter"]],
            ]),
            patch.object(ub, "_filter_valid_animal_names_many", side_effect=lambda beliefs, checker, block_size: beliefs),
            patch.object(ub, "_check_beliefs_many", side_effect=[
                [["Seal"], ["Fox", "Wolf"]],
                [["Otter"], ["Fox"]],
                [["Otter"]],
            ]),
        ):
            states = ub._update_beliefs_many(histories, self.beliefs, self.questioner, False, config)

        self.assertEqual([state.beliefs for state in states], [["Seal", "Otter"], ["Fox", "Wolf"]])

    def test_depth_2_zero_probability_branch_skip_does_not_change_total(self) -> None:
        with (
            patch.object(gcq, "_draw_belief_samples", return_value=(["Otter"], [1.0])),
            patch.object(gcq, "_score_questions_from_samples", return_value=([0.3], [1.0], [0.0])),
            patch.object(gcq, "_update_beliefs_many", return_value=[BeliefState(["Seal"], [1.0])]) as mock_update_many,
            patch.object(gcq, "_generate_future_candidate_questions_batched", return_value=[["Ask about fins?"]]),
            patch.object(gcq, "_score_future_questions_batched", return_value=[[2.0]]),
            patch.object(gcq, "write_to_log"),
        ):
            values = gcq.evaluate_questions_forward_search(
                self.beliefs,
                self.history,
                ["Is it aquatic?"],
                eig=True,
                deterministic=False,
                questioner=self.questioner,
                config=self.config,
                depth=2,
            )

        self.assertEqual(values, [2.3])
        self.assertEqual(len(mock_update_many.call_args.args[0]), 1)


class ConfigAndWiringTests(unittest.TestCase):
    def test_load_config_accepts_search_depth_1_and_2(self) -> None:
        base_yaml = textwrap.dedent(
            """
            version: 0
            animals:
              - ["Otter"]
            model_pairs: []
            method_names: ["EIG"]
            search_depth: {search_depth}
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            for search_depth in (1, 2):
                config_path = Path(tmp_dir) / f"config-{search_depth}.yaml"
                config_path.write_text(base_yaml.format(search_depth=search_depth), encoding="utf-8")
                loaded = load_config(str(config_path))
                self.assertEqual(loaded.search_depth, search_depth)

    def test_load_config_rejects_invalid_search_depth(self) -> None:
        config_text = textwrap.dedent(
            """
            version: 0
            animals:
              - ["Otter"]
            model_pairs: []
            method_names: ["EIG"]
            search_depth: 3
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "search_depth must be one of: 1, 2"):
                load_config(str(config_path))

    def test_search_depth_is_in_output_stem_and_config_log(self) -> None:
        questioner = ModelSpec(model="Qwen/Qwen2.5-4B")
        answerer = ModelSpec(model="Qwen/Qwen2.5-4B")
        output_stem = build_output_stem(
            run_id="run123",
            method_name="EIG",
            questioner=questioner,
            answerer=answerer,
            version=0,
            belief_state_mode="uniform",
            search_depth=2,
        )

        formatted = format_config_for_log(Config(search_depth=2))

        self.assertIn("depth-2", output_stem)
        self.assertIn('"search_depth": 2', formatted)

    def test_game_uses_configured_search_depth(self) -> None:
        config = Config(search_depth=2, log_path=Path(tempfile.gettempdir()) / "depth-search-test.log")
        initial_beliefs = BeliefState(["Otter", "Fox"], [0.5, 0.5])

        with (
            patch.object(questions_game, "generate_original_beliefs", return_value=["Otter", "Fox"]),
            patch.object(questions_game, "initialize_belief_state", return_value=initial_beliefs),
            patch.object(questions_game, "generate_candidate_questions", return_value=["Q1", "Q2"]),
            patch.object(questions_game, "evaluate_questions_forward_search", return_value=[0.2, 0.8]) as mock_search,
            patch.object(questions_game, "get_question_answered", return_value="Correct!"),
            patch.object(questions_game, "write_to_log"),
        ):
            result = questions_game.twenty_questions_animals_single_complex(
                goal_animal="Otter",
                eig=True,
                deterministic=False,
                questioner=DummyQuestioner(),
                answerer=DummyQuestioner(),
                config=config,
            )

        self.assertEqual(result[0], 1)
        self.assertEqual(mock_search.call_args.kwargs["depth"], 2)


if __name__ == "__main__":
    unittest.main()
