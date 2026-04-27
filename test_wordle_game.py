import math
import tempfile
import textwrap
import unittest
from pathlib import Path

from helpers import BeliefState, Config, ModelSpec, build_output_stem, load_config
from wordle_game import (
    evaluate_wordle_guesses,
    evaluate_wordle_guesses_forward_search,
    filter_wordle_solutions,
    format_wordle_constraint_summary,
    generate_wordle_naive_guess,
    generate_wordle_opening_beliefs,
    generate_wordle_candidate_guesses_from_llm,
    run_wordle_single,
    score_wordle_guess,
    update_wordle_beliefs,
    validate_wordle_word,
    wordle_feedback,
)


class DummyQuestioner:
    def __init__(self, completions: list[str]):
        self.completions = list(completions)
        self.calls = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append({"messages": messages, "temperature": temperature, "num_responses": num_responses})
        if not self.completions:
            raise AssertionError("No more completions configured")
        return [self.completions.pop(0)]


class WordleUtilityTests(unittest.TestCase):
    def test_feedback_handles_duplicate_letters(self) -> None:
        self.assertEqual(wordle_feedback("allee", "apple"), "GYBBG")
        self.assertEqual(wordle_feedback("sissy", "cigar"), "BGBBB")

    def test_filter_wordle_solutions_keeps_exact_feedback_matches(self) -> None:
        solutions = ["cigar", "rebut", "sissy", "humph"]
        feedback = wordle_feedback("cigar", "sissy")

        self.assertEqual(filter_wordle_solutions(solutions, "cigar", feedback), ["sissy"])

    def test_score_wordle_guess_matches_exact_entropy_math(self) -> None:
        beliefs = BeliefState(["cigar", "rebut"], [0.5, 0.5])

        self.assertAlmostEqual(score_wordle_guess(beliefs, "cigar", eig=True), math.log(2), places=6)
        self.assertAlmostEqual(score_wordle_guess(beliefs, "cigar", eig=False), math.log(2), places=6)

    def test_depth_three_wordle_search_recurses_over_future_branches(self) -> None:
        beliefs = BeliefState(["rebut", "civet", "event", "tenet", "deter"], [0.2] * 5)
        questioner = DummyQuestioner(["cigar\nrebut\n"] * 100)

        values = evaluate_wordle_guesses_forward_search(
            beliefs,
            ["slate"],
            [],
            questioner,
            eig=True,
            config=Config(target_num_questions=2, min_num_samples=1),
            depth=3,
        )

        self.assertEqual(len(values), 1)
        self.assertTrue(all(value >= 0 for value in values))
        self.assertGreater(len(questioner.calls), 0)

    def test_depth_search_regenerates_beliefs_for_hypothetical_feedback(self) -> None:
        beliefs = BeliefState(["cigar", "rebut"], [0.5, 0.5])
        questioner = DummyQuestioner([
            "proud\nworld\n",
            "rebut\nproud\n",
        ])

        values = evaluate_wordle_guesses_forward_search(
            beliefs,
            ["cigar"],
            [],
            questioner,
            eig=True,
            config=Config(target_num_questions=2, min_num_samples=2),
            depth=2,
        )

        self.assertEqual(len(values), 1)
        self.assertGreater(values[0], math.log(2))
        self.assertEqual(len(questioner.calls), 2)
        candidate_prompt = questioner.calls[1]["messages"][-1]["content"]
        self.assertIn("rebut: 0.333", candidate_prompt)
        self.assertIn("proud: 0.333", candidate_prompt)
        self.assertIn("world: 0.333", candidate_prompt)

    def test_generate_and_score_selects_best_guess_from_tiny_lexicon(self) -> None:
        beliefs = BeliefState(["cigar", "rebut", "humph"], [1 / 3, 1 / 3, 1 / 3])
        config = Config(target_num_questions=2)
        questioner = DummyQuestioner(["cigar\nrebut\n"])
        candidates = generate_wordle_candidate_guesses_from_llm(beliefs, [], questioner, config)
        scores = evaluate_wordle_guesses(beliefs, candidates, eig=True)

        self.assertEqual(candidates, ["cigar", "rebut"])
        self.assertIn(candidates[scores.index(max(scores))], candidates)

    def test_full_mini_game_solves_with_deterministic_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                max_wordle_guesses=3,
                search_depth=1,
                min_num_samples=2,
                log_path=Path(tmp_dir) / "wordle.log",
            )
            questioner = DummyQuestioner([
                "cigar\nrebut\n",
                "cigar\n",
                "rebut\n",
                "rebut\n",
            ])
            trace = run_wordle_single(
                "rebut",
                questioner,
                "EIG",
                config,
            )

        self.assertEqual(trace, [0, 1, 1])

    def test_naive_wordle_uses_first_remaining_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                max_wordle_guesses=3,
                min_num_samples=1,
                log_path=Path(tmp_dir) / "wordle.log",
            )
            questioner = DummyQuestioner([
                "cigar\nrebut\nhumph\n",
                "cigar\n",
                "humph\n",
                "humph\n",
            ])
            trace = run_wordle_single(
                "humph",
                questioner,
                "naive",
                config,
            )

        self.assertEqual(trace, [0, 1, 1])

    def test_update_wordle_beliefs_never_falls_back_to_incompatible_words(self) -> None:
        beliefs = BeliefState(["urban"], [1.0])
        history = [
            type("Turn", (), {"guess": "stare", "feedback": "BBYYB"})(),
            type("Turn", (), {"guess": "urban", "feedback": "BYBGB"})(),
        ]
        questioner = DummyQuestioner([
            "board\nbravo\nbrush\n",
            "dream\nlaser\nlayer\n",
            "curly\nscour\ncoder\n",
            "board\nbravo\nbrush\n",
            "dream\nlaser\nlayer\n",
            "curly\nscour\ncoder\n",
        ])

        updated = update_wordle_beliefs(beliefs, history, questioner, Config())

        self.assertEqual(updated.beliefs, [])
        self.assertEqual(updated.probabilities, [])

    def test_update_wordle_beliefs_retries_until_minimum_compatible_words(self) -> None:
        beliefs = BeliefState(["cigar"], [1.0])
        history = [type("Turn", (), {"guess": "cigar", "feedback": "GGGGG"})()]
        questioner = DummyQuestioner([
            "rebut\n",
            "cigar\n",
        ])

        updated = update_wordle_beliefs(beliefs, history, questioner, Config(min_num_samples=2))

        self.assertEqual(updated.beliefs, ["cigar"])
        self.assertEqual(len(questioner.calls), 2)

    def test_update_wordle_beliefs_accumulates_compatible_words_across_retries(self) -> None:
        beliefs = BeliefState([], [])
        history = [type("Turn", (), {"guess": "fuzzy", "feedback": "BBBBB"})()]
        questioner = DummyQuestioner([
            "cigar\n",
            "slate\ncrane\n",
        ])

        updated = update_wordle_beliefs(beliefs, history, questioner, Config(min_num_samples=3, max_num_samples=10))

        self.assertEqual(updated.beliefs, ["cigar", "slate", "crane"])
        self.assertEqual(len(questioner.calls), 2)

    def test_wordle_constraint_summary_includes_pattern_required_and_absent_letters(self) -> None:
        history = [
            type("Turn", (), {"guess": "stare", "feedback": "BYBBB"})(),
            type("Turn", (), {"guess": "night", "feedback": "YBBYY"})(),
        ]

        summary = format_wordle_constraint_summary(history)

        self.assertIn("Pattern: _ _ _ _ _", summary)
        self.assertIn("Required letters:", summary)
        self.assertIn("t at least 1", summary)
        self.assertIn("h at least 1", summary)
        self.assertIn("Absent letters:", summary)
        self.assertIn("s", summary)
        self.assertIn("Forbidden positions:", summary)

    def test_update_wordle_beliefs_truncates_after_filtering(self) -> None:
        beliefs = BeliefState([], [])
        history = [type("Turn", (), {"guess": "fuzzy", "feedback": "BBBBB"})()]
        questioner = DummyQuestioner(["cigar\nslate\ncrane\n"])

        updated = update_wordle_beliefs(beliefs, history, questioner, Config(min_num_samples=1, max_num_samples=2))

        self.assertEqual(updated.beliefs, ["cigar", "slate"])

    def test_wordle_valid_words_filter_applies_to_opening_beliefs(self) -> None:
        valid_words_path = str(Path("wordle_test_fixtures/valid-wordle-words.txt").resolve())
        questioner = DummyQuestioner(["abcde\ncigar\ncrowr\nrebut\n"])

        beliefs = generate_wordle_opening_beliefs(
            questioner,
            Config(min_num_samples=1, max_num_samples=10, wordle_valid_words_path=valid_words_path),
        )

        self.assertEqual(beliefs, ["cigar", "rebut"])

    def test_wordle_valid_words_filter_applies_to_updated_beliefs(self) -> None:
        valid_words_path = str(Path("wordle_test_fixtures/valid-wordle-words.txt").resolve())
        beliefs = BeliefState([], [])
        history = [type("Turn", (), {"guess": "fuzzy", "feedback": "BBBBB"})()]
        questioner = DummyQuestioner(["abcde\ncigar\n"])

        updated = update_wordle_beliefs(
            beliefs,
            history,
            questioner,
            Config(min_num_samples=1, max_num_samples=10, wordle_valid_words_path=valid_words_path),
        )

        self.assertEqual(updated.beliefs, ["cigar"])

    def test_wordle_valid_words_filter_applies_to_candidate_and_naive_guesses(self) -> None:
        valid_words_path = str(Path("wordle_test_fixtures/valid-wordle-words.txt").resolve())
        config = Config(target_num_questions=5, wordle_valid_words_path=valid_words_path)
        beliefs = BeliefState(["cigar", "rebut", "humph"], [1 / 3, 1 / 3, 1 / 3])
        questioner = DummyQuestioner([
            "abcde\ncigar\ncrowr\nrebut\n",
            "mkept\nslate\n",
        ])

        candidates = generate_wordle_candidate_guesses_from_llm(beliefs, [], questioner, config)
        naive_guess = generate_wordle_naive_guess([], questioner, config)

        self.assertEqual(candidates, ["cigar", "rebut"])
        self.assertEqual(naive_guess, "slate")

    def test_validate_wordle_word_rejects_invalid_words(self) -> None:
        with self.assertRaisesRegex(ValueError, "five alphabetic letters"):
            validate_wordle_word("toolong")
        with self.assertRaisesRegex(ValueError, "five alphabetic letters"):
            validate_wordle_word("ab3de")


class WordleConfigTests(unittest.TestCase):
    def test_load_config_accepts_relative_wordle_fixture_paths(self) -> None:
        config = load_config("wordle_test_fixtures/wordle_config.yaml")

        self.assertEqual(config.game, "wordle")
        self.assertTrue(Path(config.wordle_solution_words_path or "").is_file())
        self.assertTrue(Path(config.wordle_valid_words_path or "").is_file())

    def test_wordle_output_stem_uses_wordle_suffix(self) -> None:
        spec = ModelSpec(model="google/gemma-4-E4B-it", thinking=False)

        output_stem = build_output_stem("run", "EIG", spec, spec, 0, game="wordle")

        self.assertTrue(output_stem.endswith("_0_wordle"))

    def test_load_config_rejects_missing_wordle_word_list(self) -> None:
        config_text = textwrap.dedent(
            """
            game: "wordle"
            wordle_solution_words_path: "missing-solutions.txt"
            method_names: ["EIG"]
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "wordle_solution_words_path does not exist"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_wordle_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            word_list = tmp_path / "words.txt"
            word_list.write_text("cigar\n", encoding="utf-8")
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    game: "wordle"
                    wordle_solution_words_path: "{word_list}"
                    method_names: ["EIG"]
                    max_wordle_guesses: 0
                    """
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_wordle_guesses must be at least 1"):
                load_config(str(config_path))

    def test_load_config_rejects_missing_wordle_valid_words_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            word_list = tmp_path / "words.txt"
            word_list.write_text("cigar\n", encoding="utf-8")
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    game: "wordle"
                    wordle_solution_words_path: "{word_list}"
                    wordle_valid_words_path: "missing-valid-words.txt"
                    method_names: ["EIG"]
                    """
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "wordle_valid_words_path does not exist"):
                load_config(str(config_path))

    def test_load_config_accepts_wordle_search_depth_above_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            word_list = tmp_path / "words.txt"
            word_list.write_text("cigar\n", encoding="utf-8")
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    game: "wordle"
                    wordle_solution_words_path: "{word_list}"
                    model_pairs: []
                    method_names: ["EIG"]
                    search_depth: 3
                    """
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

        self.assertEqual(config.search_depth, 3)


if __name__ == "__main__":
    unittest.main()
