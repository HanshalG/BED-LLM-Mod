import math
import tempfile
import textwrap
import unittest
from pathlib import Path

from helpers import BeliefState, Config, ModelSpec, build_output_stem, load_config
from wordle_game import (
    evaluate_wordle_guesses,
    filter_wordle_solutions,
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
        ])

        updated = update_wordle_beliefs(beliefs, history, questioner, Config())

        self.assertEqual(updated.beliefs, [])
        self.assertEqual(updated.probabilities, [])

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


if __name__ == "__main__":
    unittest.main()
