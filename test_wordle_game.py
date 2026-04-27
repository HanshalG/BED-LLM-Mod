import math
import tempfile
import textwrap
import unittest
from pathlib import Path

from helpers import BeliefState, Config, ModelSpec, build_output_stem, load_config
from wordle_game import (
    evaluate_wordle_guesses,
    filter_wordle_solutions,
    generate_wordle_candidate_guesses,
    run_wordle_single,
    score_wordle_guess,
    validate_wordle_word,
    wordle_feedback,
)


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
        beliefs = BeliefState(["cigar", "rebut"], [0.5, 0.5])
        config = Config(wordle_candidate_pool_size=2, wordle_allowed_candidate_pool_size=0)
        candidates = generate_wordle_candidate_guesses(beliefs, ["arise", "slate", "cigar", "rebut"], config)
        scores = evaluate_wordle_guesses(beliefs, candidates, eig=True)

        self.assertEqual(candidates, ["cigar", "rebut"])
        self.assertEqual(candidates[scores.index(max(scores))], "cigar")

    def test_full_mini_game_solves_with_deterministic_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(
                max_wordle_guesses=3,
                wordle_candidate_pool_size=10,
                wordle_allowed_candidate_pool_size=0,
                search_depth=1,
                log_path=Path(tmp_dir) / "wordle.log",
            )
            trace = run_wordle_single(
                "rebut",
                ["cigar", "rebut"],
                ["cigar", "rebut"],
                "EIG",
                config,
            )

        self.assertEqual(trace, [0, 1, 1])

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
        self.assertTrue(Path(config.wordle_allowed_guesses_path or "").is_file())

    def test_wordle_output_stem_uses_wordle_suffix(self) -> None:
        spec = ModelSpec(model="deterministic-wordle")

        output_stem = build_output_stem("run", "EIG", spec, spec, 0, game="wordle")

        self.assertTrue(output_stem.endswith("_0_wordle"))

    def test_load_config_rejects_missing_wordle_word_list(self) -> None:
        config_text = textwrap.dedent(
            """
            game: "wordle"
            wordle_solution_words_path: "missing-solutions.txt"
            wordle_allowed_guesses_path: "missing-allowed.txt"
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
                    wordle_allowed_guesses_path: "{word_list}"
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
