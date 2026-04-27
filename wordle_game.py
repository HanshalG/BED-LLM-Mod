from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import wandb
except ModuleNotFoundError:
    class _NoOpWandb:
        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _NoOpWandb()

from helpers import BeliefState, Config, format_belief_state, make_belief_state, write_to_log

WORDLE_GREEN = "G"
WORDLE_YELLOW = "Y"
WORDLE_GRAY = "B"
WORDLE_FEEDBACK_SYMBOLS = {WORDLE_GREEN, WORDLE_YELLOW, WORDLE_GRAY}


@dataclass(frozen=True)
class WordleTurn:
    guess: str
    feedback: str


def validate_wordle_word(word: str) -> str:
    normalized = word.strip().lower()
    if len(normalized) != 5 or not normalized.isalpha():
        raise ValueError(f"Wordle words must be five alphabetic letters: {word!r}")
    return normalized


def load_wordle_words(path: str | Path) -> list[str]:
    word_path = Path(path)
    words: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(word_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            word = validate_wordle_word(line)
        except ValueError as exc:
            raise ValueError(f"Invalid Wordle word in {word_path} at line {line_number}: {line!r}") from exc
        if word not in seen:
            seen.add(word)
            words.append(word)

    if not words:
        raise ValueError(f"Wordle word list is empty: {word_path}")
    return words


def validate_wordle_feedback(feedback: str) -> str:
    normalized = feedback.strip().upper()
    if len(normalized) != 5 or any(symbol not in WORDLE_FEEDBACK_SYMBOLS for symbol in normalized):
        raise ValueError(
            f"Wordle feedback must be five symbols from {sorted(WORDLE_FEEDBACK_SYMBOLS)}: {feedback!r}"
        )
    return normalized


def wordle_feedback(guess: str, solution: str) -> str:
    guess = validate_wordle_word(guess)
    solution = validate_wordle_word(solution)
    feedback = [WORDLE_GRAY] * 5
    remaining_solution_letters: Counter[str] = Counter()

    for index, (guess_letter, solution_letter) in enumerate(zip(guess, solution)):
        if guess_letter == solution_letter:
            feedback[index] = WORDLE_GREEN
        else:
            remaining_solution_letters[solution_letter] += 1

    for index, guess_letter in enumerate(guess):
        if feedback[index] == WORDLE_GREEN:
            continue
        if remaining_solution_letters[guess_letter] > 0:
            feedback[index] = WORDLE_YELLOW
            remaining_solution_letters[guess_letter] -= 1

    return "".join(feedback)


def filter_wordle_solutions(solutions: list[str], guess: str, feedback: str) -> list[str]:
    expected_feedback = validate_wordle_feedback(feedback)
    normalized_guess = validate_wordle_word(guess)
    return [
        solution
        for solution in solutions
        if wordle_feedback(normalized_guess, solution) == expected_feedback
    ]


def format_wordle_history(history: list[WordleTurn]) -> str:
    if not history:
        return "[]"
    return "[" + ", ".join(f"{turn.guess}:{turn.feedback}" for turn in history) + "]"


def initialize_wordle_beliefs(solution_words: list[str]) -> BeliefState:
    return make_belief_state(solution_words, fallback_to_uniform=True)


def _entropy(probabilities: list[float]) -> float:
    entropy = 0.0
    for probability in probabilities:
        if probability > 0.0:
            entropy -= probability * float(np.log(probability))
    return entropy


def _feedback_buckets(beliefs: BeliefState, guess: str) -> dict[str, list[tuple[str, float]]]:
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for solution, probability in zip(beliefs.beliefs, beliefs.probabilities):
        buckets[wordle_feedback(guess, solution)].append((solution, probability))
    return dict(buckets)


def score_wordle_guess(beliefs: BeliefState, guess: str, eig: bool) -> float:
    if len(beliefs.beliefs) == 0:
        return 0.0

    buckets = _feedback_buckets(beliefs, guess)
    feedback_probabilities = [
        sum(probability for _solution, probability in bucket)
        for bucket in buckets.values()
    ]
    feedback_entropy = _entropy(feedback_probabilities)
    if not eig:
        return feedback_entropy

    current_entropy = _entropy(list(beliefs.probabilities))
    expected_posterior_entropy = 0.0
    for bucket, bucket_probability in zip(buckets.values(), feedback_probabilities):
        if bucket_probability <= 0.0:
            continue
        normalized_bucket_probabilities = [
            probability / bucket_probability
            for _solution, probability in bucket
        ]
        expected_posterior_entropy += bucket_probability * _entropy(normalized_bucket_probabilities)
    return current_entropy - expected_posterior_entropy


def _posterior_for_feedback(beliefs: BeliefState, guess: str, feedback: str) -> BeliefState:
    filtered_words: list[str] = []
    filtered_probabilities: list[float] = []
    expected_feedback = validate_wordle_feedback(feedback)
    for solution, probability in zip(beliefs.beliefs, beliefs.probabilities):
        if wordle_feedback(guess, solution) == expected_feedback:
            filtered_words.append(solution)
            filtered_probabilities.append(probability)
    return make_belief_state(filtered_words, filtered_probabilities, fallback_to_uniform=True)


def generate_wordle_candidate_guesses(beliefs: BeliefState, allowed_guesses: list[str], config: Config) -> list[str]:
    if len(beliefs.beliefs) == 0:
        return []

    ordered_solutions = [
        word
        for word, _probability in sorted(
            zip(beliefs.beliefs, beliefs.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
    ]
    if len(ordered_solutions) > config.wordle_candidate_pool_size:
        ordered_solutions = ordered_solutions[:config.wordle_candidate_pool_size]

    candidate_words: list[str] = []
    seen: set[str] = set()
    for word in ordered_solutions:
        if word not in seen:
            seen.add(word)
            candidate_words.append(word)

    allowed_remaining = config.wordle_allowed_candidate_pool_size
    for word in allowed_guesses:
        if allowed_remaining <= 0:
            break
        if word in seen:
            continue
        seen.add(word)
        candidate_words.append(word)
        allowed_remaining -= 1

    return candidate_words


def evaluate_wordle_guesses(
    beliefs: BeliefState,
    candidate_guesses: list[str],
    eig: bool,
) -> list[float]:
    return [
        score_wordle_guess(beliefs, guess, eig)
        for guess in candidate_guesses
    ]


def evaluate_wordle_guesses_forward_search(
    beliefs: BeliefState,
    candidate_guesses: list[str],
    allowed_guesses: list[str],
    eig: bool,
    config: Config,
    depth: int = 1,
) -> list[float]:
    if depth == 1:
        return evaluate_wordle_guesses(beliefs, candidate_guesses, eig)
    if depth != 2:
        raise ValueError("evaluate_wordle_guesses_forward_search only supports depth=1 or depth=2")

    immediate_values = evaluate_wordle_guesses(beliefs, candidate_guesses, eig)
    total_values = immediate_values.copy()
    for guess_index, guess in enumerate(candidate_guesses):
        expected_future_value = 0.0
        buckets = _feedback_buckets(beliefs, guess)
        for feedback, bucket in buckets.items():
            branch_probability = sum(probability for _solution, probability in bucket)
            if branch_probability <= 0.0:
                continue
            future_beliefs = _posterior_for_feedback(beliefs, guess, feedback)
            if len(future_beliefs.beliefs) <= 1:
                continue
            future_candidates = generate_wordle_candidate_guesses(future_beliefs, allowed_guesses, config)
            future_values = evaluate_wordle_guesses(future_beliefs, future_candidates, eig)
            if future_values:
                expected_future_value += branch_probability * max(future_values)
        total_values[guess_index] += expected_future_value
    return total_values


def run_wordle_single(
    target_word: str,
    solution_words: list[str],
    allowed_guesses: list[str],
    method_name: str,
    config: Config,
) -> list[int]:
    target_word = validate_wordle_word(target_word)
    beliefs = initialize_wordle_beliefs(solution_words)
    history: list[WordleTurn] = []
    correct_guess = [0] * config.max_wordle_guesses
    eig = method_name == "EIG"

    print(f"[wordle] Starting target {target_word} with {len(beliefs.beliefs)} possible solution(s)")
    write_to_log(f"\n\nStarting on Wordle target {target_word}\n", config)
    write_to_log(f"Initial Wordle beliefs: {format_belief_state(beliefs, top_n=20)}\n", config)

    for round_index in range(config.max_wordle_guesses):
        write_to_log(f"\nWordle target {target_word}: Round {round_index + 1}\n", config)
        print(
            f"[wordle] {target_word}: round {round_index + 1}/{config.max_wordle_guesses} "
            f"with {len(beliefs.beliefs)} possible solution(s)"
        )
        candidate_guesses = generate_wordle_candidate_guesses(beliefs, allowed_guesses, config)
        scores = evaluate_wordle_guesses_forward_search(
            beliefs,
            candidate_guesses,
            allowed_guesses,
            eig=eig,
            config=config,
            depth=config.search_depth,
        )
        best_index = int(np.argmax(scores))
        best_guess = candidate_guesses[best_index]
        best_score = scores[best_index]
        feedback = wordle_feedback(best_guess, target_word)
        history.append(WordleTurn(best_guess, feedback))

        print(f"[wordle] Guess {best_guess} -> {feedback} (score={best_score:.4f})")
        write_to_log(
            f"Best guess: {best_guess} (score={best_score:.4f}), Feedback: {feedback}\n",
            config,
        )
        if best_guess == target_word:
            correct_guess[round_index:] = [1] * (len(correct_guess) - round_index)
            write_to_log(f"Solved Wordle target {target_word} in round {round_index + 1}\n", config)
            return correct_guess

        beliefs = _posterior_for_feedback(beliefs, best_guess, feedback)
        write_to_log(f"Wordle history: {format_wordle_history(history)}\n", config)
        write_to_log(f"Current Wordle beliefs: {format_belief_state(beliefs, top_n=20)}\n", config)
        if len(beliefs.beliefs) == 0:
            print("[wordle] No possible solutions remain after feedback")
            return correct_guess

    return correct_guess


def run_wordle(method_name: str, config: Config) -> list[float]:
    if method_name not in {"Entropy", "EIG"}:
        raise ValueError("Wordle deterministic mode supports method_names: Entropy, EIG")
    if config.wordle_solution_words_path is None or config.wordle_allowed_guesses_path is None:
        raise ValueError("Wordle config requires solution and allowed guess word-list paths")

    solution_words = load_wordle_words(config.wordle_solution_words_path)
    allowed_guesses = load_wordle_words(config.wordle_allowed_guesses_path)
    allowed_guesses = list(dict.fromkeys(allowed_guesses + solution_words))
    target_words = solution_words
    accuracies = [0.0] * config.max_wordle_guesses
    print(f"[wordle] Running method {method_name} across {len(target_words)} target word(s)")

    for target_index, target_word in enumerate(target_words, start=1):
        wandb.log({
            "event": "start wordle",
            "target_word": target_word,
            "method": method_name,
        })
        correct_guess = run_wordle_single(target_word, solution_words, allowed_guesses, method_name, config)
        accuracies = [accuracy + correct for accuracy, correct in zip(accuracies, correct_guess)]
        running_accuracy = [accuracy / target_index for accuracy in accuracies]
        write_to_log(f"Running Wordle accuracy trace: {running_accuracy}\n", config)
        print(f"[wordle] Finished {target_word}. Running accuracy trace: {running_accuracy}")

    return [accuracy / len(target_words) for accuracy in accuracies]
