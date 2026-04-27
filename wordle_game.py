from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import wandb
except ModuleNotFoundError:
    class _NoOpWandb:
        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _NoOpWandb()

from helpers import BeliefState, Config, convert_string_to_array, format_belief_state, make_belief_state, write_to_log

if TYPE_CHECKING:
    from model import Model

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


def clean_wordle_words(raw_words: list[str]) -> list[str]:
    cleaned_words: list[str] = []
    seen: set[str] = set()
    for raw_word in raw_words:
        try:
            word = validate_wordle_word(raw_word)
        except ValueError:
            continue
        if word not in seen:
            seen.add(word)
            cleaned_words.append(word)
    return cleaned_words


def _wordle_system_prompt() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are an expert Wordle solver. The hidden answer is a common lowercase five-letter English word. "
            "Feedback uses exactly five symbols: G means the letter is correct in that position, "
            "Y means the letter is in the answer but in a different position, and B means gray/black. "
            "Apply normal Wordle duplicate-letter rules: green letters are assigned first, then yellows only for "
            "remaining unmatched copies of that letter. Every candidate answer you output must be consistent with "
            "every prior guess and feedback pattern. Return only lowercase five-letter alphabetic words, one per "
            "line, with no numbering, punctuation, explanation, markdown, or extra text."
        ),
    }


def _format_turns_for_prompt(history: list[WordleTurn]) -> str:
    if not history:
        return "No guesses yet."
    return "\n".join(
        f"Guess: {turn.guess} Feedback: {turn.feedback}"
        for turn in history
    )


def _generate_wordle_words_from_llm(
    prompt: str,
    questioner: "Model",
    generation_temperature: float,
) -> list[str]:
    completion = questioner.chat_complete(
        messages=[_wordle_system_prompt(), {"role": "user", "content": prompt}],
        temperature=generation_temperature,
    )[0]
    return clean_wordle_words(convert_string_to_array(completion))


def _wordle_feedback_rules_text() -> str:
    return (
        "Interpret feedback strictly:\n"
        "- G: this exact position is fixed to that letter.\n"
        "- Y: this letter appears in the answer, but not in that guessed position.\n"
        "- B: this guessed letter has no remaining unmatched copy in the answer after greens/yellows are assigned.\n"
        "- Re-check duplicate letters carefully before returning any word.\n"
        "Before finalizing each output word, mentally compare it against every guess and ensure it would produce "
        "exactly the listed feedback."
    )


def generate_wordle_opening_beliefs(questioner: "Model", config: Config) -> list[str]:
    prompt = (
        "Generate plausible hidden answer candidates for a new Wordle game with no guesses yet.\n\n"
        "Use common Wordle-style answer words, not obscure abbreviations, proper nouns, plurals ending in s, "
        "or random letter strings.\n"
        f"Generate up to {config.max_num_samples} distinct words, aiming for at least {config.min_num_samples}. "
        "Vary the letters and word shapes so the belief set is useful for search.\n"
        "Return only one lowercase five-letter word per line."
    )
    return _generate_wordle_words_from_llm(prompt, questioner, config.generation_temperature_diverse)


def generate_wordle_beliefs(
    history: list[WordleTurn],
    questioner: "Model",
    config: Config,
    current_beliefs: list[str] | None = None,
) -> list[str]:
    current_context = ""
    if current_beliefs:
        current_context = f"\nCurrent candidate words to consider or improve on: {current_beliefs[:50]}"
    prompt = (
        "Using the Wordle feedback history below, generate possible hidden answer words.\n\n"
        f"{_wordle_feedback_rules_text()}\n\n"
        f"{_format_turns_for_prompt(history)}"
        f"{current_context}\n\n"
        "Only include words that would produce exactly the shown feedback for every prior guess. "
        "Do not include a previously guessed word unless it is still logically possible. "
        "Avoid obscure words, proper nouns, non-words, and random strings.\n"
        f"Generate up to {config.max_num_samples} distinct candidates, aiming for at least {config.min_num_samples}. "
        "Return only one lowercase five-letter word per line."
    )
    return _generate_wordle_words_from_llm(prompt, questioner, config.generation_temperature_diverse)


def generate_wordle_candidate_guesses_from_llm(
    beliefs: BeliefState,
    history: list[WordleTurn],
    questioner: "Model",
    config: Config,
) -> list[str]:
    if len(beliefs.beliefs) <= 2 and len(beliefs.beliefs) > 0:
        return [beliefs.beliefs[int(np.argmax(beliefs.probabilities))]]

    weighted_beliefs = ", ".join(
        f"{word}: {probability:.3f}"
        for word, probability in zip(beliefs.beliefs, beliefs.probabilities)
    )
    prompt = (
        "Using this Wordle feedback history and current belief state, propose strong next Wordle guesses. "
        "A strong guess should either be a likely answer or an exploratory word that separates the remaining "
        "beliefs into informative feedback groups.\n\n"
        f"{_wordle_feedback_rules_text()}\n\n"
        f"History:\n{_format_turns_for_prompt(history)}\n\n"
        f"Beliefs with probabilities: {weighted_beliefs}\n\n"
        "Prefer common valid five-letter English words. Do not output impossible candidate answers unless the word "
        "is intentionally useful as an exploratory guess. Avoid repeats from the guess history.\n"
        f"Generate up to {config.target_num_questions} distinct candidate guesses. "
        "Return only one lowercase five-letter word per line."
    )
    guesses = _generate_wordle_words_from_llm(prompt, questioner, config.generation_temperature_diverse)
    if len(guesses) == 0 and len(beliefs.beliefs) > 0:
        return [beliefs.beliefs[int(np.argmax(beliefs.probabilities))]]
    return guesses[:config.target_num_questions]


def generate_wordle_naive_guess(history: list[WordleTurn], questioner: "Model", config: Config) -> str:
    prompt = (
        "Using the Wordle feedback history below, generate your single best next Wordle guess.\n\n"
        f"{_wordle_feedback_rules_text()}\n\n"
        f"{_format_turns_for_prompt(history)}\n\n"
        "Choose a common valid five-letter English word. Prefer a word that could be the answer when possible; "
        "otherwise choose an exploratory word that tests useful remaining letters. Avoid repeating previous guesses. "
        "Return exactly one lowercase five-letter word and nothing else."
    )
    guesses = _generate_wordle_words_from_llm(prompt, questioner, config.generation_temperature_simple)
    if not guesses:
        raise ValueError("Wordle naive guess generation produced no valid five-letter word")
    return guesses[0]


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
    history: list[WordleTurn],
    questioner: "Model",
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
            future_history = history + [WordleTurn(guess, feedback)]
            future_candidates = generate_wordle_candidate_guesses_from_llm(
                future_beliefs,
                future_history,
                questioner,
                config,
            )
            future_values = evaluate_wordle_guesses(future_beliefs, future_candidates, eig)
            if future_values:
                expected_future_value += branch_probability * max(future_values)
        total_values[guess_index] += expected_future_value
    return total_values


def update_wordle_beliefs(
    beliefs: BeliefState,
    history: list[WordleTurn],
    questioner: "Model",
    config: Config,
) -> BeliefState:
    latest_turn = history[-1]
    filtered_prior_beliefs = filter_wordle_solutions(
        beliefs.beliefs,
        latest_turn.guess,
        latest_turn.feedback,
    )
    filtered_generated_beliefs: list[str] = []
    for attempt_idx in range(3):
        generated_beliefs = generate_wordle_beliefs(
            history,
            questioner,
            config,
            current_beliefs=filtered_prior_beliefs + filtered_generated_beliefs,
        )
        filtered_generated_beliefs = [
            word
            for word in generated_beliefs
            if all(wordle_feedback(turn.guess, word) == turn.feedback for turn in history)
        ]
        if filtered_generated_beliefs or attempt_idx == 2:
            break
        print(
            "[wordle] LLM generated no feedback-compatible beliefs; retrying with exact history constraints"
        )

    merged_beliefs = filtered_prior_beliefs + filtered_generated_beliefs
    return make_belief_state(merged_beliefs, fallback_to_uniform=True)


def run_wordle_single(
    target_word: str,
    questioner: "Model",
    method_name: str,
    config: Config,
) -> list[int]:
    target_word = validate_wordle_word(target_word)
    opening_beliefs = generate_wordle_opening_beliefs(questioner, config)
    beliefs = initialize_wordle_beliefs(opening_beliefs)
    history: list[WordleTurn] = []
    correct_guess = [0] * config.max_wordle_guesses
    eig = method_name == "EIG"
    naive = method_name == "naive"

    print(f"[wordle] Starting target {target_word} with {len(beliefs.beliefs)} possible solution(s)")
    write_to_log(f"\n\nStarting on Wordle target {target_word}\n", config)
    write_to_log(f"Initial Wordle beliefs: {format_belief_state(beliefs, top_n=20)}\n", config)

    for round_index in range(config.max_wordle_guesses):
        write_to_log(f"\nWordle target {target_word}: Round {round_index + 1}\n", config)
        print(
            f"[wordle] {target_word}: round {round_index + 1}/{config.max_wordle_guesses} "
            f"with {len(beliefs.beliefs)} possible solution(s)"
        )
        if naive:
            best_guess = generate_wordle_naive_guess(history, questioner, config)
            best_score = 0.0
        else:
            candidate_guesses = generate_wordle_candidate_guesses_from_llm(
                beliefs,
                history,
                questioner,
                config,
            )
            if len(candidate_guesses) == 0:
                candidate_guesses = [generate_wordle_naive_guess(history, questioner, config)]
            scores = evaluate_wordle_guesses_forward_search(
                beliefs,
                candidate_guesses,
                history,
                questioner,
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

        beliefs = update_wordle_beliefs(beliefs, history, questioner, config)
        write_to_log(f"Wordle history: {format_wordle_history(history)}\n", config)
        write_to_log(f"Current Wordle beliefs: {format_belief_state(beliefs, top_n=20)}\n", config)
        if len(beliefs.beliefs) == 0:
            print("[wordle] No possible solutions remain after feedback")
            return correct_guess

    return correct_guess


def run_wordle(method_name: str, questioner: "Model", config: Config) -> list[float]:
    if method_name not in {"naive", "Entropy", "EIG"}:
        raise ValueError("Wordle deterministic mode supports method_names: naive, Entropy, EIG")
    if config.wordle_solution_words_path is None:
        raise ValueError("Wordle config requires a target word-list path")

    solution_words = load_wordle_words(config.wordle_solution_words_path)
    target_words = solution_words
    accuracies = [0.0] * config.max_wordle_guesses
    print(f"[wordle] Running method {method_name} across {len(target_words)} target word(s)")

    for target_index, target_word in enumerate(target_words, start=1):
        wandb.log({
            "event": "start wordle",
            "target_word": target_word,
            "method": method_name,
        })
        correct_guess = run_wordle_single(target_word, questioner, method_name, config)
        accuracies = [accuracy + correct for accuracy, correct in zip(accuracies, correct_guess)]
        running_accuracy = [accuracy / target_index for accuracy in accuracies]
        write_to_log(f"Running Wordle accuracy trace: {running_accuracy}\n", config)
        print(f"[wordle] Finished {target_word}. Running accuracy trace: {running_accuracy}")

    return [accuracy / len(target_words) for accuracy in accuracies]
