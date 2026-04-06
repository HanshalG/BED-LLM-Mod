import time

import numpy as np
import wandb

from helpers import get_question_answered, generate_original_beliefs, write_to_log, Config
from generate_candidate_questions import generate_candidate_questions, generate_candidate_question_naive, \
    evaluate_questions_forward_search
from model import Model
from sample_beliefs import sample_beliefs, sample_beliefs_naive
from update_beliefs import update_beliefs_batched

NUM_ROUNDS = 20


def twenty_questions_animals_single_EIG(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> list[int]:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=True, deterministic=False, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_entropy(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> list[int]:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=False, deterministic=False, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_split(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> list[int]:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=False, deterministic=True, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_complex(goal_animal: str, eig: bool, deterministic: bool, questioner: Model, answerer: Model, config: Config) -> list[int]:
    history_questioner = []
    beliefs = generate_original_beliefs(questioner, config)
    write_to_log(f"Original beliefs: {beliefs}\n", config.version)
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        best_question_score = None

        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config.version)
        # Generate candidate questions, select the question with best EIG
        cand_questions = generate_candidate_questions(beliefs, history_questioner, questioner,
                                                      config.generation_temperature_diverse, config.target_num_questions)
        if len(cand_questions) > 1:
            question_EIGs = evaluate_questions_forward_search(
                beliefs,
                history_questioner,
                cand_questions,
                eig,
                deterministic,
                questioner,
                config,
                depth=2,
            )
            best_idx = int(np.argmax(question_EIGs))
            best_question = cand_questions[best_idx]
            best_question_score = float(question_EIGs[best_idx])
        else:
            best_question = cand_questions[0]

        # Ask the best question, end game if correct animal was guessed
        answer = get_question_answered(best_question, goal_animal, answerer, config.answer_temperature)
        write_to_log(f"Best question: {best_question}, Answer: {answer}\n", config.version)
        if answer == "Correct!":
            correct_guess[i:NUM_ROUNDS] = [1] * (len(correct_guess) - i)
            return correct_guess

        # update the current beliefs to incorporate new questions
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        beliefs = update_beliefs_batched(history_questioner, beliefs, questioner, deterministic, config)
        write_to_log(f"Current beliefs: {beliefs}\n", config.version)

        # greedy decoding of current most likely belief
        guess = sample_beliefs(beliefs, history_questioner, questioner, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
          correct_guess[i] = 1
        write_to_log(f"Current best guess: {guess}\n", config.version)

        elapsed_time = time.perf_counter() - start_time

    return correct_guess


def twenty_questions_animals_single_naive(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> list[int]:
    history_questioner = []
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config.version)
        # prompt to ask a good question
        best_question = generate_candidate_question_naive(history_questioner, questioner, config.generation_temperature_simple)

        # Ask question, end game if correct animal was guessed
        answer = get_question_answered(best_question, goal_animal, answerer, config.answer_temperature)
        write_to_log(f"Best question: {best_question}, Answer: {answer}\n", config.version)
        if answer == "Correct!":
            correct_guess[i:NUM_ROUNDS] = [1] * (len(correct_guess) - i)
            return correct_guess

        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]

        # greedy decoding of current most likely belief
        guess = sample_beliefs_naive(history_questioner, questioner, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
            correct_guess[i] = 1
        write_to_log(f"Current best guess: {guess}\n", config.version)
    return correct_guess


extraction_methods = {
    "naive": twenty_questions_animals_single_naive,
    "split": twenty_questions_animals_single_split,
    "Entropy": twenty_questions_animals_single_entropy,
    "EIG": twenty_questions_animals_single_EIG,
}


def twenty_questions_animals(questioner: Model, answerer: Model, target_animals: list[str], extraction_method_name: str, config: Config) -> list[float]:
    extraction_method = extraction_methods[extraction_method_name]
    accuracies = [0.0]*NUM_ROUNDS
    for goal_animal in target_animals:
        write_to_log(f"\n\nStarting on animal {goal_animal}\n", config.version)
        wandb.log({
            "event": "start animal",
            "goal_animal": goal_animal,
            "method": extraction_method_name,
        })
        correct_guess = extraction_method(goal_animal, questioner, answerer, config)
        accuracies = [a + c for a, c in zip(accuracies, correct_guess)]
    return [a / len(target_animals) for a in accuracies]
