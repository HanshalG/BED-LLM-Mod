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
    print(f"[game] Generating initial beliefs for {goal_animal}")
    beliefs = generate_original_beliefs(questioner, config)
    print(f"[game] Starting belief set has {len(beliefs)} candidate(s)")
    write_to_log(f"Original beliefs: {beliefs}\n", config)
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        best_question_score = None

        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config)
        print(f"[game] {goal_animal}: round {i+1}/{NUM_ROUNDS} with {len(beliefs)} belief(s)")
        # Generate candidate questions, select the question with best EIG
        print(f"[game] Generating up to {config.target_num_questions} candidate question(s)")
        cand_questions = generate_candidate_questions(beliefs, history_questioner, questioner,
                                                      config.generation_temperature_diverse, config.target_num_questions)
        print(f"[game] Generated {len(cand_questions)} candidate question(s)")
        if len(cand_questions) > 1:
            print(f"[game] Scoring candidate questions using {'EIG' if eig else 'entropy'} search")
            question_EIGs = evaluate_questions_forward_search(
                beliefs,
                history_questioner,
                cand_questions,
                eig,
                deterministic,
                questioner,
                config,
                depth=1,
            )
            best_idx = int(np.argmax(question_EIGs))
            best_question = cand_questions[best_idx]
            best_question_score = float(question_EIGs[best_idx])
        else:
            best_question = cand_questions[0]

        if best_question_score is None:
            print(f"[game] Selected question: {best_question}")
        else:
            print(f"[game] Selected question: {best_question} (score={best_question_score:.4f})")
        # Ask the best question, end game if correct animal was guessed
        print(f"[game] Asking answerer: {best_question}")
        answer = get_question_answered(best_question, goal_animal, answerer, config.answer_temperature)
        print(f"[game] Answer received: {answer}")
        if best_question_score is None:
            write_to_log(f"Best question: {best_question}, Answer: {answer}\n", config)
        else:
            write_to_log(
                f"Best question: {best_question} (score={best_question_score:.4f}), Answer: {answer}\n",
                config,
            )

        #next 3 best questions and score
        if len(cand_questions) >= 4:
            cand_questions_scores = sorted(zip(cand_questions, question_EIGs), key=lambda x: x[1], reverse=True)
            for j in range(1, 4):
                print(f"[game] Next best question {j}: {cand_questions_scores[j][0]} (score={cand_questions_scores[j][1]:.4f})")
                write_to_log(
                    f"Next best question {j}: {cand_questions_scores[j][0]} (score={cand_questions_scores[j][1]:.4f})\n",
                    config,
                )

        if answer == "Correct!":
            print(f"[game] Goal animal {goal_animal} identified in round {i+1}")
            correct_guess[i:NUM_ROUNDS] = [1] * (len(correct_guess) - i)
            return correct_guess

        # update the current beliefs to incorporate new questions
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        print("[game] Updating beliefs with the latest question-answer pair")
        beliefs = update_beliefs_batched(history_questioner, beliefs, questioner, deterministic, config)
        print(f"[game] Belief set now has {len(beliefs)} candidate(s)")
        write_to_log(f"Current beliefs: {beliefs}\n", config)

        # greedy decoding of current most likely belief
        print("[game] Sampling current best guess")
        guess = sample_beliefs(beliefs, history_questioner, questioner, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
            correct_guess[i] = 1
        print(f"[game] Current best guess after round {i+1}: {guess}")
        write_to_log(f"Current best guess: {guess}\n", config)

        elapsed_time = time.perf_counter() - start_time
        print(f"[game] Round {i+1} finished in {elapsed_time:.2f}s")

    return correct_guess


def twenty_questions_animals_single_naive(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> list[int]:
    history_questioner = []
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config)
        print(f"[game-naive] {goal_animal}: round {i+1}/{NUM_ROUNDS}")
        # prompt to ask a good question
        print("[game-naive] Generating next question")
        best_question = generate_candidate_question_naive(history_questioner, questioner, config.generation_temperature_simple)
        print(f"[game-naive] Asking answerer: {best_question}")

        # Ask question, end game if correct animal was guessed
        answer = get_question_answered(best_question, goal_animal, answerer, config.answer_temperature)
        print(f"[game-naive] Answer received: {answer}")
        write_to_log(f"Best question: {best_question}, Answer: {answer}\n", config)
        if answer == "Correct!":
            print(f"[game-naive] Goal animal {goal_animal} identified in round {i+1}")
            correct_guess[i:NUM_ROUNDS] = [1] * (len(correct_guess) - i)
            return correct_guess

        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]

        # greedy decoding of current most likely belief
        print("[game-naive] Sampling current best guess")
        guess = sample_beliefs_naive(history_questioner, questioner, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower():
            correct_guess[i] = 1
        print(f"[game-naive] Current best guess after round {i+1}: {guess}")
        write_to_log(f"Current best guess: {guess}\n", config)
        elapsed_time = time.perf_counter() - start_time
        print(f"[game-naive] Round {i+1} finished in {elapsed_time:.2f}s")
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
    print(f"[game] Running method {extraction_method_name} across {len(target_animals)} animal(s)")
    for animal_idx, goal_animal in enumerate(target_animals, start=1):
        write_to_log(f"\n\nStarting on animal {goal_animal}\n", config)
        print(f"Starting on animal {goal_animal}")
        wandb.log({
            "event": "start animal",
            "goal_animal": goal_animal,
            "method": extraction_method_name,
        })
        correct_guess = extraction_method(goal_animal, questioner, answerer, config)
        accuracies = [a + c for a, c in zip(accuracies, correct_guess)]
        running_accuracy = [a / animal_idx for a in accuracies]
        write_to_log(f"Running accuracy trace: {running_accuracy}\n", config)
        print(f"[game] Finished {goal_animal}. Running accuracy trace: {running_accuracy}")
    return [a / len(target_animals) for a in accuracies]
