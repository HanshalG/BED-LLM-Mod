import time
from dataclasses import dataclass

import wandb
import numpy as np

from helpers import Config, format_belief_state, format_categorical_belief_summary, generate_original_beliefs, \
    get_configured_prior, get_question_answered, is_guess_correct_via_answerer, print_and_log, write_to_log
from generate_candidate_questions import generate_candidate_questions, generate_candidate_question_naive, \
    evaluate_questions_forward_search
from model import Model
from sample_beliefs import sample_beliefs, sample_beliefs_naive
from update_beliefs import initialize_belief_state, update_beliefs_batched

NUM_ROUNDS = 20


@dataclass(frozen=True)
class GameMetrics:
    correct_guess: list[float]
    correct_belief_mass: list[float]

    def __getitem__(self, index):
        return self.correct_guess[index]

    def __eq__(self, other):
        if isinstance(other, list):
            return self.correct_guess == other
        return super().__eq__(other)


def _coerce_game_metrics(metrics: GameMetrics | list[int]) -> GameMetrics:
    if isinstance(metrics, GameMetrics):
        return metrics
    return GameMetrics(
        correct_guess=list(metrics),
        correct_belief_mass=[0.0] * len(metrics),
    )


def probability_mass_on_belief(beliefs, goal_animal: str) -> float:
    goal_key = goal_animal.strip().lower()
    return sum(
        probability
        for belief, probability in zip(beliefs.beliefs, beliefs.probabilities)
        if belief.strip().lower() == goal_key
    )


def twenty_questions_animals_single_EIG(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> GameMetrics:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=True, deterministic=False, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_entropy(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> GameMetrics:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=False, deterministic=False, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_split(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> GameMetrics:
    return twenty_questions_animals_single_complex(goal_animal=goal_animal, eig=False, deterministic=True, questioner=questioner, answerer=answerer, config=config)


def twenty_questions_animals_single_complex(goal_animal: str, eig: bool, deterministic: bool, questioner: Model, answerer: Model, config: Config) -> GameMetrics:
    history_questioner = []
    if config.belief_generation_enabled:
        print(f"[game] Generating initial beliefs for {goal_animal}")
        initial_beliefs = generate_original_beliefs(questioner, config)
    else:
        configured_prior = get_configured_prior(config)
        if configured_prior is None:
            raise ValueError("belief_generation_enabled=false requires a configured prior")
        print(f"[game] Using configured prior support for initial beliefs for {goal_animal}")
        initial_beliefs = configured_prior.beliefs
    beliefs = initialize_belief_state(initial_beliefs, history_questioner, questioner, config)
    print(f"[game] Starting belief set has {len(beliefs.beliefs)} candidate(s)")
    write_to_log(f"Original beliefs: {format_belief_state(beliefs)}\n", config)
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Running in weighted belief mode with opening beliefs: "
            f"{format_categorical_belief_summary(beliefs)}",
            config,
        )
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    correct_belief_mass = [0.0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        best_question_score = None
        best_question = None

        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config)
        print(f"[game] {goal_animal}: round {i+1}/{NUM_ROUNDS} with {len(beliefs.beliefs)} belief(s)")
        cand_questions = []
        question_EIGs = []
        if (
            config.belief_state_mode == "categorical"
            and config.belief_guess_threshold is not None
            and len(beliefs.beliefs) > 0
        ):
            guess_idx = int(np.argmax(beliefs.probabilities))
            top_belief = beliefs.beliefs[guess_idx]
            top_probability = float(beliefs.probabilities[guess_idx])
            if top_probability >= config.belief_guess_threshold:
                best_question = f"Is it {top_belief}?"
                print_and_log(
                    f"[categorical] Guess threshold reached: {top_belief} ({top_probability:.3f}) >= "
                    f"{config.belief_guess_threshold:.3f}; asking identity question",
                    config,
                )

        if best_question is None:
            # Generate candidate questions, select the question with best EIG
            print(f"[game] Generating up to {config.target_num_questions} candidate question(s)")
            cand_questions = generate_candidate_questions(beliefs, history_questioner, questioner,
                                                          config.generation_temperature_diverse, config.target_num_questions)
            print(f"[game] Generated {len(cand_questions)} candidate question(s)")
        if config.belief_state_mode == "categorical" and cand_questions:
            print_and_log(
                f"[categorical] Candidate questions selected for scoring ({len(cand_questions)}): "
                f"{cand_questions}",
                config,
            )
        if best_question is None:
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
                    depth=config.search_depth,
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
            correct_belief_mass[i:NUM_ROUNDS] = [1.0] * (len(correct_belief_mass) - i)
            return GameMetrics(correct_guess=correct_guess, correct_belief_mass=correct_belief_mass)

        # update the current beliefs to incorporate new questions
        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]
        print("[game] Updating beliefs with the latest question-answer pair")
        prior_top_belief = beliefs.beliefs[0] if len(beliefs.beliefs) > 0 else None
        beliefs = update_beliefs_batched(history_questioner, beliefs, questioner, deterministic, config)
        print(f"[game] Belief set now has {len(beliefs.beliefs)} candidate(s)")
        write_to_log(f"Current beliefs: {format_belief_state(beliefs)}\n", config)
        correct_mass = probability_mass_on_belief(beliefs, goal_animal)
        correct_belief_mass[i] = correct_mass
        print_and_log(
            f"[belief-mass] Probability mass assigned to correct belief after round {i+1}: {correct_mass:.6f}",
            config,
        )
        wandb.log({
            "correct_belief_mass": correct_mass,
            "round": i + 1,
            "goal_animal": goal_animal,
        })
        if config.belief_state_mode == "categorical":
            new_top_belief = beliefs.beliefs[0] if len(beliefs.beliefs) > 0 else None
            print_and_log(
                f"[categorical] Post-update weighted beliefs: {format_categorical_belief_summary(beliefs)}",
                config,
            )
            if prior_top_belief == new_top_belief:
                print_and_log(
                    f"[categorical] Top belief unchanged after round {i+1}: {new_top_belief}",
                    config,
                )
            else:
                print_and_log(
                    f"[categorical] Top belief changed after round {i+1}: {prior_top_belief} -> {new_top_belief}",
                    config,
                )

        # greedy decoding of current most likely belief
        print("[game] Sampling current best guess")
        if config.belief_state_mode == "categorical" and len(beliefs.beliefs) > 0:
            guess_idx = int(np.argmax(beliefs.probabilities))
            guess = beliefs.beliefs[guess_idx]
            print_and_log(
                f"[categorical] Greedy weighted guess: {guess} ({beliefs.probabilities[guess_idx]:.3f})",
                config,
            )
        else:
            guess = sample_beliefs(beliefs.beliefs, history_questioner, questioner, config.generation_temperature_simple)
        if guess.lower() == goal_animal.lower() or is_guess_correct_via_answerer(
            guess,
            goal_animal,
            answerer,
            config.answer_temperature,
        ):
            correct_guess[i] = 1
        print(f"[game] Current best guess after round {i+1}: {guess}")
        write_to_log(f"Current best guess: {guess}\n", config)

        elapsed_time = time.perf_counter() - start_time
        print(f"[game] Round {i+1} finished in {elapsed_time:.2f}s")

    return GameMetrics(correct_guess=correct_guess, correct_belief_mass=correct_belief_mass)


def twenty_questions_animals_single_naive(goal_animal: str, questioner: Model, answerer: Model, config: Config) -> GameMetrics:
    history_questioner = []
    prior_beliefs = get_configured_prior(config)
    # correct_guess[i] = 1 <--> questioner had it right after i-th question
    correct_guess = [0]*NUM_ROUNDS
    correct_belief_mass = [0.0]*NUM_ROUNDS
    for i in range(NUM_ROUNDS):
        start_time = time.perf_counter()
        write_to_log(f"\nGoal animal {goal_animal}: Round {i+1}\n", config)
        print(f"[game-naive] {goal_animal}: round {i+1}/{NUM_ROUNDS}")
        # prompt to ask a good question
        print("[game-naive] Generating next question")
        best_question = generate_candidate_question_naive(
            history_questioner,
            questioner,
            config.generation_temperature_simple,
            prior_beliefs=prior_beliefs,
        )
        print(f"[game-naive] Asking answerer: {best_question}")

        # Ask question, end game if correct animal was guessed
        answer = get_question_answered(best_question, goal_animal, answerer, config.answer_temperature)
        print(f"[game-naive] Answer received: {answer}")
        write_to_log(f"Best question: {best_question}, Answer: {answer}\n", config)
        if answer == "Correct!":
            print(f"[game-naive] Goal animal {goal_animal} identified in round {i+1}")
            correct_guess[i:NUM_ROUNDS] = [1] * (len(correct_guess) - i)
            return GameMetrics(correct_guess=correct_guess, correct_belief_mass=correct_belief_mass)

        history_questioner = history_questioner +  [{"role": "assistant", "content": best_question}, {"role": "user", "content": answer}]

        # greedy decoding of current most likely belief
        print("[game-naive] Sampling current best guess")
        guess = sample_beliefs_naive(
            history_questioner,
            questioner,
            config.generation_temperature_simple,
            prior_beliefs=prior_beliefs,
        )
        if guess.lower() == goal_animal.lower() or is_guess_correct_via_answerer(
            guess,
            goal_animal,
            answerer,
            config.answer_temperature,
        ):
            correct_guess[i] = 1
        print(f"[game-naive] Current best guess after round {i+1}: {guess}")
        write_to_log(f"Current best guess: {guess}\n", config)
        elapsed_time = time.perf_counter() - start_time
        print(f"[game-naive] Round {i+1} finished in {elapsed_time:.2f}s")
    return GameMetrics(correct_guess=correct_guess, correct_belief_mass=correct_belief_mass)


extraction_methods = {
    "naive": twenty_questions_animals_single_naive,
    "split": twenty_questions_animals_single_split,
    "Entropy": twenty_questions_animals_single_entropy,
    "EIG": twenty_questions_animals_single_EIG,
}


def twenty_questions_animals(questioner: Model, answerer: Model, target_animals: list[str], extraction_method_name: str, config: Config) -> GameMetrics:
    extraction_method = extraction_methods[extraction_method_name]
    accuracies = [0.0]*NUM_ROUNDS
    correct_belief_masses = [0.0]*NUM_ROUNDS
    prior_orders_by_trial: list[list[str] | None] = [None] * len(target_animals)
    if config.answerer_sample_from_prior:
        base_prior = get_configured_prior(config)
        if base_prior is None or len(base_prior.beliefs) == 0:
            raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
        num_trials = config.answerer_num_prior_trials
        if num_trials is None:
            num_trials = len(base_prior.beliefs)
        rng = np.random.default_rng(config.answerer_prior_seed)
        target_animals = []
        prior_orders_by_trial = []
        prior_summaries = []
        for _trial_idx in range(num_trials):
            if config.answerer_randomize_prior_order_per_trial:
                prior_order = [
                    base_prior.beliefs[int(index)]
                    for index in rng.permutation(len(base_prior.beliefs))
                ]
            else:
                prior_order = list(base_prior.beliefs)
            config.active_prior_animals = prior_order
            trial_prior = get_configured_prior(config)
            if trial_prior is None or len(trial_prior.beliefs) == 0:
                raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
            sampled_index = int(rng.choice(
                len(trial_prior.beliefs),
                p=trial_prior.probabilities,
            ))
            target_animals.append(trial_prior.beliefs[sampled_index])
            prior_orders_by_trial.append(prior_order)
            prior_summaries.append(format_categorical_belief_summary(trial_prior, top_n=5))
        config.active_prior_animals = None
        print_and_log(
            f"[categorical] Sampled answerer target sequence from prior: {target_animals}",
            config,
        )
        if config.answerer_randomize_prior_order_per_trial:
            print_and_log(
                f"[categorical] Randomized answerer prior summaries by trial: {prior_summaries}",
                config,
            )
        else:
            print_and_log(
                f"[categorical] Answerer sampling prior: {prior_summaries[0]}",
                config,
            )
    print(f"[game] Running method {extraction_method_name} across {len(target_animals)} animal(s)")
    for animal_idx, goal_animal in enumerate(target_animals, start=1):
        config.active_prior_animals = prior_orders_by_trial[animal_idx - 1]
        try:
            write_to_log(f"\n\nStarting on animal {goal_animal}\n", config)
            print(f"Starting on animal {goal_animal}")
            wandb.log({
                "event": "start animal",
                "goal_animal": goal_animal,
                "method": extraction_method_name,
            })
            game_metrics = _coerce_game_metrics(extraction_method(goal_animal, questioner, answerer, config))
            accuracies = [a + c for a, c in zip(accuracies, game_metrics.correct_guess)]
            correct_belief_masses = [
                total_mass + round_mass
                for total_mass, round_mass in zip(correct_belief_masses, game_metrics.correct_belief_mass)
            ]
            running_accuracy = [a / animal_idx for a in accuracies]
            running_correct_belief_mass = [mass / animal_idx for mass in correct_belief_masses]
            write_to_log(f"Running accuracy trace: {running_accuracy}\n", config)
            write_to_log(f"Running correct belief mass trace: {running_correct_belief_mass}\n", config)
            print(f"[game] Finished {goal_animal}. Running accuracy trace: {running_accuracy}")
        finally:
            config.active_prior_animals = None
    config.active_prior_animals = None
    return GameMetrics(
        correct_guess=[a / len(target_animals) for a in accuracies],
        correct_belief_mass=[mass / len(target_animals) for mass in correct_belief_masses],
    )
