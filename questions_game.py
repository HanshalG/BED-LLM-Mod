"""20 Questions (animals) experiment helpers — BEDRunner-backed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wandb

from core.bed_runner import RunResult
from core.constants import NUM_ROUNDS_ANIMALS
from environments.animals.game_metrics import animals_num_rounds
from helpers import (
    Config,
    format_categorical_belief_summary,
    get_answerer_prior,
    get_configured_prior,
    print_and_log,
    write_to_log,
)
from model import Model

NUM_ROUNDS = NUM_ROUNDS_ANIMALS


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


def _summary_to_game_metrics(summary) -> GameMetrics:
    return GameMetrics(
        correct_guess=list(summary.metrics.get("accuracy", [])),
        correct_belief_mass=list(summary.metrics.get("correct_belief_mass", [])),
    )


def _run_single_animal_trial(
    goal_animal: str,
    method_name: str,
    questioner: Model,
    answerer: Model,
    config: Config,
) -> GameMetrics:
    from core import BEDRunner, build_method
    from core.defaults import register_defaults
    from environments.animals import AnimalsBEDEnvironment

    register_defaults()
    env = AnimalsBEDEnvironment(
        config=config,
        answerer=answerer,
        target_animals=[goal_animal],
    )
    method = build_method("animals", method_name, config)
    trial = BEDRunner(
        environment=env,
        method=method,
        model=questioner,
        config=config,
        num_trials=1,
        num_rounds=animals_num_rounds(config),
    ).run_single_trial(0)
    summary = env.summarize_run(RunResult(trials=(trial,)), config)
    return _summary_to_game_metrics(summary)


def twenty_questions_animals_single_EIG(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "EIG", questioner, answerer, config)


def twenty_questions_animals_single_strategy_eig(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "StrategyEIG", questioner, answerer, config)


def twenty_questions_animals_single_strategy_eig_root(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "StrategyEIG+root", questioner, answerer, config)


def twenty_questions_animals_single_entropy(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "Entropy", questioner, answerer, config)


def twenty_questions_animals_single_split(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "split", questioner, answerer, config)


def twenty_questions_animals_single_complex(
    goal_animal: str,
    eig: bool,
    deterministic: bool,
    questioner: Model,
    answerer: Model,
    config: Config,
) -> GameMetrics:
    del deterministic
    method_name = "EIG" if eig else "Entropy"
    return _run_single_animal_trial(goal_animal, method_name, questioner, answerer, config)


def twenty_questions_animals_single_naive(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "naive", questioner, answerer, config)


def twenty_questions_animals_single_naive_belief(
    goal_animal: str, questioner: Model, answerer: Model, config: Config
) -> GameMetrics:
    return _run_single_animal_trial(goal_animal, "naive+belief", questioner, answerer, config)


extraction_methods = {
    "naive": twenty_questions_animals_single_naive,
    "naive+belief": twenty_questions_animals_single_naive_belief,
    "split": twenty_questions_animals_single_split,
    "Entropy": twenty_questions_animals_single_entropy,
    "EIG": twenty_questions_animals_single_EIG,
    "StrategyEIG": twenty_questions_animals_single_strategy_eig,
    "StrategyEIG+root": twenty_questions_animals_single_strategy_eig_root,
}


def twenty_questions_animals(
    questioner: Model,
    answerer: Model,
    target_animals: list[str],
    extraction_method_name: str,
    config: Config,
) -> GameMetrics:
    num_rounds = animals_num_rounds(config)
    if config.answerer_sample_from_prior:
        questioner_prior = get_configured_prior(config)
        base_prior = get_answerer_prior(config)
        if base_prior is None or len(base_prior.beliefs) == 0:
            raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
        num_trials = config.answerer_num_prior_trials
        if num_trials is None:
            num_trials = len(base_prior.beliefs)
        rng = np.random.default_rng(config.answerer_prior_seed)
        target_animals = []
        prior_summaries = []
        for _trial_idx in range(num_trials):
            if config.answerer_randomize_prior_order_per_trial:
                prior_order = [
                    base_prior.beliefs[int(index)]
                    for index in rng.permutation(len(base_prior.beliefs))
                ]
            else:
                prior_order = list(base_prior.beliefs)
            config.active_answerer_prior_animals = prior_order
            try:
                trial_prior = get_answerer_prior(config)
            finally:
                config.active_answerer_prior_animals = None
            if trial_prior is None or len(trial_prior.beliefs) == 0:
                raise ValueError("answerer_sample_from_prior=true requires a non-empty configured prior")
            sampled_index = int(
                rng.choice(len(trial_prior.beliefs), p=trial_prior.probabilities)
            )
            target_animals.append(trial_prior.beliefs[sampled_index])
            prior_summaries.append(format_categorical_belief_summary(trial_prior, top_n=5))
        if questioner_prior is None:
            print_and_log("[categorical] Questioner prior: none", config)
        else:
            print_and_log(
                f"[categorical] Questioner prior: {format_categorical_belief_summary(questioner_prior, top_n=5)}",
                config,
            )
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
    accuracies = [0.0] * num_rounds
    correct_belief_masses = [0.0] * num_rounds
    for animal_idx, goal_animal in enumerate(target_animals, start=1):
        try:
            write_to_log(f"\n\nStarting on animal {goal_animal}\n", config)
            print(f"Starting on animal {goal_animal}")
            wandb.log({
                "event": "start animal",
                "goal_animal": goal_animal,
                "method": extraction_method_name,
            })
            game_metrics = _coerce_game_metrics(
                _run_single_animal_trial(
                    goal_animal,
                    extraction_method_name,
                    questioner,
                    answerer,
                    config,
                )
            )
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
            config.active_answerer_prior_animals = None
    config.active_prior_animals = None
    config.active_answerer_prior_animals = None
    return GameMetrics(
        correct_guess=[a / len(target_animals) for a in accuracies],
        correct_belief_mass=[mass / len(target_animals) for mass in correct_belief_masses],
    )
