"""Animals (20 Questions) module of :class:`core.Environment`.

The animals package owns the 20-Questions prompts, belief updates, candidate
generation, strategy search, and sampling code used by :class:`core.BEDRunner`.

The animals environment differs from location-finding in two important ways
that the adapter has to bridge:

1. **The "observation" comes from a second LLM**, the *answerer*, rather than
   from a deterministic simulator.  This adapter therefore takes an
   ``answerer`` model in its constructor and uses it inside :meth:`observe`.
2. **The likelihood is itself estimated via LLM calls** rather than a closed-form
   density.  We expose this through :meth:`log_likelihood_many`, which
   batches the underlying ``chat_probabilities_messages_batched`` call.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from core.experiment_summary import ExperimentSummary
from core.llm_likelihood import log_likelihood_many_llm_binary
from core.bed_runner import RunResult
from environments.animals.questions import generate_candidate_questions
from environments.animals.questions import evaluate_questions_forward_search, generate_candidate_question_naive
from helpers import (
    generate_original_beliefs,
    get_configured_prior,
    get_question_answered,
    is_guess_correct_via_answerer,
)
from environments.animals.sampling import sample_beliefs, sample_beliefs_naive
from environments.animals.prompts import answer_likelihood_messages
from environments.animals.beliefs import initialize_belief_state, update_beliefs_batched


# Type aliases for clarity:
#   S = str      — the goal animal (ground truth)
#   H = str      — a hypothesis is the name of a candidate animal
#   A = str      — an action is a Yes/No question (or "Is it <name>?")
#   O = str      — an observation is "Yes", "No", or "Correct!"


@dataclass
class AnimalsBEDEnvironment(Environment[str, str, str, str]):
    """Adapter that exposes 20 Questions through the BED ABC.

    Parameters
    ----------
    config:
        Flat :class:`helpers.Config` consumed by the animals package.
    answerer:
        The second LLM that plays the answerer role.  Required because the
        :class:`core.BEDRunner` itself only carries one ``model`` (the
        questioner); the answerer is a property of the environment.
    target_animals:
        The pool of ground-truth animals to draw from in
        :meth:`sample_hidden_state`.  If omitted, defaults to the first row of
        ``config.animals`` (matching ``config.version`` semantics).
    """

    config: Any
    answerer: Any  # core.types.Model
    target_animals: list[str] | None = None
    observation_labels: tuple[str, str] = ("Yes", "No")
    _animal_pool: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pool = self.target_animals
        if pool is None:
            version = int(getattr(self.config, "version", 0))
            animals_table = list(getattr(self.config, "animals", []) or [])
            if 0 <= version < len(animals_table):
                pool = list(animals_table[version])
        if not pool:
            raise ValueError(
                "AnimalsBEDEnvironment requires a non-empty target_animals list "
                "(either passed explicitly or available via config.animals[config.version])"
            )
        self._animal_pool = list(pool)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "animals"

    def configure_for_run(self, config: Any) -> "AnimalsBEDEnvironment":
        """Set the target pool for the current run, including prior sampling."""
        target_animals = list(config.animals[config.version])
        if config.answerer_sample_from_prior:
            from helpers import get_answerer_prior

            base_prior = get_answerer_prior(config)
            if base_prior is None or len(base_prior.hypotheses) == 0:
                raise ValueError(
                    "answerer_sample_from_prior=true requires a non-empty configured prior"
                )
            num_trials = config.answerer_num_prior_trials
            if num_trials is None:
                num_trials = len(base_prior.hypotheses)
            rng = np.random.default_rng(config.answerer_prior_seed)
            target_animals = []
            for _trial_idx in range(num_trials):
                if config.answerer_randomize_prior_order_per_trial:
                    prior_order = [
                        base_prior.hypotheses[int(index)]
                        for index in rng.permutation(len(base_prior.hypotheses))
                    ]
                else:
                    prior_order = list(base_prior.hypotheses)
                config.active_answerer_prior_animals = prior_order
                try:
                    trial_prior = get_answerer_prior(config)
                finally:
                    config.active_answerer_prior_animals = None
                if trial_prior is None or len(trial_prior.hypotheses) == 0:
                    raise ValueError(
                        "answerer_sample_from_prior=true requires a non-empty configured prior"
                    )
                sampled_index = int(
                    rng.choice(len(trial_prior.hypotheses), p=trial_prior.probabilities)
                )
                target_animals.append(trial_prior.hypotheses[sampled_index])
        self.target_animals = target_animals
        object.__setattr__(self, "_animal_pool", list(target_animals))
        return self

    def trial_count(self, config: Any) -> int:
        return max(1, len(self._animal_pool))

    def round_count(self, config: Any) -> int:
        return int(getattr(config, "animals_num_rounds", 20))

    def run_seed(self, config: Any) -> int | None:
        return getattr(config, "seed", None)

    # ------------------------------------------------------------------
    # Hidden state / simulation
    # ------------------------------------------------------------------

    def sample_hidden_state(self, rng: np.random.Generator) -> str:
        index = int(rng.integers(0, len(self._animal_pool)))
        return self._animal_pool[index]

    def sample_hidden_state_for_trial(self, trial_index: int, rng: np.random.Generator) -> str:
        if self.trial_count(self.config) == len(self._animal_pool):
            return self._animal_pool[trial_index % len(self._animal_pool)]
        return self.sample_hidden_state(rng)

    def observe(self, action: str, hidden_state: str, rng: np.random.Generator) -> str:
        # The answerer LLM is the source of randomness here.  ``rng`` is
        # accepted for protocol compliance but unused.
        return get_question_answered(
            action,
            hidden_state,
            self.answerer,
            self.config.answer_temperature,
        )

    # ------------------------------------------------------------------
    # Probabilistic model
    # ------------------------------------------------------------------

    def log_prior(self, hypothesis: str) -> float:
        prior = get_configured_prior(self.config)
        if prior is None:
            # Uniform over the configured pool.
            if not self._animal_pool:
                return float("-inf")
            if hypothesis in self._animal_pool:
                return -math.log(len(self._animal_pool))
            return float("-inf")
        lookup = {
            belief.lower(): probability
            for belief, probability in zip(prior.hypotheses, prior.probabilities)
        }
        probability = lookup.get(hypothesis.lower(), 0.0)
        return math.log(max(probability, 1e-300))

    def log_likelihood(self, hypothesis: str, action: str, observation: str) -> float:
        # The general path goes through the questioner — but the abstract
        # contract is single-item.  We expose the batched fast path below;
        # this fallback simply uses it for one item.
        if observation not in {"Yes", "No"}:
            return 0.0  # observations like "Correct!" carry no likelihood info
        probabilities = self.log_likelihood_many([hypothesis], action, observation)
        return float(probabilities[0])

    def build_likelihood_messages(self, hypothesis: str, action: str) -> list[dict[str, str]]:
        return answer_likelihood_messages(hypothesis, action, list(self.observation_labels))

    def get_questioner(self) -> Any:
        questioner = getattr(self, "_questioner", None)
        if questioner is None:
            raise RuntimeError(
                "log_likelihood_many requires set_questioner(model) to be called "
                "(usually done by BEDRunner via initial_belief_state)."
            )
        return questioner

    def log_likelihood_many(
        self,
        hypotheses: Sequence[str],
        action: str,
        observation: str,
    ) -> np.ndarray:
        return log_likelihood_many_llm_binary(
            self,
            hypotheses,
            action,
            observation,
            non_informative_observations=frozenset({"Correct!"}),
        )

    def set_questioner(self, questioner: Any) -> None:
        """Attach the questioner Model so likelihood batching can call into it."""
        self._questioner = questioner

    def set_active_method(self, method_name: str) -> None:
        self._active_method_name = method_name

    # ------------------------------------------------------------------
    # Belief support
    # ------------------------------------------------------------------

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[str]:
        # The animals belief support comes from one of two places:
        # (a) the configured prior, if belief_generation is disabled;
        # (b) a fresh LLM generation, otherwise.
        self.set_questioner(model)
        if config.belief_generation_enabled:
            initial_beliefs = generate_original_beliefs(model, config)
        else:
            prior = get_configured_prior(config)
            if prior is None:
                raise ValueError(
                    "belief_generation_enabled=false requires a configured prior"
                )
            initial_beliefs = list(prior.hypotheses)
        return initialize_belief_state(initial_beliefs, [], model, config)

    def update_belief_state(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> BeliefState[str]:
        self.set_questioner(model)
        history_messages = _history_to_messages(history)
        deterministic = False  # categorical/EIG path; deterministic mode is method-controlled
        return update_beliefs_batched(
            history_messages,
            belief_state,
            model,
            deterministic,
            config,
        )

    # ------------------------------------------------------------------
    # Action proposal
    # ------------------------------------------------------------------

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> list[str]:
        if (
            getattr(config, "belief_state_mode", None) == "categorical"
            and getattr(config, "belief_guess_threshold", None) is not None
            and belief_state.hypotheses
        ):
            top = belief_state.top()
            if top is not None and top[1] >= float(config.belief_guess_threshold):
                return [f"Is it {top[0]}?"]
        history_messages = _history_to_messages(history)
        return generate_candidate_questions(
            belief_state,
            history_messages,
            model,
            config.generation_temperature_diverse,
            config.target_num_questions,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def round_metrics(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        hidden_state: str,
    ) -> dict[str, float]:
        goal_lower = hidden_state.strip().lower()
        correct_mass = belief_state.probability_of(
            lambda h: h.strip().lower() == goal_lower
        )
        top = belief_state.top()
        top_match = float(
            top is not None and top[0].strip().lower() == goal_lower
        )
        metrics = {
            "correct_belief_mass": correct_mass,
            "top_belief_correct": top_match,
            "support_size": float(belief_state.support_size),
            "ess": belief_state.effective_sample_size(),
        }
        questioner = getattr(self, "_questioner", None)
        guess_correct = self._guess_correct(
            belief_state,
            history,
            hidden_state,
            model=questioner,
            config=self.config,
            check_via_answerer=True,
        )
        metrics["guess_correct"] = float(guess_correct)
        return metrics

    def generate_naive_action(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> str:
        history_messages = _history_to_messages(history)
        belief_conditioned = method_name == "naive+belief"
        prior_beliefs = (
            belief_state
            if belief_conditioned
            else get_configured_prior(config)
        )
        label = "current posterior belief state" if belief_conditioned else "prior distribution"
        return generate_candidate_question_naive(
            history_messages,
            model,
            config.generation_temperature_simple,
            prior_beliefs=prior_beliefs,
            belief_context_label=label,
        )

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        hidden_state: str,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        guess_correct = self._guess_correct(
            belief_state,
            history,
            hidden_state,
            model=model,
            config=config,
            method_name=method_name,
        )
        return {
            "guess_correct": float(guess_correct),
            "correct_belief_mass": belief_state.probability_of(
                lambda h: h.strip().lower() == hidden_state.strip().lower()
            ),
        }

    def choose_strategy_action(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> tuple[str, float, Any]:
        self.set_questioner(model)
        library = getattr(self, "_strategy_library", None)
        if library is None:
            library = AnimalsStrategyLibrary()
            self._strategy_library = library

        if fixed_root:
            strategy_pairs = self._generate_strategy_root_pairs(
                belief_state,
                history,
                model,
                config,
                library,
            )
            strategies = [pair[0] for pair in strategy_pairs]
            questions = [pair[1] for pair in strategy_pairs]
        else:
            strategies = self._generate_strategies(
                belief_state,
                history,
                model,
                config,
                library,
            )
            questions = [
                self._generate_question_from_strategy(strategy, belief_state, history, model, config)
                for strategy in strategies
            ]

        scored = []
        for strategy, question in zip(strategies, questions):
            if not question:
                continue
            evaluation = self._evaluate_strategy_rollouts(
                strategy,
                question,
                belief_state,
                history,
                model,
                config,
                rng,
            )
            scored.append((strategy, question, evaluation))
        if not scored:
            raise ValueError("Animals StrategyEIG could not produce a valid strategy question")

        best_idx = int(np.argmax([evaluation.mean_score for _strategy, _question, evaluation in scored]))
        best_strategy, best_question, best_evaluation = scored[best_idx]
        library.replace_entries(
            [
                AnimalsStrategyEntry(
                    strategy=strategy,
                    mean_score=float(evaluation.mean_score),
                    score_variance=float(evaluation.score_variance),
                    root_query_fingerprint=question if fixed_root else "",
                    round_index=round_index,
                    root_question=question if fixed_root else None,
                )
                for strategy, question, evaluation in scored
            ]
        )
        return best_question, float(best_evaluation.mean_score), {
            "strategy": best_strategy,
            "strategies": [strategy for strategy, _question, _evaluation in scored],
            "questions": [question for _strategy, question, _evaluation in scored],
            "scores": [float(evaluation.mean_score) for _strategy, _question, evaluation in scored],
            "score_variances": [
                float(evaluation.score_variance)
                for _strategy, _question, evaluation in scored
            ],
            "rollout_scores": [
                list(evaluation.rollout_scores)
                for _strategy, _question, evaluation in scored
            ],
            "fixed_root": fixed_root,
            "planning_depth": int(getattr(config, "animals_strategy_planning_depth", 1)),
            "num_rollouts": int(getattr(config, "animals_strategy_num_rollouts", 1)),
        }

    def _evaluate_strategy_rollouts(
        self,
        strategy: str,
        first_question: str,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
    ) -> "AnimalsStrategyEvaluation":
        if not belief_state.hypotheses:
            return AnimalsStrategyEvaluation(0.0, 0.0, ())

        num_rollouts = max(1, int(getattr(config, "animals_strategy_num_rollouts", 1)))
        planning_depth = max(1, int(getattr(config, "animals_strategy_planning_depth", 1)))
        hypotheses = tuple(belief_state.hypotheses)
        probabilities = np.asarray(belief_state.probabilities, dtype=float)
        rollout_scores: list[float] = []

        for _rollout_idx in range(num_rollouts):
            truth_idx = int(rng.choice(len(hypotheses), p=probabilities))
            truth = hypotheses[truth_idx]
            current_belief = belief_state
            rollout_history = list(history)
            start_entropy = current_belief.entropy()

            for depth_idx in range(planning_depth):
                if current_belief.support_size <= 1:
                    break
                if depth_idx == 0:
                    question = first_question
                else:
                    question = self._generate_question_from_strategy(
                        strategy,
                        current_belief,
                        rollout_history,
                        model,
                        config,
                    )
                if not question:
                    break

                yes_probabilities = self._yes_probabilities(current_belief, question)
                try:
                    current_truth_idx = current_belief.hypotheses.index(truth)
                except ValueError:
                    break
                yes_probability = float(yes_probabilities[current_truth_idx])
                observation = "Yes" if rng.random() < yes_probability else "No"
                current_belief = _bayes_update_binary_belief(
                    current_belief,
                    yes_probabilities,
                    observation,
                )
                rollout_history.append((question, observation))

            rollout_scores.append(max(0.0, start_entropy - current_belief.entropy()))

        scores = np.asarray(rollout_scores, dtype=float)
        return AnimalsStrategyEvaluation(
            mean_score=float(np.mean(scores)),
            score_variance=float(np.var(scores)),
            rollout_scores=tuple(float(score) for score in rollout_scores),
        )

    def _yes_probabilities(self, belief_state: BeliefState[str], question: str) -> np.ndarray:
        if not belief_state.hypotheses:
            return np.empty(0, dtype=float)
        questioner = self.get_questioner()
        positive_label, negative_label = self.observation_labels
        conversations = [
            self.build_likelihood_messages(hypothesis, question)
            for hypothesis in belief_state.hypotheses
        ]
        rows = questioner.chat_probabilities_messages_batched(
            conversations,
            [positive_label, negative_label],
            temperature=self.config.answer_temperature,
            block_size=self.config.batched_block_size,
        )
        return np.asarray(
            [
                min(max(float(row.get(positive_label, 0.0)), 1.0e-9), 1.0 - 1.0e-9)
                for row in rows
            ],
            dtype=float,
        )

    def _generate_strategies(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        library: AnimalsStrategyLibrary,
    ) -> list[str]:
        retrieved = library.retrieve_top_m(_strategy_count(config, "retrieved"))
        strategies = [entry.strategy for entry in retrieved]
        generated_count = (
            _strategy_count(config, "mutation")
            + _strategy_count(config, "crossover")
            + _strategy_count(config, "diverse")
        )
        if not strategies and generated_count == 0:
            generated_count = max(1, _strategy_count(config, "retrieved"))
        if generated_count > 0:
            strategies.extend(
                _parse_strategy_list(
                    _chat_one(
                        model,
                        _animals_strategy_messages(
                            belief_state,
                            history,
                            config,
                            count=generated_count,
                            retrieved=retrieved,
                        ),
                        config,
                    )
                )
            )
        return _dedupe_texts(strategies)

    def _generate_strategy_root_pairs(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        library: AnimalsStrategyLibrary,
    ) -> list[tuple[str, str]]:
        count = (
            _strategy_count(config, "retrieved")
            + _strategy_count(config, "mutation")
            + _strategy_count(config, "crossover")
            + _strategy_count(config, "diverse")
        )
        count = max(1, count)
        completion = _chat_one(
            model,
            _animals_strategy_root_messages(
                belief_state,
                history,
                config,
                count=count,
                retrieved=library.retrieve_top_m(_strategy_count(config, "retrieved")),
            ),
            config,
        )
        pairs = _parse_strategy_root_pairs(completion)
        if pairs:
            return pairs
        return [
            (strategy, self._generate_question_from_strategy(strategy, belief_state, history, model, config))
            for strategy in self._generate_strategies(belief_state, history, model, config, library)
        ]

    def _generate_question_from_strategy(
        self,
        strategy: str,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> str:
        question = _chat_one(
            model,
            _animals_strategy_question_messages(strategy, belief_state, history, config),
            config,
        ).strip()
        parsed = _parse_question_text(question)
        if parsed:
            return parsed
        fallback = generate_candidate_question_naive(
            _history_to_messages(history),
            model,
            config.generation_temperature_simple,
            prior_beliefs=belief_state,
            belief_context_label="current posterior belief state",
        )
        return fallback.strip()

    def early_stop(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        hidden_state: str,
        latest_observation: str,
    ) -> bool:
        return latest_observation == "Correct!"

    def summarize_run(self, run_result: RunResult, config: Any) -> ExperimentSummary:
        from environments.animals.game_metrics import summarize_animals_run_metrics

        base = super().summarize_run(run_result, config)
        raw_metrics = dict(base.metrics)
        if "guess_correct" in raw_metrics and "accuracy" not in raw_metrics:
            raw_metrics["accuracy"] = raw_metrics.pop("guess_correct")
        method_name = getattr(self, "_active_method_name", None)
        if method_name is None:
            methods = getattr(config, "method_names", None) or []
            method_name = methods[0] if methods else "EIG"
        metrics = summarize_animals_run_metrics(
            run_result,
            raw_metrics,
            method_name=str(method_name),
            config=config,
        )
        return ExperimentSummary(metrics=metrics, logs=base.logs)

    def _guess_correct(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        hidden_state: str,
        model: Any | None,
        config: Any,
        method_name: str | None = None,
        *,
        check_via_answerer: bool = True,
    ) -> bool:
        history_messages = _history_to_messages(history)
        if config.belief_state_mode == "categorical" and belief_state.hypotheses:
            guess_idx = int(np.argmax(belief_state.probabilities))
            guess = belief_state.hypotheses[guess_idx]
        elif model is not None:
            guess = sample_beliefs(
                list(belief_state.hypotheses),
                history_messages,
                model,
                config.generation_temperature_simple,
            )
        else:
            top = belief_state.top()
            guess = top[0] if top is not None else ""
        if guess.strip().lower() == hidden_state.strip().lower():
            return True
        if not check_via_answerer:
            return False
        return is_guess_correct_via_answerer(
            guess,
            hidden_state,
            self.answerer,
            config.answer_temperature,
        )


def _history_to_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """Convert BEDRunner-style (action, observation) pairs into the message format
    that the animals belief/question modules expect: a flat list of role-tagged
    dicts in chronological order."""
    messages: list[dict[str, str]] = []
    for question, answer in history:
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})
    return messages


def _strategy_count(config: Any, name: str) -> int:
    animal_key = f"animals_strategy_num_{name}"
    location_key = f"location_strategy_num_{name}"
    if hasattr(config, animal_key):
        return int(getattr(config, animal_key))
    return int(getattr(config, location_key, {"retrieved": 2, "mutation": 1, "crossover": 1, "diverse": 2}[name]))


def _format_animals_belief_state(belief_state: BeliefState[str], top_n: int = 10) -> str:
    entries = belief_state.top_k(top_n)
    return json.dumps(
        [
            {"animal": hypothesis, "probability": probability}
            for hypothesis, probability in entries
        ]
    )


def _format_animals_history(history: Sequence[tuple[str, str]]) -> str:
    return json.dumps(
        [
            {"question": question, "answer": answer}
            for question, answer in history
        ]
    )


def _chat_one(model: Any, messages: list[dict[str, str]], config: Any) -> str:
    return model.chat_complete(
        messages,
        temperature=getattr(config, "generation_temperature_diverse", 1.0),
    )[0]


def _animals_strategy_messages(
    belief_state: BeliefState[str],
    history: Sequence[tuple[str, str]],
    config: Any,
    *,
    count: int,
    retrieved: Sequence[AnimalsStrategyEntry],
) -> list[dict[str, str]]:
    retrieved_text = "\n".join(
        f"- {entry.strategy} (score={entry.mean_score:.4f})"
        for entry in retrieved
    ) or "None"
    system = (
        "You design high-level strategies for a 20 Questions animal-identification game. "
        "A strategy is not itself a question; it is an instruction for choosing future Yes/No questions. "
        "Return only JSON."
    )
    user = (
        f"Observation history:\n{_format_animals_history(history)}\n\n"
        f"Current belief summary:\n{_format_animals_belief_state(belief_state, top_n=getattr(config, 'animals_strategy_belief_summary_top_k', 5))}\n\n"
        f"Retrieved elite strategies:\n{retrieved_text}\n\n"
        f"Generate exactly {count} strategies. Prefer strategies that discriminate between high-probability animals "
        "and leave useful follow-up questions for later rounds.\n\n"
        'Return exactly {"strategies":["strategy text", "..."]}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _animals_strategy_root_messages(
    belief_state: BeliefState[str],
    history: Sequence[tuple[str, str]],
    config: Any,
    *,
    count: int,
    retrieved: Sequence[AnimalsStrategyEntry],
) -> list[dict[str, str]]:
    retrieved_text = "\n".join(
        f"- {entry.strategy} (root={entry.root_question or 'none'}, score={entry.mean_score:.4f})"
        for entry in retrieved
    ) or "None"
    system = (
        "You design strategies and fixed root Yes/No questions for a 20 Questions animal game. "
        "Each root_question must be a single question answerable by Yes or No. Return only JSON."
    )
    user = (
        f"Observation history:\n{_format_animals_history(history)}\n\n"
        f"Current belief summary:\n{_format_animals_belief_state(belief_state, top_n=getattr(config, 'animals_strategy_belief_summary_top_k', 5))}\n\n"
        f"Retrieved elite strategies:\n{retrieved_text}\n\n"
        f"Generate exactly {count} strategy/root_question pairs.\n\n"
        'Return exactly {"candidates":[{"strategy":"strategy text","root_question":"question?"}]}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _animals_strategy_question_messages(
    strategy: str,
    belief_state: BeliefState[str],
    history: Sequence[tuple[str, str]],
    config: Any,
) -> list[dict[str, str]]:
    system = (
        "You choose the next Yes/No question for a 20 Questions animal-identification game. "
        "Follow the provided strategy. Return only the question."
    )
    user = (
        f"Observation history:\n{_format_animals_history(history)}\n\n"
        f"Current belief summary:\n{_format_animals_belief_state(belief_state, top_n=getattr(config, 'animals_strategy_belief_summary_top_k', 5))}\n\n"
        f"Strategy to follow:\n{strategy}\n\n"
        "Generate one useful Yes/No question. Do not include numbering or explanation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _parse_strategy_list(text: str) -> list[str]:
    for value in _extract_json_values(text):
        if isinstance(value, dict) and isinstance(value.get("strategies"), list):
            return [
                str(item).strip()
                for item in value["strategies"]
                if str(item).strip()
            ]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    return [
        line.strip(" -\t")
        for line in text.splitlines()
        if line.strip(" -\t")
    ]


def _parse_strategy_root_pairs(text: str) -> list[tuple[str, str]]:
    for value in _extract_json_values(text):
        raw_items: Any
        if isinstance(value, dict):
            raw_items = value.get("candidates", value.get("strategies", []))
        else:
            raw_items = value
        if not isinstance(raw_items, list):
            continue
        pairs: list[tuple[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("strategy", "")).strip()
            question = _parse_question_text(str(item.get("root_question", item.get("question", ""))))
            if strategy and question:
                pairs.append((strategy, question))
        if pairs:
            return pairs
    return []


def _parse_question_text(text: str) -> str:
    stripped = text.strip()
    for value in _extract_json_values(stripped):
        if isinstance(value, dict):
            candidate = value.get("question", value.get("root_question", ""))
            if candidate:
                stripped = str(candidate).strip()
                break
    stripped = re.sub(r"^\s*[-*\d.)]+\s*", "", stripped).strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def _dedupe_texts(items: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _bayes_update_binary_belief(
    belief_state: BeliefState[str],
    yes_probabilities: Sequence[float],
    observation: str,
) -> BeliefState[str]:
    prior = np.asarray(belief_state.probabilities, dtype=float)
    yes = np.asarray(yes_probabilities, dtype=float)
    if len(yes) != len(prior):
        raise ValueError("yes_probabilities must match belief support size")
    likelihood = yes if observation == "Yes" else 1.0 - yes
    weights = prior * np.clip(likelihood, 1.0e-12, 1.0)
    return BeliefState.from_unnormalized(
        belief_state.hypotheses,
        weights,
        fallback_to_uniform=True,
    ).sorted_descending()


@dataclass(frozen=True)
class AnimalsStrategyEvaluation:
    mean_score: float
    score_variance: float
    rollout_scores: tuple[float, ...]


@dataclass(frozen=True)
class AnimalsStrategyEntry:
    strategy: str
    mean_score: float
    score_variance: float
    root_query_fingerprint: str
    round_index: int
    root_question: str | None = None


class AnimalsStrategyLibrary:
    def __init__(self) -> None:
        self.entries: list[AnimalsStrategyEntry] = []

    def retrieve_top_m(self, count: int) -> list[AnimalsStrategyEntry]:
        if count <= 0:
            return []
        ranked = sorted(
            self.entries,
            key=lambda entry: (entry.mean_score, -entry.score_variance),
            reverse=True,
        )
        return ranked[:count]

    def replace_entries(self, entries: list[AnimalsStrategyEntry]) -> None:
        self.entries = list(entries)
