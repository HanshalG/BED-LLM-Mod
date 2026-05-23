"""Animals (20 Questions) implementation of :class:`core.Environment`.

This adapter wraps the existing functions in :mod:`questions_game`,
:mod:`update_beliefs`, :mod:`generate_candidate_questions`, and :mod:`helpers`
so the 20-Questions problem can be driven through :class:`core.BEDRunner`.

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
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from core.experiment_summary import ExperimentSummary
from core.llm_likelihood import log_likelihood_many_llm_binary
from core.bed_runner import RunResult
from generate_candidate_questions import generate_candidate_questions
from generate_candidate_questions import evaluate_questions_forward_search, generate_candidate_question_naive
from helpers import (
    BeliefState as LegacyAnimalsBeliefState,
    generate_original_beliefs,
    get_configured_prior,
    get_question_answered,
    is_guess_correct_via_answerer,
)
from sample_beliefs import sample_beliefs, sample_beliefs_naive
from prompts import answer_likelihood_messages
from update_beliefs import initialize_belief_state, update_beliefs_batched


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
        Legacy flat :class:`helpers.Config`.  Required by the existing animals
        code paths that this adapter delegates to.
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

    # ------------------------------------------------------------------
    # Hidden state / simulation
    # ------------------------------------------------------------------

    def sample_hidden_state(self, rng: np.random.Generator) -> str:
        index = int(rng.integers(0, len(self._animal_pool)))
        return self._animal_pool[index]

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
            for belief, probability in zip(prior.beliefs, prior.probabilities)
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
            initial_beliefs = list(prior.beliefs)
        legacy_state = initialize_belief_state(initial_beliefs, [], model, config)
        return _belief_state_from_legacy(legacy_state)

    def update_belief_state(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> BeliefState[str]:
        self.set_questioner(model)
        legacy_belief = _belief_state_to_legacy(belief_state)
        history_messages = _history_to_messages(history)
        deterministic = False  # categorical/EIG path; deterministic mode is method-controlled
        legacy_updated = update_beliefs_batched(
            history_messages,
            legacy_belief,
            model,
            deterministic,
            config,
        )
        return _belief_state_from_legacy(legacy_updated)

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
        legacy_belief = _belief_state_to_legacy(belief_state)
        history_messages = _history_to_messages(history)
        return generate_candidate_questions(
            legacy_belief,
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
            _belief_state_to_legacy(belief_state)
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
        legacy_belief = _belief_state_to_legacy(belief_state)
        history_messages = _history_to_messages(history)
        candidates = generate_candidate_questions(
            legacy_belief,
            history_messages,
            model,
            config.generation_temperature_diverse,
            config.target_num_questions,
            verbose=False,
        )
        if not candidates:
            raise ValueError("Animals StrategyEIG could not produce candidate questions")
        scores = evaluate_questions_forward_search(
            legacy_belief,
            history_messages,
            candidates,
            True,
            False,
            model,
            config,
            depth=config.search_depth,
        )
        best_idx = int(np.argmax(scores)) if scores else 0
        return candidates[best_idx], float(scores[best_idx]) if scores else 0.0, {
            "candidates": candidates,
            "scores": scores,
            "fixed_root": fixed_root,
        }

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belief_state_to_legacy(state: BeliefState[str]) -> LegacyAnimalsBeliefState:
    return LegacyAnimalsBeliefState(
        beliefs=list(state.hypotheses),
        probabilities=list(state.probabilities),
    )


def _belief_state_from_legacy(state: LegacyAnimalsBeliefState) -> BeliefState[str]:
    return BeliefState(
        hypotheses=tuple(state.beliefs),
        probabilities=tuple(float(p) for p in state.probabilities),
    ).renormalized()


def _history_to_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """Convert BEDRunner-style (action, observation) pairs into the message format
    that the legacy ``update_beliefs_batched`` / ``generate_candidate_questions``
    expect: a flat list of role-tagged dicts in chronological order."""
    messages: list[dict[str, str]] = []
    for question, answer in history:
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})
    return messages
