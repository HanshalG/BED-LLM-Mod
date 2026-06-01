"""Abstract Environment interface for BED experiments.

An :class:`Environment` captures everything that varies between BED problems:

- the type of the hidden state (``S``, e.g. ``str`` for the target animal,
  ``ndarray`` for source locations);
- the type of hypothesis the questioner maintains over the hidden state (``H``,
  often the same as ``S`` but kept separate to allow approximate posteriors);
- the type of action the questioner may take (``A``, e.g. a Yes/No question
  string or a query location);
- the type of observation the environment returns (``O``).

Environments are *value providers*: they expose pure-ish methods that, given
the LLM (when needed), the current belief state, and the history, produce
hypotheses, candidate actions, log-likelihoods, posteriors, and metrics.

The BED outer loop in :class:`core.bed_runner.BEDRunner` consumes any
``Environment`` instance and drives a complete experiment without knowing
which environment it is dealing with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

import numpy as np

from .belief import BeliefState
from .experiment_summary import ExperimentSummary

if TYPE_CHECKING:
    from .bed_runner import RunResult


H = TypeVar("H")  # Hypothesis type
A = TypeVar("A")  # Action type
O = TypeVar("O")  # Observation type
S = TypeVar("S")  # Hidden-state type (often same as H, kept separate for clarity)


class Environment(ABC, Generic[S, H, A, O]):
    """Abstract base class for BED environments.

    Modules only need to fill in the methods that are *environment*-
    specific.  Method-specific concerns (how to score actions for EIG vs
    Naive vs StrategyEIG) live in :class:`core.method.Method`.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. ``"animals"``, ``"location_finding"``)."""

    # ------------------------------------------------------------------
    # Hidden state / simulation
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_hidden_state(self, rng: np.random.Generator) -> S:
        """Sample a hidden ground-truth state for a single trial."""

    def sample_hidden_state_for_trial(self, trial_index: int, rng: np.random.Generator) -> S:
        """Sample/choose hidden state for a specific trial index."""
        return self.sample_hidden_state(rng)

    def sample_hidden_states_for_trials(
        self,
        trial_indices: Sequence[int],
        rng: np.random.Generator,
    ) -> list[S]:
        """Sample/choose hidden states for a cross-trial batch."""
        return [
            self.sample_hidden_state_for_trial(trial_index, rng)
            for trial_index in trial_indices
        ]

    @abstractmethod
    def observe(self, action: A, hidden_state: S, rng: np.random.Generator) -> O:
        """Return the (possibly noisy) observation for the given action."""

    def observe_many(
        self,
        actions: Sequence[A],
        hidden_states: Sequence[S],
        rng: np.random.Generator,
    ) -> list[O]:
        """Return observations for a cross-trial batch."""
        if len(actions) != len(hidden_states):
            raise ValueError("actions and hidden_states must have the same length")
        return [
            self.observe(action, hidden_state, rng)
            for action, hidden_state in zip(actions, hidden_states)
        ]

    # ------------------------------------------------------------------
    # Probabilistic model
    # ------------------------------------------------------------------

    @abstractmethod
    def log_prior(self, hypothesis: H) -> float:
        """Log of the (possibly unnormalised) prior over hypotheses."""

    @abstractmethod
    def log_likelihood(self, hypothesis: H, action: A, observation: O) -> float:
        """Log p(observation | action, hypothesis).

        For environments where the likelihood is estimated via an LLM call
        (e.g. animals), modules should provide a batched fast path
        through :meth:`log_likelihood_many`.
        """

    def log_likelihood_many(
        self,
        hypotheses: Sequence[H],
        action: A,
        observation: O,
    ) -> np.ndarray:
        """Vectorised default: loop over :meth:`log_likelihood`.

        Environments should override this when a faster batched module
        exists.  The default keeps the contract correct for new environments
        that haven't bothered yet.
        """
        return np.asarray(
            [self.log_likelihood(h, action, observation) for h in hypotheses],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Belief support: generation and posterior update
    # ------------------------------------------------------------------

    @abstractmethod
    def initial_belief_state(
        self,
        model: Any,  # core.types.Model — Any here to avoid heavy import
        config: Any,
    ) -> BeliefState[H]:
        """Bootstrap a belief state for the start of a trial."""

    def initial_belief_states(
        self,
        trial_indices: Sequence[int],
        model: Any,
        config: Any,
    ) -> list[BeliefState[H]]:
        """Bootstrap belief states for a cross-trial batch."""
        return [self.initial_belief_state(model, config) for _trial_index in trial_indices]

    @abstractmethod
    def update_belief_state(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        model: Any,
        config: Any,
    ) -> BeliefState[H]:
        """Update the belief state given a new history.

        Modules may regenerate hypotheses, score them, prune, etc. —
        anything the environment needs to refresh its finite support.
        """

    def update_belief_states(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        model: Any,
        config: Any,
    ) -> list[BeliefState[H]]:
        """Update belief states for a cross-trial batch."""
        if len(belief_states) != len(histories):
            raise ValueError("belief_states and histories must have the same length")
        return [
            self.update_belief_state(belief_state, history, model, config)
            for belief_state, history in zip(belief_states, histories)
        ]

    # ------------------------------------------------------------------
    # Action proposal
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_candidate_actions(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        model: Any,
        config: Any,
    ) -> list[A]:
        """Propose a list of candidate actions to score for this round."""

    def generate_candidate_actions_many(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        model: Any,
        config: Any,
    ) -> list[list[A]]:
        """Propose candidate actions for a cross-trial batch."""
        if len(belief_states) != len(histories):
            raise ValueError("belief_states and histories must have the same length")
        return [
            list(self.generate_candidate_actions(belief_state, history, model, config))
            for belief_state, history in zip(belief_states, histories)
        ]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @abstractmethod
    def round_metrics(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        hidden_state: S,
    ) -> dict[str, float]:
        """Per-round metrics to log (e.g. accuracy, RMSE, correct mass)."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def validate_config(self, config: Any) -> None:
        """Validate environment-specific config before a run starts."""

    def required_model_roles(self, config: Any) -> tuple[str, ...]:
        """Return ``ModelPair`` roles that this environment needs."""
        return ("questioner", "answerer")

    def trial_count(self, config: Any) -> int:
        """Number of trials to run for this environment."""
        return int(getattr(config, "num_trials", 1) or 1)

    def round_count(self, config: Any) -> int:
        """Number of rounds to run per trial for this environment."""
        return int(getattr(config, "num_rounds", 1) or 1)

    def run_seed(self, config: Any) -> int | None:
        """Seed used by the shared runner for this environment."""
        return getattr(config, "seed", None)

    def configure_for_run(self, config: Any) -> "Environment[S, H, A, O]":
        """Apply environment-specific run setup and return the environment."""
        return self

    def early_stop(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        hidden_state: S,
        latest_observation: O,
    ) -> bool:
        """Return True to terminate the trial early (default: never)."""
        return False

    def generate_naive_action(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> A:
        """Generate the environment's direct naive action.

        Generic :class:`methods.Naive` calls this hook instead of consuming the
        candidate list.  Environments that do not define a task-specific naive
        prompt can leave the default and use candidate-based methods instead.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement generate_naive_action")

    def generate_naive_actions_many(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[A]:
        """Generate direct naive actions for a cross-trial batch."""
        if len(belief_states) != len(histories):
            raise ValueError("belief_states and histories must have the same length")
        return [
            self.generate_naive_action(
                belief_state,
                history,
                model,
                config,
                method_name=method_name,
            )
            for belief_state, history in zip(belief_states, histories)
        ]

    def naive_requires_belief_state(self, method_name: str | None = None) -> bool:
        """Return whether a naive variant needs maintained posterior beliefs.

        The conservative default preserves legacy behavior for environments
        whose naive metrics or prompts still consume posterior state.
        """
        return True

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        hidden_state: S,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        """Optional second naive call after the observation.

        Animals uses this phase to sample/score the current best guess; location
        finding uses it to request a final source estimate and compute RMSE.
        """
        return {}

    def naive_metrics_after_observations(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        hidden_states: Sequence[S],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[dict[str, float]]:
        """Compute naive post-observation metrics for a cross-trial batch."""
        if len(belief_states) != len(histories) or len(histories) != len(hidden_states):
            raise ValueError("belief_states, histories, and hidden_states must have the same length")
        return [
            self.naive_metrics_after_observation(
                belief_state,
                history,
                hidden_state,
                model,
                config,
                method_name=method_name,
            )
            for belief_state, history, hidden_state in zip(belief_states, histories, hidden_states)
        ]

    def representative_observation(self, action: A, predictive_mean: float) -> O:
        """Synthetic observation for depth-2+ continuous EIG branch expansion.

        Used when expanding forward-search branches at each hypothesis's predictive
        mean.  Environments with Gaussian predictive models should implement this;
        discrete/binary environments use :mod:`methods.animals_special` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement representative_observation"
        )

    def belief_after_branch_observation(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        model: Any,
        config: Any,
    ) -> BeliefState[H]:
        """Update beliefs after a forward-search branch observation.

        Defaults to :meth:`update_belief_state`.  Continuous environments may
        override with a cheaper analytical refresh when appropriate.
        """
        return self.update_belief_state(belief_state, history, model, config)

    def choose_strategy_action(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> tuple[A, float, Any]:
        """Choose an action via the environment's StrategyEIG protocol."""
        raise NotImplementedError(f"{type(self).__name__} does not implement choose_strategy_action")

    def choose_strategy_actions_many(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        model: Any,
        config: Any,
        rngs: Sequence[np.random.Generator],
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> list[tuple[A, float, Any]]:
        """Choose StrategyEIG actions for a cross-trial batch."""
        if len(belief_states) != len(histories) or len(histories) != len(rngs):
            raise ValueError("belief_states, histories, and rngs must have the same length")
        return [
            self.choose_strategy_action(
                belief_state,
                history,
                model,
                config,
                rng,
                round_index,
                fixed_root=fixed_root,
            )
            for belief_state, history, rng in zip(belief_states, histories, rngs)
        ]

    def on_empty_candidates(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        round_index: int,
        config: Any,
    ) -> bool:
        """Return True to skip the round when no candidate actions were generated."""
        return False

    def summarize_run(self, run_result: "RunResult", config: Any) -> ExperimentSummary:
        """Convert a :class:`RunResult` into task-level per-round metric series."""
        if not run_result.trials:
            return ExperimentSummary(metrics={})
        max_rounds = max(len(trial.rounds) for trial in run_result.trials)
        accumulator: dict[str, list[list[float]]] = {}
        for trial in run_result.trials:
            for round_idx, round_result in enumerate(trial.rounds):
                for name, value in round_result.metrics.items():
                    series = accumulator.setdefault(name, [[] for _ in range(max_rounds)])
                    series[round_idx].append(float(value))
        metrics = {
            name: [
                (sum(values) / len(values)) if values else float("nan")
                for values in series
            ]
            for name, series in accumulator.items()
        }
        return ExperimentSummary(metrics=metrics)

    def save_artifacts(
        self,
        run_result: "RunResult",
        output_dir: Path,
        config: Any,
    ) -> dict[str, Path]:
        """Write optional per-run artifacts (plots, traces) under ``output_dir``."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
