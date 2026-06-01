"""Location-finding module of :class:`core.Environment`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from pathlib import Path

from environments.location_finding.beliefs import _merge_hypotheses, build_location_belief_state, build_location_posterior, build_location_posteriors_many, sample_location_eig_belief_state
from environments.location_finding.generation import _generate_location_hypotheses_many, choose_location_naive, choose_locations_naive_many, estimate_sources_naive, estimate_sources_naive_many, generate_location_candidates, generate_location_candidates_many, generate_location_hypotheses
from environments.location_finding.physics import _hypothesis_log_prior, _top_source_rmse, observation_log_likelihood, signal_intensity_for_hypothesis, source_rmse
from environments.location_finding.plotting import _plot_location_trial
from environments.location_finding.strategy import choose_location_with_strategy_rollouts, choose_locations_with_strategy_rollouts_many
from environments.location_finding.types import Location, LocationFindingEnv, LocationObservation, LocationStrategyLibrary, SourceConfig, _LocationTrialState
from core import BeliefState, Environment
from core.bed_runner import RunResult
from core.experiment_summary import ExperimentSummary


# Type aliases for clarity.  The location-finding environment uses:
#   S = np.ndarray           — the (num_sources, dim) ground-truth source matrix
#   H = SourceConfig          — a hypothesis is a tuple of source coordinates
#   A = Location              — an action is a single query location
#   O = LocationObservation   — an observation pairs the query with a noisy float


@dataclass
class LocationBEDEnvironment(Environment["np.ndarray", SourceConfig, Location, LocationObservation]):
    """Adapter that exposes the location-finding problem through the BED ABC.

    The configuration object is the flat :class:`helpers.Config`; this adapter
    accepts it directly so existing experiment configs keep working while new
    code uses the typed views in :mod:`core.config`.
    """

    config: Any  # helpers.Config
    rng: np.random.Generator | None = None
    true_theta: np.ndarray | None = None  # optional pinned ground truth (for tests)
    _last_model: Any = field(default=None, repr=False, compare=False)
    _last_location_belief: BeliefState[SourceConfig] | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "location_finding"

    def validate_config(self, config: Any) -> None:
        if config.location_dim != 2:
            raise ValueError("Location Finding currently supports 2D source locations")
        if config.location_noise_sd != 0.5:
            raise ValueError(
                "The initial Location Finding module requires known noise_sd=0.5"
            )

    def trial_count(self, config: Any) -> int:
        return int(config.location_num_trials)

    def round_count(self, config: Any) -> int:
        return int(config.location_num_rounds)

    def run_seed(self, config: Any) -> int | None:
        return getattr(config, "location_seed", None)

    def required_model_roles(self, config: Any) -> tuple[str, ...]:
        return ("questioner",)

    # ------------------------------------------------------------------
    # Hidden state / simulation
    # ------------------------------------------------------------------

    def sample_hidden_state(self, rng: np.random.Generator) -> np.ndarray:
        if self.true_theta is not None:
            return np.asarray(self.true_theta, dtype=float)
        # Standard Normal prior, identical to the original LocationFindingEnv reset path.
        return rng.normal(
            0.0,
            1.0,
            size=(self.config.location_num_sources, self.config.location_dim),
        )

    def observe(
        self,
        action: Location,
        hidden_state: np.ndarray,
        rng: np.random.Generator,
    ) -> LocationObservation:
        # Build a fresh single-shot environment around the supplied ground truth
        # so we don't accumulate state across rounds (the BEDRunner owns the
        # history).  This mirrors LocationFindingEnv.run_experiment exactly.
        env = LocationFindingEnv(
            num_sources=self.config.location_num_sources,
            dim=self.config.location_dim,
            noise_sd=self.config.location_noise_sd,
            true_theta=hidden_state,
            rng=rng,
        )
        return env.run_experiment(action)

    # ------------------------------------------------------------------
    # Probabilistic model
    # ------------------------------------------------------------------

    def log_prior(self, hypothesis: SourceConfig) -> float:
        return _hypothesis_log_prior(hypothesis)

    def log_likelihood(
        self,
        hypothesis: SourceConfig,
        action: Location,
        observation: LocationObservation,
    ) -> float:
        mean = signal_intensity_for_hypothesis(hypothesis, action)
        return observation_log_likelihood(observation.value, mean, self.config.location_noise_sd)

    def log_likelihood_many(
        self,
        hypotheses: Sequence[SourceConfig],
        action: Location,
        observation: LocationObservation,
    ) -> np.ndarray:
        """Vectorised: stack hypotheses into (H, S, D) and compute means in one shot."""
        if not hypotheses:
            return np.empty(0, dtype=float)
        theta = np.asarray(list(hypotheses), dtype=float)  # (H, S, D)
        query = np.asarray(action, dtype=float)             # (D,)
        b, m, alpha = 0.1, 1e-4, 1.0
        distances_sq = np.sum((theta - query[np.newaxis, np.newaxis, :]) ** 2, axis=2)  # (H, S)
        means = b + np.sum(alpha / (m + distances_sq), axis=1)                          # (H,)
        sd = self.config.location_noise_sd
        if observation.value <= 0.0:
            return np.full(len(hypotheses), float("-inf"), dtype=float)
        z = (math.log(observation.value) - np.log(means)) / sd
        return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)

    def predictive_means(
        self,
        hypotheses: Sequence[SourceConfig],
        action: Location,
    ) -> np.ndarray:
        """Return the noiseless signal mean for every hypothesis at ``action``."""
        return np.asarray(
            [signal_intensity_for_hypothesis(hypothesis, action) for hypothesis in hypotheses],
            dtype=float,
        )

    def representative_observation(self, action: Location, predictive_mean: float) -> LocationObservation:
        return LocationObservation(query=action, value=float(predictive_mean))

    def belief_state_for_eig_scoring(
        self,
        belief_state: BeliefState[SourceConfig],
        config: Any,
    ) -> BeliefState[SourceConfig]:
        rng = self.rng if self.rng is not None else np.random.default_rng(
            getattr(config, "location_seed", None)
        )
        sampled, _collapsed = sample_location_eig_belief_state(belief_state, config, rng)
        return sampled

    def score_continuous_forward_search_depth2_batched(
        self,
        belief_state: BeliefState[SourceConfig],
        candidates: Sequence[Location],
        model: Any,
        history: Sequence[tuple[Location, LocationObservation]],
        config: Any,
        *,
        noise_sd: float,
        quadrature_order: int,
        immediate_scores: Sequence[float] | None = None,
    ) -> list[float]:
        from environments.location_finding.depth2_eig import score_location_candidates_depth2_batched

        del noise_sd, quadrature_order
        observations = [observation for _action, observation in history]
        return score_location_candidates_depth2_batched(
            belief_state,
            list(candidates),
            config,
            model,
            observations,
            immediate_scores=immediate_scores,
        )

    # ------------------------------------------------------------------
    # Belief support
    # ------------------------------------------------------------------

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[SourceConfig]:
        self._last_model = model
        hypotheses = generate_location_hypotheses(
            model,
            observations=[],
            belief_state=None,
            config=config,
            label="initial belief generation",
        )
        location_state = build_location_posterior(
            model,
            hypotheses,
            [],
            config,
            label="initial posterior scoring",
        )
        self._last_location_belief = location_state
        return location_state

    def initial_belief_states(
        self,
        trial_indices: Sequence[int],
        model: Any,
        config: Any,
    ) -> list[BeliefState[SourceConfig]]:
        self._last_model = model
        hypotheses_many = _generate_location_hypotheses_many(
            model,
            [[] for _trial_index in trial_indices],
            [None for _trial_index in trial_indices],
            config,
            label="batched initial belief generation",
        )
        states = build_location_posteriors_many(
            model,
            hypotheses_many,
            [[] for _trial_index in trial_indices],
            config,
            label="batched initial posterior scoring",
        )
        if states:
            self._last_location_belief = states[-1]
        return states

    def update_belief_state(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        model: Any,
        config: Any,
    ) -> BeliefState[SourceConfig]:
        observations = [obs for _action, obs in history]
        hypotheses = generate_location_hypotheses(
            model,
            observations,
            belief_state,
            config,
            label="belief refresh",
        )
        merged = _merge_hypotheses(belief_state, hypotheses)
        location_state = build_location_posterior(
            model,
            merged,
            observations,
            config,
            label="posterior scoring",
        )
        self._last_location_belief = location_state
        return location_state

    def update_belief_states(
        self,
        belief_states: Sequence[BeliefState[SourceConfig]],
        histories: Sequence[Sequence[tuple[Location, LocationObservation]]],
        model: Any,
        config: Any,
    ) -> list[BeliefState[SourceConfig]]:
        observations_many = [[obs for _action, obs in history] for history in histories]
        generated_many = _generate_location_hypotheses_many(
            model,
            observations_many,
            list(belief_states),
            config,
            label="batched belief refresh",
        )
        merged_many = [
            _merge_hypotheses(belief_state, generated)
            for belief_state, generated in zip(belief_states, generated_many)
        ]
        states = build_location_posteriors_many(
            model,
            merged_many,
            observations_many,
            config,
            context_states=list(belief_states),
            label="batched posterior scoring",
        )
        if states:
            self._last_location_belief = states[-1]
        return states

    def belief_after_branch_observation(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        model: Any,
        config: Any,
    ) -> BeliefState[SourceConfig]:
        if getattr(config, "location_posterior_mode", None) == "analytical_likelihood":
            observations = [obs for _action, obs in history]
            location_state = build_location_belief_state(
                list(belief_state.hypotheses),
                observations,
                config,
            )
            self._last_location_belief = location_state
            return location_state
        return self.update_belief_state(belief_state, history, model, config)

    # ------------------------------------------------------------------
    # Action proposal
    # ------------------------------------------------------------------

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        model: Any,
        config: Any,
    ) -> list[Location]:
        observations = [obs for _action, obs in history]
        return generate_location_candidates(model, belief_state, observations, config)

    def generate_candidate_actions_many(
        self,
        belief_states: Sequence[BeliefState[SourceConfig]],
        histories: Sequence[Sequence[tuple[Location, LocationObservation]]],
        model: Any,
        config: Any,
    ) -> list[list[Location]]:
        observations_many = [[obs for _action, obs in history] for history in histories]
        return generate_location_candidates_many(model, list(belief_states), observations_many, config)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def round_metrics(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        hidden_state: np.ndarray,
    ) -> dict[str, float]:
        rmse = _top_source_rmse(belief_state, hidden_state)
        top = belief_state.top()
        top_probability = top[1] if top is not None else 0.0
        return {
            "source_rmse": rmse,
            "top_probability": float(top_probability),
            "support_size": float(belief_state.support_size),
            "ess": belief_state.effective_sample_size(),
        }

    def generate_naive_action(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> Location:
        location_belief = (
            belief_state
            if method_name == "naive+belief"
            else None
        )
        location = choose_location_naive(
            model,
            [obs for _action, obs in history],
            config,
            belief_state=location_belief,
        )
        if location is None:
            raise ValueError("Location naive method could not produce a valid location")
        return location

    def generate_naive_actions_many(
        self,
        belief_states: Sequence[BeliefState[SourceConfig]],
        histories: Sequence[Sequence[tuple[Location, LocationObservation]]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[Location]:
        observations_many = [[obs for _action, obs in history] for history in histories]
        prompt_beliefs = [
            belief_state if method_name == "naive+belief" else None
            for belief_state in belief_states
        ]
        locations = choose_locations_naive_many(
            model,
            observations_many,
            config,
            belief_states=prompt_beliefs,
        )
        if any(location is None for location in locations):
            raise ValueError("Location naive method could not produce a valid location")
        return [location for location in locations if location is not None]

    def naive_requires_belief_state(self, method_name: str | None = None) -> bool:
        return method_name == "naive+belief"

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        hidden_state: np.ndarray,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        observations = [obs for _action, obs in history]
        estimate = estimate_sources_naive(model, observations, config)
        return {
            "source_rmse": source_rmse(estimate, hidden_state),
            "top_probability": 1.0,
        }

    def naive_metrics_after_observations(
        self,
        belief_states: Sequence[BeliefState[SourceConfig]],
        histories: Sequence[Sequence[tuple[Location, LocationObservation]]],
        hidden_states: Sequence[np.ndarray],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> list[dict[str, float]]:
        del belief_states, method_name
        observations_many = [[obs for _action, obs in history] for history in histories]
        estimates = estimate_sources_naive_many(model, observations_many, config)
        return [
            {
                "source_rmse": source_rmse(estimate, hidden_state),
                "top_probability": 1.0,
            }
            for estimate, hidden_state in zip(estimates, hidden_states)
        ]

    def choose_strategy_action(
        self,
        belief_state: BeliefState[SourceConfig],
        history: Sequence[tuple[Location, LocationObservation]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> tuple[Location, float, Any]:
        library = getattr(self, "_strategy_library", None)
        if library is None:
            library = LocationStrategyLibrary()
            self._strategy_library = library
        location, score, evaluation = choose_location_with_strategy_rollouts(
            model,
            belief_state,
            [obs for _action, obs in history],
            library,
            config,
            rng,
            round_index,
            fixed_root=fixed_root,
        )
        if location is None:
            raise ValueError("Location StrategyEIG could not produce a valid location")
        return location, float(score), evaluation

    def choose_strategy_actions_many(
        self,
        belief_states: Sequence[BeliefState[SourceConfig]],
        histories: Sequence[Sequence[tuple[Location, LocationObservation]]],
        model: Any,
        config: Any,
        rngs: Sequence[np.random.Generator],
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> list[tuple[Location, float, Any]]:
        libraries = getattr(self, "_strategy_libraries_by_history", None)
        if libraries is None:
            libraries = {}
            self._strategy_libraries_by_history = libraries
        states: list[_LocationTrialState] = []
        for idx, (belief_state, history, rng) in enumerate(zip(belief_states, histories, rngs)):
            history_key = id(history)
            library = libraries.get(history_key)
            if library is None:
                library = LocationStrategyLibrary()
                libraries[history_key] = library
            state = _LocationTrialState(
                trial_idx=idx,
                env=LocationFindingEnv(
                    num_sources=config.location_num_sources,
                    dim=config.location_dim,
                    noise_sd=config.location_noise_sd,
                    rng=rng,
                ),
                observations=[obs for _action, obs in history],
                rng=rng,
                belief_state=belief_state,
                strategy_library=library,
            )
            states.append(state)
        results = choose_locations_with_strategy_rollouts_many(
            model,
            states,
            config,
            round_index,
            fixed_root=fixed_root,
        )
        if any(location is None for location, _score, _evaluation in results):
            raise ValueError("Location StrategyEIG could not produce a valid location")
        return [
            (location, float(score), evaluation)
            for location, score, evaluation in results
            if location is not None
        ]

    def summarize_run(self, run_result: RunResult, config: Any) -> ExperimentSummary:
        base = super().summarize_run(run_result, config)
        metrics = dict(base.metrics)
        if "selected_eig" not in metrics:
            metrics["selected_eig"] = [0.0] * max(
                (len(series) for series in metrics.values()),
                default=0,
            )
        return ExperimentSummary(
            metrics={
                "source_rmse": metrics.get("source_rmse", []),
                "top_probability": metrics.get("top_probability", []),
                "selected_eig": metrics.get("selected_eig", []),
            },
            logs=base.logs,
        )

    def save_artifacts(
        self,
        run_result: RunResult,
        output_dir: Path,
        config: Any,
    ) -> dict[str, Path]:
        if not getattr(config, "location_plot_trials", False):
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._last_model
        artifacts: dict[str, Path] = {}
        for trial in run_result.trials:
            if not trial.rounds:
                continue
            observations = [round_result.observation for round_result in trial.rounds]
            env = LocationFindingEnv(
                num_sources=config.location_num_sources,
                dim=config.location_dim,
                noise_sd=config.location_noise_sd,
                true_theta=trial.hidden_state,
                rng=np.random.default_rng(0),
            )
            final_metrics = trial.rounds[-1].metrics
            final_rmse = float(final_metrics.get("source_rmse", float("inf")))
            top_probability = float(final_metrics.get("top_probability", 0.0))
            belief_state = (
                trial.final_belief_state if trial.final_belief_state is not None else self._last_location_belief
            )
            if belief_state is None or not belief_state.hypotheses:
                estimate = None
                if model is not None and observations:
                    estimate = estimate_sources_naive(model, observations, config)
                belief_state = BeliefState(
                    hypotheses=[] if estimate is None else [estimate],
                    probabilities=[] if estimate is None else [1.0],
                )
            plot_path = output_dir / f"location_trial_{trial.trial_index + 1:03d}.png"
            _plot_location_trial(
                env,
                observations,
                belief_state,
                trial.trial_index,
                final_rmse,
                top_probability,
                plot_path,
            )
            artifacts[f"location_trial_plot_{trial.trial_index + 1:03d}"] = plot_path
        return artifacts
