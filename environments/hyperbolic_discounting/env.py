"""Thin :class:`core.Environment` adapter for hyperbolic temporal discounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from core.bed_runner import RunResult
from core.experiment_summary import ExperimentSummary
from environments.hyperbolic_discounting import runner as _htd
from helpers import Config


@dataclass
class HyperbolicBEDEnvironment(
    Environment[_htd.HyperbolicParams, _htd.HyperbolicParams, _htd.HyperbolicDesign, _htd.HyperbolicObservation]
):
    config: Config
    rng: np.random.Generator | None = None
    true_theta: _htd.HyperbolicParams | None = None
    _last_model: Any = field(default=None, repr=False, compare=False)
    _last_belief: _htd.HyperbolicBeliefState | None = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "hyperbolic_discounting"

    def sample_hidden_state(self, rng: np.random.Generator) -> _htd.HyperbolicParams:
        if self.true_theta is not None:
            return self.true_theta
        env = _htd.HyperbolicDiscountingEnv(
            noise_sd=self.config.htd_noise_sd,
            rng=rng,
            k_mean=self.config.htd_k_mean,
            k_std=self.config.htd_k_std,
            alpha_scale=self.config.htd_alpha_scale,
        )
        return env.true_params

    def observe(
        self,
        action: _htd.HyperbolicDesign,
        hidden_state: _htd.HyperbolicParams,
        rng: np.random.Generator,
    ) -> _htd.HyperbolicObservation:
        env = _htd.HyperbolicDiscountingEnv(
            noise_sd=self.config.htd_noise_sd,
            true_params=hidden_state,
            rng=rng,
            k_mean=self.config.htd_k_mean,
            k_std=self.config.htd_k_std,
            alpha_scale=self.config.htd_alpha_scale,
        )
        return env.run_experiment(action)

    def log_prior(self, hypothesis: _htd.HyperbolicParams) -> float:
        return _htd.log_prior(hypothesis, self.config)

    def log_likelihood(
        self,
        hypothesis: _htd.HyperbolicParams,
        action: _htd.HyperbolicDesign,
        observation: _htd.HyperbolicObservation,
    ) -> float:
        mean = _htd.latent_mean(action, hypothesis)
        return _htd._log_normal_pdf(observation.value, mean, self.config.htd_noise_sd)

    def log_likelihood_many(
        self,
        hypotheses: Sequence[_htd.HyperbolicParams],
        action: _htd.HyperbolicDesign,
        observation: _htd.HyperbolicObservation,
    ) -> np.ndarray:
        if not hypotheses:
            return np.empty(0, dtype=float)
        means = np.asarray([_htd.latent_mean(action, hypothesis) for hypothesis in hypotheses], dtype=float)
        sd = self.config.htd_noise_sd
        z = (observation.value - means) / sd
        return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)

    def predictive_means(
        self,
        hypotheses: Sequence[_htd.HyperbolicParams],
        action: _htd.HyperbolicDesign,
    ) -> np.ndarray:
        return np.asarray([_htd.latent_mean(action, hypothesis) for hypothesis in hypotheses], dtype=float)

    def representative_observation(
        self,
        action: _htd.HyperbolicDesign,
        predictive_mean: float,
    ) -> _htd.HyperbolicObservation:
        return _htd.HyperbolicObservation(design=action, value=float(predictive_mean))

    def belief_state_for_eig_scoring(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        config: Any,
    ) -> BeliefState[_htd.HyperbolicParams]:
        hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
        rng = self.rng if self.rng is not None else np.random.default_rng(getattr(config, "htd_seed", None))
        sampled, _collapsed = _htd.sample_hyperbolic_eig_belief_state(hyperbolic_belief, config, rng)
        return _belief_state_from_hyperbolic(sampled)

    def score_continuous_forward_search_depth2_batched(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        candidates: Sequence[_htd.HyperbolicDesign],
        model: Any,
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        config: Any,
        *,
        noise_sd: float,
        quadrature_order: int,
        immediate_scores: Sequence[float] | None = None,
    ) -> list[float]:
        from environments.hyperbolic_discounting.depth2_eig import score_hyperbolic_candidates_depth2_batched

        del noise_sd, quadrature_order
        observations = [observation for _design, observation in history]
        return score_hyperbolic_candidates_depth2_batched(
            _belief_state_to_hyperbolic(belief_state),
            list(candidates),
            config,
            model,
            observations,
            immediate_scores=immediate_scores,
        )

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[_htd.HyperbolicParams]:
        self._last_model = model
        hypotheses = _htd.generate_hyperbolic_hypotheses(
            model,
            observations=[],
            belief_state=None,
            config=config,
            label="initial belief generation",
            rng=self.rng,
        )
        if not hypotheses:
            seed_rng = self.rng or np.random.default_rng(config.htd_seed)
            hypotheses = _htd._seed_hyperbolic_hypotheses(config, seed_rng)
        state = _htd.build_hyperbolic_posterior(
            model,
            hypotheses,
            [],
            config,
            label="initial posterior scoring",
        )
        self._last_belief = state
        return _belief_state_from_hyperbolic(state)

    def update_belief_state(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        model: Any,
        config: Any,
    ) -> BeliefState[_htd.HyperbolicParams]:
        observations = [observation for _design, observation in history]
        hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
        hypotheses = _htd.generate_hyperbolic_hypotheses(
            model,
            observations,
            hyperbolic_belief,
            config,
            label="belief refresh",
            rng=self.rng,
        )
        merged = _htd._merge_hypotheses(hyperbolic_belief, hypotheses)
        state = _htd.build_hyperbolic_posterior(
            model,
            merged,
            observations,
            config,
            label="posterior scoring",
        )
        self._last_belief = state
        return _belief_state_from_hyperbolic(state)

    def belief_after_branch_observation(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        model: Any,
        config: Any,
    ) -> BeliefState[_htd.HyperbolicParams]:
        if getattr(config, "htd_posterior_mode", None) == "analytical_likelihood":
            observations = [observation for _design, observation in history]
            hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
            state = _htd.build_hyperbolic_belief_state(list(hyperbolic_belief.hypotheses), observations, config)
            self._last_belief = state
            return _belief_state_from_hyperbolic(state)
        return self.update_belief_state(belief_state, history, model, config)

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        model: Any,
        config: Any,
    ) -> list[_htd.HyperbolicDesign]:
        if model is None:
            return _htd.eval_holdout_designs(config)[: config.htd_target_num_candidates]
        hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
        observations = [observation for _design, observation in history]
        return _htd.generate_hyperbolic_candidates(model, hyperbolic_belief, observations, config)

    def round_metrics(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        hidden_state: _htd.HyperbolicParams,
    ) -> dict[str, float]:
        del history
        hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
        rmse = _htd._top_parameter_rmse(hyperbolic_belief, hidden_state)
        top = belief_state.top()
        top_probability = top[1] if top is not None else 0.0
        holdout_designs = _htd.eval_holdout_designs(self.config)
        estimate = hyperbolic_belief.hypotheses[0] if hyperbolic_belief.hypotheses else None
        implied_acc = (
            _htd.implied_choice_accuracy(estimate, hidden_state, holdout_designs)
            if estimate is not None
            else 0.0
        )
        return {
            "parameter_rmse": rmse,
            "k_rmse": _htd.k_rmse(hyperbolic_belief.hypotheses[0], hidden_state) if hyperbolic_belief.hypotheses else float("inf"),
            "top_probability": float(top_probability),
            "implied_choice_accuracy": float(implied_acc),
            "support_size": float(belief_state.support_size),
            "ess": belief_state.effective_sample_size(),
        }

    def generate_naive_action(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> _htd.HyperbolicDesign:
        del belief_state, method_name
        design = _htd.choose_design_naive(model, [observation for _design, observation in history], config)
        if design is None:
            rng = self.rng or np.random.default_rng(config.htd_seed)
            return _htd.normalize_design(
                rng.uniform(*config.htd_ir_bounds),
                rng.uniform(*config.htd_dr_bounds),
                int(rng.integers(config.htd_days_bounds[0], config.htd_days_bounds[1] + 1)),
            )
        return design

    def naive_metrics_after_observation(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        hidden_state: _htd.HyperbolicParams,
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> dict[str, float]:
        del belief_state, method_name
        observations = [observation for _design, observation in history]
        estimate = _htd.estimate_params_naive(model, observations, config)
        return {
            "parameter_rmse": _htd.parameter_rmse(estimate, hidden_state),
            "k_rmse": _htd.k_rmse(estimate, hidden_state),
            "top_probability": 1.0,
        }

    def choose_strategy_action(
        self,
        belief_state: BeliefState[_htd.HyperbolicParams],
        history: Sequence[tuple[_htd.HyperbolicDesign, _htd.HyperbolicObservation]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> tuple[_htd.HyperbolicDesign, float, Any]:
        del rng, round_index, fixed_root
        hyperbolic_belief = _belief_state_to_hyperbolic(belief_state)
        observations = [observation for _design, observation in history]
        design, score = _htd.choose_strategy_design(model, hyperbolic_belief, observations, config)
        if design is None:
            raise ValueError("Hyperbolic StrategyEIG could not produce a valid design")
        return design, score, None

    def summarize_run(self, run_result: RunResult, config: Any) -> ExperimentSummary:
        base = super().summarize_run(run_result, config)
        metrics = dict(base.metrics)
        if "selected_eig" not in metrics:
            metrics["selected_eig"] = [0.0] * max((len(series) for series in metrics.values()), default=0)
        if "implied_choice_accuracy" not in metrics:
            metrics["implied_choice_accuracy"] = [0.0] * max((len(series) for series in metrics.values()), default=0)
        return ExperimentSummary(
            metrics={
                "parameter_rmse": metrics.get("parameter_rmse", []),
                "k_rmse": metrics.get("k_rmse", []),
                "top_probability": metrics.get("top_probability", []),
                "selected_eig": metrics.get("selected_eig", []),
                "implied_choice_accuracy": metrics.get("implied_choice_accuracy", []),
            },
            logs=base.logs,
        )

    def run_batched_experiment(
        self,
        questioner: Any,
        method: Any,
        config: Any,
        *,
        output_dir: Path | None = None,
        rng: np.random.Generator | None = None,
    ) -> ExperimentSummary:
        from environments.hyperbolic_discounting.batched_trials import run_hyperbolic_trials_batched

        metrics = run_hyperbolic_trials_batched(
            questioner,
            config,
            rng=rng or self.rng,
            output_dir=output_dir,
            method_name=method.name,
        )
        return ExperimentSummary(
            metrics={
                "parameter_rmse": list(metrics.parameter_rmse),
                "k_rmse": list(metrics.k_rmse),
                "top_probability": list(metrics.top_probability),
                "selected_eig": list(metrics.selected_eig),
                "implied_choice_accuracy": list(metrics.implied_choice_accuracy),
            },
            logs=[],
        )


def _belief_state_to_hyperbolic(state: BeliefState[_htd.HyperbolicParams]) -> _htd.HyperbolicBeliefState:
    return _htd.HyperbolicBeliefState(
        hypotheses=list(state.hypotheses),
        probabilities=list(state.probabilities),
    )


def _belief_state_from_hyperbolic(state: _htd.HyperbolicBeliefState) -> BeliefState[_htd.HyperbolicParams]:
    return BeliefState(
        hypotheses=tuple(state.hypotheses),
        probabilities=tuple(float(probability) for probability in state.probabilities),
    )
